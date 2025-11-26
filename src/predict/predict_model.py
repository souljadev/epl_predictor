import sys
import sqlite3
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import yaml

# ------------------------------------------------------------
# PATHS / IMPORTS
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from predictor import train_models, predict_fixtures  # noqa: E402
from db import (  # noqa: E402
    init_db,
    get_upcoming_fixtures,
    insert_predictions,
)

DB_PATH = ROOT / "data" / "soccer_agent.db"


# ------------------------------------------------------------
# CONFIG HELPERS
# ------------------------------------------------------------
def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ------------------------------------------------------------
# TRAINING DATA FROM DB
# ------------------------------------------------------------
def load_training_results_from_db() -> pd.DataFrame:
    """
    Load historical results from SQLite for model training.
    Uses DB.results as the single source of truth.

    Expects columns: date, home_team, away_team, FTHG, FTAG
    Returns: DataFrame with columns [Date, HomeTeam, AwayTeam, FTHG, FTAG]
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT date, home_team, away_team, FTHG, FTAG
        FROM results
        """,
        conn,
    )
    conn.close()

    if df.empty:
        raise ValueError("No rows found in results table; cannot train model.")

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Only use completed matches (defensive, but results should already be only completed)
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()

    df.rename(
        columns={
            "home_team": "HomeTeam",
            "away_team": "AwayTeam",
        },
        inplace=True,
    )

    return df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]


# ------------------------------------------------------------
# FIXTURES FROM DB
# ------------------------------------------------------------
def prepare_fixtures_for_model(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DB fixtures (date, home_team, away_team) into the shape
    expected by predict_fixtures:
      - Date (datetime64)
      - HomeTeam
      - AwayTeam
    """
    if fixtures_df.empty:
        return fixtures_df

    rename_map = {}
    if "date" in fixtures_df.columns:
        rename_map["date"] = "Date"
    if "home_team" in fixtures_df.columns:
        rename_map["home_team"] = "HomeTeam"
    if "away_team" in fixtures_df.columns:
        rename_map["away_team"] = "AwayTeam"

    df = fixtures_df.rename(columns=rename_map).copy()

    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"get_upcoming_fixtures must provide columns: {required} (missing: {missing})")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Safety: only keep fixtures today or later
    today = pd.Timestamp(date.today())
    df = df[df["Date"] >= today].copy()

    return df


# ------------------------------------------------------------
# SCORE SAMPLING (Poisson-based, balanced option B)
# ------------------------------------------------------------
def sample_score_from_xg(lamH: float, lamA: float) -> str:
    """
    Sample a plausible scoreline from expected goals using independent
    Poisson draws for home and away goals.

    If lambdas are invalid, fall back to rounded xG.
    """
    try:
        if lamH < 0 or lamA < 0 or not np.isfinite(lamH) or not np.isfinite(lamA):
            raise ValueError
        h = np.random.poisson(lamH)
        a = np.random.poisson(lamA)
    except Exception:
        h = int(round(lamH if np.isfinite(lamH) else 1.0))
        a = int(round(lamA if np.isfinite(lamA) else 1.0))
    return f"{h}-{a}"


# ------------------------------------------------------------
# ARG PARSING
# ------------------------------------------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Run DC+Elo model predictions for upcoming fixtures (DB-only).")
    parser.add_argument(
        "days_ahead",
        nargs="?",
        type=int,
        default=7,
        help="Number of days ahead from today to include fixtures (default: 7).",
    )
    parser.add_argument(
        "run_id",
        nargs="?",
        type=str,
        default=None,
        help="Optional run_id label (default: current UTC timestamp).",
    )

    args = parser.parse_args()
    days_ahead = args.days_ahead
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return days_ahead, run_id


# ------------------------------------------------------------
# CORE PIPELINE
# ------------------------------------------------------------
def run_model_predictions(days_ahead: int = 7, run_id: str | None = None) -> pd.DataFrame:
    """
    1) Load config
    2) Load historical results from DB.results
    3) Train DC + Elo models
    4) Fetch upcoming fixtures from DB.fixtures
    5) Predict probs + xG via predict_fixtures
    6) Upsert predictions into unified DB.predictions
    """
    cfg = load_config()

    # 1) Training data (DB-only)
    results_hist = load_training_results_from_db()

    # 2) Model configuration
    model_cfg = cfg.get("model", {})
    dc_cfg = model_cfg.get("dc", {})
    elo_cfg = model_cfg.get("elo", {})
    ensemble_cfg = model_cfg.get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    # 3) Train models
    dc_model, elo_model = train_models(results_hist, dc_cfg, elo_cfg)

    # 4) Upcoming fixtures from DB
    fixtures_db = get_upcoming_fixtures(days_ahead=days_ahead)
    if fixtures_db.empty:
        print("No upcoming fixtures found in DB for the given window.")
        return pd.DataFrame()

    fixtures_for_model = prepare_fixtures_for_model(fixtures_db)

    if fixtures_for_model.empty:
        print("No valid fixtures after cleaning (dates/team names).")
        return pd.DataFrame()

    # 5) Predict using DC+Elo ensemble
    preds_df = predict_fixtures(
        fixtures_for_model,
        dc_model,
        elo_model,
        w_dc=w_dc,
        w_elo=w_elo,
    )

    if preds_df.empty:
        print("predict_fixtures returned no rows.")
        return pd.DataFrame()

    # 6) Upsert into unified predictions table
    model_version = f"dc_elo_ensemble_live_{datetime.utcnow().strftime('%Y%m%d')}"

    for _, row in preds_df.iterrows():
        date_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
        home = str(row["HomeTeam"])
        away = str(row["AwayTeam"])

        pH = float(row["pH"])
        pD = float(row["pD"])
        pA = float(row["pA"])
        lamH = float(row["ExpHomeGoals"])
        lamA = float(row["ExpAwayGoals"])
        lamT = float(row.get("ExpTotalGoals", lamH + lamA))

        # Balanced Option B: Poisson-based score sampling from xG
        score_pred = sample_score_from_xg(lamH, lamA)

        row_dict = {
            "date": date_str,
            "home_team": home,
            "away_team": away,
            "model_version": model_version,
            # Keeping these as empty strings for now; can store JSON later if needed
            "dixon_coles_probs": "",
            "elo_probs": "",
            "ensemble_probs": "",
            "home_win_prob": pH,
            "draw_prob": pD,
            "away_win_prob": pA,
            "exp_goals_home": lamH,
            "exp_goals_away": lamA,
            "exp_total_goals": lamT,
            "score_pred": score_pred,
            # ChatGPT fills this later
            "chatgpt_pred": None,
        }

        insert_predictions(row_dict)

    return preds_df


# ------------------------------------------------------------
# CLI ENTRYPOINT
# ------------------------------------------------------------
def main():
    # Ensure DB exists & schema is up to date
    init_db()

    days_ahead, run_id = parse_args()

    print("====================================================")
    print("Starting live model predictions from DB fixtures...")
    print(f"Days ahead window: 0–{days_ahead}")
    print(f"Assigned run_id: {run_id}")
    print("====================================================\n")

    preds_df = run_model_predictions(days_ahead=days_ahead, run_id=run_id)

    print(f"\nPredictions complete. {len(preds_df)} rows written to DB.\n")


if __name__ == "__main__":
    main()
