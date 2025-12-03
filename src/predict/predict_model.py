import sys
import sqlite3
from pathlib import Path
from datetime import datetime, date
import math

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

    today = pd.Timestamp(date.today())
    df = df[df["Date"] >= today].copy()

    return df


# ------------------------------------------------------------
# POISSON HELPERS (DETERMINISTIC SCORE)
# ------------------------------------------------------------
def poisson_pmf(k: int, lam: float) -> float:
    """Simple Poisson PMF without external deps."""
    if lam < 0 or not math.isfinite(lam) or k < 0:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0


def most_likely_score_from_xg(lamH: float, lamA: float, max_goals: int = 6) -> str:
    """
    Deterministic: pick the most likely (home_goals, away_goals) pair
    under independent Poisson(lamH) and Poisson(lamA), capped at max_goals.
    This replaces random sampling and avoids silly 7-0 type tails.
    """
    # Basic sanity / fallback
    if not np.isfinite(lamH) or lamH < 0:
        lamH = 1.0
    if not np.isfinite(lamA) or lamA < 0:
        lamA = 1.0

    best_p = -1.0
    best_h = 0
    best_a = 0

    # Precompute PMFs for efficiency
    pH_vals = [poisson_pmf(h, lamH) for h in range(max_goals + 1)]
    pA_vals = [poisson_pmf(a, lamA) for a in range(max_goals + 1)]

    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = pH_vals[h] * pA_vals[a]
            if p > best_p:
                best_p = p
                best_h = h
                best_a = a

    return f"{best_h}-{best_a}"


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
    cfg = load_config()

    results_hist = load_training_results_from_db()

    model_cfg = cfg.get("model", {})
    dc_cfg = model_cfg.get("dc", {})
    elo_cfg = model_cfg.get("elo", {})
    ensemble_cfg = model_cfg.get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    dc_model, elo_model = train_models(results_hist, dc_cfg, elo_cfg)

    fixtures_db = get_upcoming_fixtures(days_ahead=days_ahead)
    if fixtures_db.empty:
        print("No upcoming fixtures found in DB for the given window.")
        return pd.DataFrame()

    fixtures_for_model = prepare_fixtures_for_model(fixtures_db)
    if fixtures_for_model.empty:
        print("No valid fixtures after cleaning (dates/team names).")
        return pd.DataFrame()

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

        # NEW: deterministic, most likely score rather than random sample
        score_pred = most_likely_score_from_xg(lamH, lamA, max_goals=6)

        row_dict = {
            "date": date_str,
            "home_team": home,
            "away_team": away,
            "model_version": model_version,
            "dixon_coles_probs": "",
            "elo_probs": "",
            "ensemble_probs": "",
            "home_win_prob": pH,
            "draw_prob": pD,
            "away_win_prob": pA,
            "exp_goals_home": lamH,
            "exp_goals_away": lamA,
            "exp_total_goals": lamT,
            "score_pred": score_pred,   # model's deterministic correct score
            "chatgpt_pred": None,       # ChatGPT fills this later
        }

        insert_predictions(row_dict)

    return preds_df


# ------------------------------------------------------------
# CLI ENTRYPOINT
# ------------------------------------------------------------
def main():
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
