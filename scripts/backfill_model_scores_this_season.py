"""
Backfill DC+Elo ensemble predictions for an entire season
into the unified DB.predictions table, using DB-only data.

Option C:
- Reuse the same internals as src/predict/predict_model.py
- Train only on matches *before* each matchday
- Predict for that matchday's fixtures
- Insert rows into predictions table for later evaluation
"""

import sys
import sqlite3
from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd

# =====================================================================
# ABSOLUTE PATH FIX (guaranteed to work on Windows)
# =====================================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../soccer_agent_local/scripts
ROOT = SCRIPT_DIR.parent                              # .../soccer_agent_local
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

# =====================================================================
# Imports from your internal codebase
# =====================================================================
from src.predict.predict_model import load_config, sample_score_from_xg
from predictor import train_models, predict_fixtures
from db import init_db, insert_predictions


DB_PATH = ROOT / "data" / "soccer_agent.db"


# =====================================================================
# Argument Parsing
# =====================================================================
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Backfill DC+Elo ensemble predictions into DB.predictions "
            "for an entire season, using DB-only data."
        )
    )
    parser.add_argument(
        "--season-start",
        type=str,
        default="2024-08-01",
        help="Start date (YYYY-MM-DD) for backfill window.",
    )
    parser.add_argument(
        "--season-end",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD). If omitted, uses today.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=60,
        help="Minimum number of historical matches required before predicting.",
    )

    return parser.parse_args()


# =====================================================================
# DB Helpers
# =====================================================================
def load_results_up_to(cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load historical results *before cutoff_date* from DB.results.

    Returns DataFrame with: [Date, HomeTeam, AwayTeam, FTHG, FTAG]
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT date, home_team, away_team, FTHG, FTAG
        FROM results
        WHERE date < ?
        """,
        conn,
        params=(cutoff_date.strftime("%Y-%m-%d"),),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df.rename(
        columns={
            "home_team": "HomeTeam",
            "away_team": "AwayTeam",
        },
        inplace=True,
    )

    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    return df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]


def load_matchday_fixtures(match_date: pd.Timestamp) -> pd.DataFrame:
    """
    Treat DB.results as canonical fixtures list for that matchday.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT date, home_team, away_team
        FROM results
        WHERE date = ?
        ORDER BY home_team, away_team
        """,
        conn,
        params=(match_date.strftime("%Y-%m-%d"),),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame(columns=["Date", "HomeTeam", "AwayTeam"])

    df["Date"] = pd.to_datetime(df["date"], errors="coerce")
    df.rename(
        columns={
            "home_team": "HomeTeam",
            "away_team": "AwayTeam",
        },
        inplace=True,
    )

    return df[["Date", "HomeTeam", "AwayTeam"]]


def get_matchdays_in_window(season_start: str, season_end: str | None) -> list[pd.Timestamp]:
    """
    Return sorted matchdays from DB.results in [season_start, season_end].
    """
    start_ts = pd.to_datetime(season_start)
    end_ts = pd.to_datetime(season_end) if season_end else pd.Timestamp(date.today())

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT DISTINCT date
        FROM results
        WHERE date >= ?
          AND date <= ?
        ORDER BY date
        """,
        conn,
        params=(
            start_ts.strftime("%Y-%m-%d"),
            end_ts.strftime("%Y-%m-%d"),
        ),
    )
    conn.close()

    if df.empty:
        return []

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return sorted(df["date"].unique())


# =====================================================================
# Core Backfill Loop
# =====================================================================
def backfill_model_predictions_for_season(
    season_start: str,
    season_end: str | None,
    min_matches: int,
):

    cfg = load_config()
    model_cfg = cfg.get("model", {})
    dc_cfg = model_cfg.get("dc", {})
    elo_cfg = model_cfg.get("elo", {})
    ensemble_cfg = model_cfg.get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    matchdays = get_matchdays_in_window(season_start, season_end)

    if not matchdays:
        print("No matchdays found in this date window.")
        return

    total_inserted = 0
    skipped_small = 0

    print("======================================================")
    print(" Backfilling DC+Elo Predictions")
    print("======================================================")
    print(f" Season: {season_start} → {season_end or date.today()}")
    print(f" Matchdays: {len(matchdays)}")
    print("======================================================\n")

    for idx, md in enumerate(matchdays, 1):
        md_ts = pd.to_datetime(md)

        print(f"[{idx}/{len(matchdays)}] Matchday: {md_ts.date()}")

        # 1. Load training data
        hist_df = load_results_up_to(md_ts)
        n_hist = len(hist_df)

        if n_hist < min_matches:
            print(f"  - Skipping: only {n_hist} matches, need {min_matches}")
            skipped_small += 1
            continue

        print(f"  - Training on {n_hist} matches")
        dc_model, elo_model = train_models(hist_df, dc_cfg, elo_cfg)

        # 2. Load fixtures for this matchday
        fixtures = load_matchday_fixtures(md_ts)
        if fixtures.empty:
            print("  - No fixtures found for this date.")
            continue

        print(f"  - Predicting {len(fixtures)} fixtures")
        preds_df = predict_fixtures(
            fixtures,
            dc_model,
            elo_model,
            w_dc=w_dc,
            w_elo=w_elo,
        )

        if preds_df.empty:
            print("  - No predictions produced")
            continue

        # 3. Insert to DB
        model_version = f"dc_elo_backfill_{md_ts.strftime('%Y%m%d')}"
        inserted_today = 0

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

            score_pred = sample_score_from_xg(lamH, lamA)

            insert_predictions(
                {
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
                    "score_pred": score_pred,
                    "chatgpt_pred": None,
                }
            )

            inserted_today += 1

        total_inserted += inserted_today
        print(f"  - Inserted {inserted_today} predictions\n")

    print("======================================================")
    print(" Backfill Complete")
    print(f" Matchdays processed: {len(matchdays)}")
    print(f" Skipped (small window): {skipped_small}")
    print(f" Predictions inserted: {total_inserted}")
    print("======================================================")


# =====================================================================
# Entrypoint
# =====================================================================
def main():
    init_db()
    args = parse_args()
    backfill_model_predictions_for_season(
        season_start=args.season_start,
        season_end=args.season_end,
        min_matches=args.min_matches,
    )


if __name__ == "__main__":
    main()
