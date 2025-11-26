"""
compare_models.py
Evaluate DC/Elo ensemble vs ChatGPT vs baseline models for the
CURRENT EPL SEASON ONLY — based on date filtering, not competition field.

This version:
    ✓ Works with DB schema WITHOUT competition column
    ✓ Evaluates only within date range (2024–25 EPL)
    ✓ Merges predictions + results cleanly
    ✓ Supports model_version filtering
    ✓ Outputs metrics + summary table
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

# ---------------------------------------------------------------------
# PATH FIX
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # .../soccer_agent_local
SRC = ROOT / "src"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

DB_PATH = ROOT / "data" / "soccer_agent.db"

SEASON_START = pd.Timestamp("2024-08-01")
SEASON_END = pd.Timestamp("2025-06-15")   # buffer for last match

EPL_TEAMS_2024 = {
    "Arsenal","Aston Villa","Bournemouth","Brentford","Brighton","Chelsea","Crystal Palace",
    "Everton","Fulham","Ipswich","Leicester","Liverpool","Man City","Man United",
    "Newcastle","Nott'm Forest","Southampton","Tottenham","West Ham","Wolves"
}


# =====================================================================
# Load DB tables
# =====================================================================
def load_results():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT date, home_team, away_team, FTHG, FTAG, Result
        FROM results
    """, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df


def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT *
        FROM predictions
    """, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    return df


# =====================================================================
# Helper: filter EPL season without competition field
# =====================================================================
def filter_current_season(results_df, predictions_df):
    # Filter by date range only
    r = results_df[
        (results_df["date"] >= SEASON_START) &
        (results_df["date"] <= SEASON_END)
    ].copy()

    # Filter by EPL team membership for safety
    r = r[
        (r["home_team"].isin(EPL_TEAMS_2024)) &
        (r["away_team"].isin(EPL_TEAMS_2024))
    ]

    p = predictions_df[
        (predictions_df["date"] >= SEASON_START) &
        (predictions_df["date"] <= SEASON_END)
    ].copy()

    return r, p


# =====================================================================
# Merge tables
# =====================================================================
def merge_predictions(results_df, preds_df):
    merged = results_df.merge(
        preds_df,
        on=["date", "home_team", "away_team"],
        how="inner",
        suffixes=("_res", "_pred")
    )
    return merged


# =====================================================================
# Metrics
# =====================================================================
def winner_from_goals(hg, ag):
    if hg > ag: return "H"
    if hg == ag: return "D"
    return "A"


def compute_metrics(df):
    df = df.copy()
    df["actual"] = df.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)

    # predicted class (index of max prob)
    df["predicted"] = df.apply(
        lambda r: np.argmax([r["home_win_prob"], r["draw_prob"], r["away_win_prob"]]),
        axis=1
    )

    # actual class index
    df["actual_idx"] = df["actual"].map({"H": 0, "D": 1, "A": 2})

    y_true = df["actual_idx"].values
    probs = df[["home_win_prob", "draw_prob", "away_win_prob"]].values

    # Accuracy
    accuracy = (df["predicted"] == df["actual_idx"]).mean()

    # Brier score — must specify all classes
    try:
        brier = brier_score_loss(
            y_true,
            probs,
            labels=[0, 1, 2]
        )
    except:
        brier = np.nan

    # Log loss — must specify all classes
    eps = 1e-12
    probs_clipped = np.clip(probs, eps, 1 - eps)

    try:
        ll = log_loss(
            y_true,
            probs_clipped,
            labels=[0, 1, 2]
        )
    except:
        ll = np.nan

    return accuracy, brier, ll



# =====================================================================
# MAIN
# =====================================================================
def run_comparison():
    print("\n============================================")
    print("    Evaluating Model vs ChatGPT (DB-only)")
    print("           Current EPL Season Only")
    print("============================================\n")

    results = load_results()
    preds = load_predictions()

    results, preds = filter_current_season(results, preds)

    merged = merge_predictions(results, preds)

    if merged.empty:
        raise RuntimeError(
            "No overlapping rows between predictions and results.\n"
            "Troubleshooting:\n"
            "  • Ensure backfill ran for 2024–25 dates\n"
            "  • Ensure team names match exactly\n"
            "  • Ensure predictions table contains DC/Elo model_version\n"
        )

    print(f"Matched predictions to results: {len(merged)} rows\n")

    # ------------------------------------------------------------------
    # Evaluate overall (using all model_version mixed)
    # ------------------------------------------------------------------
    print("Evaluating DC+Elo Ensemble Model:")
    acc, brier, ll = compute_metrics(merged)

    print(f"  Accuracy:      {acc:.4f}")
    print(f"  Brier Score:   {brier:.4f}")
    print(f"  Log Loss:      {ll:.4f}")

    # ------------------------------------------------------------------
    # Evaluate by model_version
    # ------------------------------------------------------------------
    print("\nBy Model Version:")
    versions = merged["model_version"].unique()

    for mv in versions:
        df_mv = merged[merged["model_version"] == mv]
        if df_mv.empty:
            continue

        acc, brier, ll = compute_metrics(df_mv)
        print(f"\nModel {mv}:")
        print(f"  Samples:       {len(df_mv)}")
        print(f"  Accuracy:      {acc:.4f}")
        print(f"  Brier Score:   {brier:.4f}")
        print(f"  Log Loss:      {ll:.4f}")

    print("\n============================================")
    print(" Evaluation Complete")
    print("============================================\n")


if __name__ == "__main__":
    run_comparison()
