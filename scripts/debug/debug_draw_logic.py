"""
debug_draw_logic.py

Comprehensive end-to-end validation for DRAW logic in the EPL agent pipeline.
Tests:
    1. winner_from_goals() correctness
    2. Results table: draws correctly detected
    3. Predictions table: draw_prob behavior
    4. Merge: actual = D rows preserved
    5. Evaluation: predicted draws computed correctly
    6. Per-draw accuracy
    7. List mismatches (if any)

Run:
    python scripts/debug/debug_draw_logic.py
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local
SRC = ROOT / "src"
DB_PATH = ROOT / "data" / "soccer_agent.db"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))


# ------------------------------------------------------------
# 1. Core draw logic
# ------------------------------------------------------------
def winner_from_goals(hg, ag):
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


def test_core_logic():
    tests = [
        (2, 1, "H"),
        (0, 0, "D"),
        (3, 3, "D"),
        (1, 4, "A"),
        (4, 1, "H"),
        (5, 5, "D"),
    ]
    print("\n[1] Testing core winner_from_goals() logic:")
    for hg, ag, expected in tests:
        out = winner_from_goals(hg, ag)
        print(f"  {hg}-{ag} → {out} (expected {expected})")


# ------------------------------------------------------------
# 2. Load DB tables
# ------------------------------------------------------------
def load_results():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT date, home_team, away_team, FTHG, FTAG, Result
        FROM results
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_predictions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT
            date,
            home_team,
            away_team,
            model_version,
            home_win_prob,
            draw_prob,
            away_win_prob
        FROM predictions
    """, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ------------------------------------------------------------
# 3. Validate draws in results
# ------------------------------------------------------------
def test_results_draws(results):
    draws = results[results["FTHG"] == results["FTAG"]]
    print(f"\n[2] Draws in results table: {len(draws)}")
    print(draws.head(10)[["date", "home_team", "away_team", "FTHG", "FTAG"]])


# ------------------------------------------------------------
# 4. Merge tables
# ------------------------------------------------------------
def merge_predictions(results, preds):
    merged = results.merge(
        preds,
        on=["date", "home_team", "away_team"],
        how="inner",
        suffixes=("_res", "_pred")
    )
    return merged


# ------------------------------------------------------------
# 5. Validate draw predictions
# ------------------------------------------------------------
def test_draw_predictions(merged):
    # Actual draws
    merged["actual"] = merged.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)

    draws = merged[merged["actual"] == "D"]
    print(f"\n[3] Actual draws matched to predictions: {len(draws)}")

    if len(draws) == 0:
        print("  No draws found — check DB/resuts.")
        return

    print(draws.head(10)[[
        "date","home_team","away_team",
        "FTHG","FTAG","actual",
        "home_win_prob","draw_prob","away_win_prob"
    ]])


# ------------------------------------------------------------
# 6. Predicted draw correctness
# ------------------------------------------------------------
def test_predicted_draws(merged):
    def predicted_label(row):
        probs = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        return {0: "H", 1: "D", 2: "A"}[int(np.argmax(probs))]

    merged["predicted"] = merged.apply(predicted_label, axis=1)

    draws = merged[merged["actual"] == "D"]
    draws["correct"] = draws["predicted"] == draws["actual"]

    acc = draws["correct"].mean()

    print(f"\n[4] Draw prediction accuracy: {acc:.3f}")
    print("Sample predicted draws:")
    print(draws.head(10)[[
        "date","home_team","away_team",
        "FTHG","FTAG","actual","predicted",
        "home_win_prob","draw_prob","away_win_prob","correct"
    ]])


# ------------------------------------------------------------
# 7. Identify mismatches
# ------------------------------------------------------------
def find_draw_mismatches(merged):
    merged["actual"] = merged.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)

    draws = merged[merged["actual"] == "D"]
    mismatches = draws[draws["predicted"] != draws["actual"]]

    print(f"\n[5] Incorrect draw predictions: {len(mismatches)}")
    if len(mismatches) > 0:
        print(mismatches.head(20)[[
            "date","home_team","away_team",
            "FTHG","FTAG","actual","predicted",
            "home_win_prob","draw_prob","away_win_prob"
        ]])
    else:
        print("  All predicted draws correct!")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    print("\n==========================================================")
    print("               DRAW LOGIC DIAGNOSTIC TOOL")
    print("==========================================================\n")

    test_core_logic()

    results = load_results()
    preds = load_predictions()

    test_results_draws(results)

    merged = merge_predictions(results, preds)
    print(f"\n[MERGE] predictions matched to results: {len(merged)} rows")

    test_draw_predictions(merged)
    test_predicted_draws(merged)
    find_draw_mismatches(merged)

    print("\n==========================================================")
    print("                    DIAGNOSTIC COMPLETE")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
