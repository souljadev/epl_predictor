"""
compare_models.py

Evaluates all prediction model_versions in the SQLite DB for the
CURRENT EPL SEASON ONLY.

Works with your real schema:

    results(date, home_team, away_team, FTHG, FTAG, Result)
    predictions(date, home_team, away_team, model_version,
                home_win_prob, draw_prob, away_win_prob, ...)

Metrics computed:
    - Accuracy
    - Brier Score
    - Log Loss
    - Draw Accuracy

Outputs:
    - Per-model metrics block
    - Summary table
    - Final print: "Status: done"
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


# ======================================================================
# CONFIG
# ======================================================================

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "soccer_agent.db"

SEASON_START = "2024-08-01"
SEASON_END = "2025-06-30"


# ======================================================================
# DB LOADING HELPERS
# ======================================================================

def get_conn():
    return sqlite3.connect(DB_PATH)


def load_results(conn):
    """
    Loads match outcomes (ground truth) from results table.
    """
    q = """
        SELECT
            date,
            home_team,
            away_team,
            FTHG,
            FTAG,
            Result
        FROM results
        WHERE date BETWEEN ? AND ?
          AND FTHG IS NOT NULL
          AND FTAG IS NOT NULL
    """

    df = pd.read_sql_query(q, conn, params=[SEASON_START, SEASON_END])

    if df.empty:
        print("⚠ No results found for this season in `results` table.")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Ground truth labels
    def outcome(row):
        if row["FTHG"] > row["FTAG"]:
            return "H"
        elif row["FTHG"] < row["FTAG"]:
            return "A"
        else:
            return "D"

    df["actual_outcome"] = df.apply(outcome, axis=1)
    return df


def load_predictions(conn):
    """
    Loads *all* model_version rows from predictions table.
    Clips + normalizes probabilities.
    """
    q = """
        SELECT
            date,
            home_team,
            away_team,
            model_version,
            home_win_prob,
            draw_prob,
            away_win_prob
        FROM predictions
        WHERE date BETWEEN ? AND ?
    """

    df = pd.read_sql_query(q, conn, params=[SEASON_START, SEASON_END])

    if df.empty:
        print("⚠ No predictions found for this season in `predictions` table.")
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["home_win_prob"] = df["home_win_prob"].astype(float)
    df["draw_prob"] = df["draw_prob"].astype(float)
    df["away_win_prob"] = df["away_win_prob"].astype(float)

    # Clean probability vectors
    probs = df[["home_win_prob", "draw_prob", "away_win_prob"]].clip(0, 1)
    sums = probs.sum(axis=1).replace(0, np.nan)
    df["p_home"] = probs["home_win_prob"] / sums
    df["p_draw"] = probs["draw_prob"] / sums
    df["p_away"] = probs["away_win_prob"] / sums

    return df


# ======================================================================
# MERGE
# ======================================================================

def merge_data(preds, results):
    merged = preds.merge(
        results,
        on=["date", "home_team", "away_team"],
        how="inner",
        validate="many_to_one",
    )

    if merged.empty:
        print("⚠ No overlap between predictions and results.")
        return merged

    return merged


# ======================================================================
# METRIC CALCULATIONS
# ======================================================================

def compute_metrics(df):
    """
    Metrics for one model_version.
    """
    if df.empty:
        return {
            "n": 0,
            "accuracy": np.nan,
            "brier": np.nan,
            "log_loss": np.nan,
            "draw_accuracy": np.nan,
        }

    mapping = {"H": 0, "D": 1, "A": 2}

    y_true_idx = df["actual_outcome"].map(mapping).values
    probs = df[["p_home", "p_draw", "p_away"]].values
    y_pred_idx = probs.argmax(axis=1)

    # Accuracy
    acc = (y_true_idx == y_pred_idx).mean()

    # Brier
    y_onehot = np.zeros_like(probs)
    for i, idx in enumerate(y_true_idx):
        y_onehot[i, idx] = 1
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

    # Log loss
    eps = 1e-12
    true_probs = probs[np.arange(len(probs)), y_true_idx]
    log_loss = -np.mean(np.log(true_probs + eps))

    # Draw accuracy
    mask = df["actual_outcome"] == "D"
    if mask.any():
        draw_acc = (y_pred_idx[mask] == y_true_idx[mask]).mean()
    else:
        draw_acc = np.nan

    return {
        "n": int(len(df)),
        "accuracy": float(acc),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "draw_accuracy": float(draw_acc),
    }


# ======================================================================
# PRINTING
# ======================================================================

def print_header():
    print("\n" + "=" * 60)
    print("  Evaluating Prediction Models — Current EPL Season")
    print("=" * 60 + "\n")


def print_model_block(name, metrics):
    print(f"Model: {name}")
    print(f"  Samples:       {metrics['n']}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  Brier Score:   {metrics['brier']:.4f}")
    print(f"  Log Loss:      {metrics['log_loss']:.4f}")
    print(f"  Draw Accuracy: {metrics['draw_accuracy']:.4f}\n")


def print_summary(summary_df):
    if summary_df.empty:
        print("No metrics to display.")
        return

    df = summary_df.copy()
    df["accuracy"] = df["accuracy"].map(lambda x: f"{x:.4f}")
    df["brier"] = df["brier"].map(lambda x: f"{x:.4f}")
    df["log_loss"] = df["log_loss"].map(lambda x: f"{x:.4f}")
    df["draw_accuracy"] = df["draw_accuracy"].map(lambda x: f"{x:.4f}")

    print("\nSummary (sorted by Brier Score):")
    print(df.to_string(index=False))
    print()


# ======================================================================
# MAIN
# ======================================================================

def run_comparison():
    print_header()

    conn = get_conn()

    results = load_results(conn)
    if results.empty:
        print("Status: done")
        return

    preds = load_predictions(conn)
    if preds.empty:
        print("Status: done")
        return

    merged = merge_data(preds, results)
    if merged.empty:
        print("Status: done")
        return

    # Compute per-model metrics
    rows = []
    for model_name, group in merged.groupby("model_version"):
        m = compute_metrics(group)
        m["model_version"] = model_name
        rows.append(m)

    print("Status: done")


if __name__ == "__main__":
    run_comparison()
