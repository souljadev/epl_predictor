"""
tune_draw_calibration.py

Auto-tunes context-aware draw calibration parameters on top of the
existing model probabilities stored in DB.predictions.

This does NOT retrain DC/Elo – it learns a better mapping from
(raw pH, pD, pA, lambda_H, lambda_A) → calibrated (pH', pD', pA')
to minimize log loss / Brier and improve draw accuracy.

Later we can extend this to also tune ensemble weights and DC/Elo
hyperparameters once more internals are exposed.
"""

import sys
import sqlite3
from pathlib import Path
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local
DB_PATH = ROOT / "data" / "soccer_agent.db"

sys.path.insert(0, str(ROOT))

SEASON_START = pd.Timestamp("2024-08-01")
SEASON_END = pd.Timestamp("2025-06-15")

EPL_TEAMS_2024 = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nott'm Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
}


# ---------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------
def load_results() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT date, home_team, away_team, FTHG, FTAG, Result
        FROM results
        """,
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def load_predictions(model_version: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    if model_version:
        df = pd.read_sql_query(
            """
            SELECT
                date,
                home_team,
                away_team,
                model_version,
                home_win_prob,
                draw_prob,
                away_win_prob,
                exp_goals_home,
                exp_goals_away,
                exp_total_goals
            FROM predictions
            WHERE model_version = ?
            """,
            conn,
            params=(model_version,),
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT
                date,
                home_team,
                away_team,
                model_version,
                home_win_prob,
                draw_prob,
                away_win_prob,
                exp_goals_home,
                exp_goals_away,
                exp_total_goals
            FROM predictions
            """,
            conn,
        )
    conn.close()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    return df


def winner_from_goals(hg, ag) -> str:
    if hg > ag:
        return "H"
    if hg == ag:
        return "D"
    return "A"


# ---------------------------------------------------------------------
# CALIBRATION PARAM STRUCT
# ---------------------------------------------------------------------
@dataclass
class DrawCalibParams:
    base_factor: float          # global multiply on pD
    close_diff_thresh: float    # |lambda_H - lambda_A| < this → close
    close_factor: float         # multiply pD in close games
    low_total_thresh: float     # (lambda_H + lambda_A) < this → low scoring
    low_total_factor: float     # multiply pD in low scoring games
    ultra_diff_thresh: float    # tighter cut for very close
    ultra_total_thresh: float   # tighter cut for very low scoring
    ultra_factor: float         # multiply pD in ultra-tight low games
    blend_weight: float         # blend with baseline EPL draw rate


def apply_calibration(
    df: pd.DataFrame,
    params: DrawCalibParams,
    epl_draw_rate: float = 0.24,
) -> pd.DataFrame:
    """
    Apply parametric context-aware calibration to probabilities in df.

    df must contain:
        home_win_prob, draw_prob, away_win_prob,
        exp_goals_home, exp_goals_away
    """
    out = df.copy()

    pH = out["home_win_prob"].to_numpy()
    pD = out["draw_prob"].to_numpy()
    pA = out["away_win_prob"].to_numpy()

    lamH = out["exp_goals_home"].to_numpy()
    lamA = out["exp_goals_away"].to_numpy()
    goal_diff = np.abs(lamH - lamA)
    total_goals = lamH + lamA

    # STEP 1 — Global base factor
    pD = pD * params.base_factor

    # STEP 2 — Close games
    close_mask = goal_diff < params.close_diff_thresh
    pD[close_mask] *= params.close_factor

    # STEP 3 — Low-scoring games
    low_mask = total_goals < params.low_total_thresh
    pD[low_mask] *= params.low_total_factor

    # STEP 4 — Ultra-tight, ultra-low
    ultra_mask = (goal_diff < params.ultra_diff_thresh) & (
        total_goals < params.ultra_total_thresh
    )
    pD[ultra_mask] *= params.ultra_factor

    # STEP 5 — Blend with baseline EPL draw rate
    pD = (1 - params.blend_weight) * pD + params.blend_weight * epl_draw_rate

    # STEP 6 — Renormalize
    Z = pH + pD + pA
    # Avoid division by zero
    zero_mask = Z <= 0
    Z[zero_mask] = 1.0
    pH = pH / Z
    pD = pD / Z
    pA = pA / Z

    out["pH_cal"] = pH
    out["pD_cal"] = pD
    out["pA_cal"] = pA

    return out


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------
def compute_metrics(df: pd.DataFrame):
    """
    Expects df with:
        FTHG, FTAG,
        pH_cal, pD_cal, pA_cal
    """
    df = df.copy()
    df["actual"] = df.apply(lambda r: winner_from_goals(r["FTHG"], r["FTAG"]), axis=1)
    df["actual_idx"] = df["actual"].map({"H": 0, "D": 1, "A": 2})

    y_true = df["actual_idx"].to_numpy()
    probs = df[["pH_cal", "pD_cal", "pA_cal"]].to_numpy()

    # accuracy
    preds = probs.argmax(axis=1)
    accuracy = (preds == y_true).mean()

    # draw accuracy
    draw_mask = df["actual_idx"] == 1
    if draw_mask.any():
        draw_acc = (preds[draw_mask.to_numpy()] == 1).mean()
    else:
        draw_acc = np.nan

    # log loss & brier
    eps = 1e-12
    probs_clip = np.clip(probs, eps, 1 - eps)
    try:
        ll = log_loss(y_true, probs_clip, labels=[0, 1, 2])
    except Exception:
        ll = np.nan

    try:
        brier = brier_score_loss(y_true, probs, labels=[0, 1, 2])
    except Exception:
        brier = np.nan

    return {
        "accuracy": accuracy,
        "draw_accuracy": draw_acc,
        "log_loss": ll,
        "brier": brier,
    }


# ---------------------------------------------------------------------
# PARAM SEARCH
# ---------------------------------------------------------------------
def tune_draw_calibration(
    model_version: str | None = None,
    season_start: pd.Timestamp = SEASON_START,
    season_end: pd.Timestamp = SEASON_END,
):
    results = load_results()
    preds = load_predictions(model_version=model_version)

    # Filter to EPL season & teams
    results = results[
        (results["date"] >= season_start)
        & (results["date"] <= season_end)
        & (results["home_team"].isin(EPL_TEAMS_2024))
        & (results["away_team"].isin(EPL_TEAMS_2024))
    ].copy()

    preds = preds[
        (preds["date"] >= season_start)
        & (preds["date"] <= season_end)
    ].copy()

    merged = results.merge(
        preds,
        on=["date", "home_team", "away_team"],
        how="inner",
        suffixes=("_res", "_pred"),
    )

    if merged.empty:
        print("No overlapping rows between results and predictions in this window.")
        return

    print(f"Using {len(merged)} matches for tuning.\n")

    # Define search space (you can widen these ranges later)
    base_factors = [1.10, 1.20, 1.30]
    close_diffs = [0.30, 0.40, 0.50]
    close_factors = [1.10, 1.20, 1.30]
    low_totals = [2.20, 2.40, 2.60]
    low_factors = [1.05, 1.15, 1.25]
    ultra_diffs = [0.20, 0.30]
    ultra_totals = [2.00, 2.20]
    ultra_factors = [1.10, 1.20]
    blend_weights = [0.05, 0.10, 0.15]

    best = None

    total_combos = (
        len(base_factors)
        * len(close_diffs)
        * len(close_factors)
        * len(low_totals)
        * len(low_factors)
        * len(ultra_diffs)
        * len(ultra_totals)
        * len(ultra_factors)
        * len(blend_weights)
    )
    print(f"Total parameter combinations: {total_combos}\n")

    combo_iter = itertools.product(
        base_factors,
        close_diffs,
        close_factors,
        low_totals,
        low_factors,
        ultra_diffs,
        ultra_totals,
        ultra_factors,
        blend_weights,
    )

    for i, combo in enumerate(combo_iter, 1):
        params = DrawCalibParams(
            base_factor=combo[0],
            close_diff_thresh=combo[1],
            close_factor=combo[2],
            low_total_thresh=combo[3],
            low_total_factor=combo[4],
            ultra_diff_thresh=combo[5],
            ultra_total_thresh=combo[6],
            ultra_factor=combo[7],
            blend_weight=combo[8],
        )

        cal_df = apply_calibration(
            merged,
            params=params,
            epl_draw_rate=0.24,
        )
        metrics = compute_metrics(cal_df)

        # Objective: primarily minimize log loss, then Brier
        score = (metrics["log_loss"], metrics["brier"])

        if best is None or score < best["score"]:
            best = {
                "params": params,
                "metrics": metrics,
                "score": score,
            }

        if i % 50 == 0:
            print(
                f"[{i}/{total_combos}] current_best_logloss={best['metrics']['log_loss']:.4f}, "
                f"draw_acc={best['metrics']['draw_accuracy']:.3f}"
            )

    if best is None:
        print("No valid parameter combination found.")
        return

    print("\n================ BEST PARAMETERS ================")
    p = best["params"]
    m = best["metrics"]
    print("Params:")
    print(f"  base_factor         = {p.base_factor:.3f}")
    print(f"  close_diff_thresh   = {p.close_diff_thresh:.3f}")
    print(f"  close_factor        = {p.close_factor:.3f}")
    print(f"  low_total_thresh    = {p.low_total_thresh:.3f}")
    print(f"  low_total_factor    = {p.low_total_factor:.3f}")
    print(f"  ultra_diff_thresh   = {p.ultra_diff_thresh:.3f}")
    print(f"  ultra_total_thresh  = {p.ultra_total_thresh:.3f}")
    print(f"  ultra_factor        = {p.ultra_factor:.3f}")
    print(f"  blend_weight        = {p.blend_weight:.3f}")

    print("\nMetrics:")
    print(f"  Accuracy        = {m['accuracy']:.4f}")
    print(f"  Draw Accuracy   = {m['draw_accuracy']:.4f}")
    print(f"  Log Loss        = {m['log_loss']:.4f}")
    print(f"  Brier Score     = {m['brier']:.4f}")
    print("=================================================\n")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Tune context-aware draw calibration parameters using DB predictions + results."
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Optional: restrict tuning to a single model_version.",
    )
    parser.add_argument(
        "--season-start",
        type=str,
        default=SEASON_START.strftime("%Y-%m-%d"),
        help="Start date (YYYY-MM-DD) for tuning window.",
    )
    parser.add_argument(
        "--season-end",
        type=str,
        default=SEASON_END.strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD) for tuning window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    season_start = pd.to_datetime(args.season_start)
    season_end = pd.to_datetime(args.season_end)

    tune_draw_calibration(
        model_version=args.model_version,
        season_start=season_start,
        season_end=season_end,
    )


if __name__ == "__main__":
    main()
