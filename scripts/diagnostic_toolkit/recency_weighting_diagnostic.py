"""
STEP 4C — Recency weighting diagnostic (DC/Elo).

This does NOT retrain the model.
It tests whether recent matches should matter more
by applying exponential time-decay to evaluation metrics.

If recency-weighted metrics are much better,
your model needs faster adaptation (decay / higher K / shorter window).
"""

from pathlib import Path
import numpy as np
import pandas as pd

EVAL_DIR = Path("data/evaluation")

# Half-life in days (try multiple)
HALF_LIVES = [30, 60, 120]


def infer_model_family(model_version: str) -> str:
    mv = str(model_version).lower()
    if "chatgpt" in mv:
        return "ChatGPT"
    if "dc_elo" in mv or "elo" in mv:
        return "DC/Elo"
    return "Other"


def compute_weights(dates: pd.Series, half_life_days: int) -> np.ndarray:
    max_date = dates.max()
    age_days = (max_date - dates).dt.days.astype(float)
    # exponential decay
    return np.exp(-np.log(2) * age_days / half_life_days)


def weighted_acc(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    pred = np.argmax(p, axis=1)
    return float(np.sum(w * (pred == y)) / np.sum(w))


def weighted_brier(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    y_oh = np.zeros_like(p)
    y_oh[np.arange(len(y)), y] = 1.0
    return float(np.sum(w * np.sum((p - y_oh) ** 2, axis=1)) / np.sum(w))


def weighted_logloss(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-np.sum(w * np.log(p[np.arange(len(y)), y])) / np.sum(w))


def main():
    csvs = sorted(EVAL_DIR.glob("match_level_eval_*.csv"))
    if not csvs:
        raise FileNotFoundError("No match_level_eval_*.csv files found.")

    src = csvs[-1]
    df = pd.read_csv(src)

    needed = {"date", "model_version", "actual_winner", "pH", "pD", "pA"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {needed}")

    df["model_family"] = df["model_version"].apply(infer_model_family)
    df["date"] = pd.to_datetime(df["date"])

    dc = df[(df["model_family"] == "DC/Elo") & df["actual_winner"].isin(["H", "D", "A"])].copy()
    if dc.empty:
        print("No DC/Elo rows available.")
        return

    p = dc[["pH", "pD", "pA"]].astype(float).to_numpy()
    p = p / p.sum(axis=1, keepdims=True)

    y_map = {"H": 0, "D": 1, "A": 2}
    y = dc["actual_winner"].map(y_map).astype(int).to_numpy()

    print("\n======================================")
    print("STEP 4C — RECENCY WEIGHTING DIAGNOSTIC")
    print("======================================")
    print(f"Matches evaluated: {len(dc)}")

    # Unweighted baseline
    w0 = np.ones(len(dc))
    print("\nUnweighted:")
    print(f"  Accuracy : {weighted_acc(p, y, w0):.4f}")
    print(f"  Brier    : {weighted_brier(p, y, w0):.4f}")
    print(f"  Log loss : {weighted_logloss(p, y, w0):.4f}")

    # Recency-weighted
    for hl in HALF_LIVES:
        w = compute_weights(dc["date"], hl)
        print(f"\nHalf-life = {hl} days:")
        print(f"  Accuracy : {weighted_acc(p, y, w):.4f}")
        print(f"  Brier    : {weighted_brier(p, y, w):.4f}")
        print(f"  Log loss : {weighted_logloss(p, y, w):.4f}")


if __name__ == "__main__":
    main()