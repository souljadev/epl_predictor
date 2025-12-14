"""
STEP 3 — Multiclass temperature scaling calibration for 3-way soccer probs.

Reads the match-level evaluation CSV and calibrates DC/Elo probabilities
(pH, pD, pA) using temperature scaling (softmax(log(p)/T)).

Outputs:
- Before vs After: accuracy, Brier, log loss
- Writes calibrated CSV next to the source file
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

EVAL_DIR = Path("data/evaluation")


def infer_model_family(model_version: str) -> str:
    mv = str(model_version).lower()
    if "chatgpt" in mv:
        return "ChatGPT"
    if "dc_elo" in mv or "elo" in mv:
        return "DC/Elo"
    return "Other"


def safe_probs(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    return p


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=1, keepdims=True)


def nll_from_probs(p: np.ndarray, y: np.ndarray) -> float:
    # y is 0/1/2 for H/D/A
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier_from_probs(p: np.ndarray, y: np.ndarray) -> float:
    y_oh = np.zeros_like(p)
    y_oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - y_oh) ** 2, axis=1)))


def acc_from_probs(p: np.ndarray, y: np.ndarray) -> float:
    pred = np.argmax(p, axis=1)
    return float(np.mean(pred == y))


def probs_to_logits(p: np.ndarray) -> np.ndarray:
    # logits up to additive constant: log(p)
    return np.log(p)


def apply_temperature(p: np.ndarray, T: float) -> np.ndarray:
    p = safe_probs(p)
    logits = probs_to_logits(p) / T
    return softmax(logits)


def grid_search_temperature(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Simple robust search for T that minimizes log loss.
    We search a reasonable range. If your model is overconfident, best T > 1.
    """
    Ts = np.concatenate([
        np.linspace(0.5, 1.0, 26),
        np.linspace(1.0, 5.0, 81),
        np.linspace(5.0, 10.0, 26),
    ])
    best_T = 1.0
    best_nll = float("inf")
    for T in Ts:
        pT = apply_temperature(p, float(T))
        nll = nll_from_probs(pT, y)
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T, best_nll


def main():
    csvs = sorted(EVAL_DIR.glob("match_level_eval_*.csv"))
    if not csvs:
        raise FileNotFoundError("No match_level_eval_*.csv files found.")

    src = csvs[-1]
    df = pd.read_csv(src)

    # Require probabilities + actual outcome
    needed = {"model_version", "actual_winner", "pH", "pD", "pA"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns for calibration: {needed}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["model_family"] = df["model_version"].apply(infer_model_family)

    # Calibrate ONLY DC/Elo rows with valid actual_winner
    dc = df[(df["model_family"] == "DC/Elo") & df["actual_winner"].isin(["H", "D", "A"])].copy()
    if dc.empty:
        print("No DC/Elo rows with probabilities + actual_winner found.")
        return

    # Build arrays
    p = dc[["pH", "pD", "pA"]].astype(float).to_numpy()
    p = safe_probs(p)

    y_map = {"H": 0, "D": 1, "A": 2}
    y = dc["actual_winner"].map(y_map).astype(int).to_numpy()

    # Before metrics
    before = {
        "accuracy": acc_from_probs(p, y),
        "brier": brier_from_probs(p, y),
        "logloss": nll_from_probs(p, y),
    }

    # Fit temperature
    best_T, best_nll = grid_search_temperature(p, y)

    # After metrics
    p_cal = apply_temperature(p, best_T)
    after = {
        "accuracy": acc_from_probs(p_cal, y),
        "brier": brier_from_probs(p_cal, y),
        "logloss": nll_from_probs(p_cal, y),
    }

    # Write calibrated probs back into the full df (only for DC/Elo rows)
    df["pH_cal"] = np.nan
    df["pD_cal"] = np.nan
    df["pA_cal"] = np.nan

    df.loc[dc.index, "pH_cal"] = p_cal[:, 0]
    df.loc[dc.index, "pD_cal"] = p_cal[:, 1]
    df.loc[dc.index, "pA_cal"] = p_cal[:, 2]

    out_path = src.with_name(src.stem + "_calibrated.csv")
    df.to_csv(out_path, index=False)

    print("\n======================================")
    print("STEP 3 — TEMPERATURE CALIBRATION (DC/Elo)")
    print("======================================")
    print(f"Rows calibrated: {len(dc)}")
    print(f"Best temperature T: {best_T:.3f}  (T>1 usually means overconfident)")
    print("\nBefore:")
    print(f"  Accuracy : {before['accuracy']:.4f}")
    print(f"  Brier    : {before['brier']:.4f}")
    print(f"  Log loss : {before['logloss']:.4f}")
    print("\nAfter:")
    print(f"  Accuracy : {after['accuracy']:.4f}")
    print(f"  Brier    : {after['brier']:.4f}")
    print(f"  Log loss : {after['logloss']:.4f}")
    print("\nWrote calibrated file:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
