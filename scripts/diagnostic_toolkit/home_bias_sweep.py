"""
STEP 4B — Home bias sweep on predicted probabilities (DC/Elo).

We do NOT retrain the model here.
We test whether shifting probability mass toward the home team (or away team)
improves log loss / Brier.

Transform:
  pH' = pH * exp(+k)
  pA' = pA * exp(-k)
  pD' unchanged
then renormalize.

Grid-search k to minimize log loss.

Outputs:
- Best k
- Before vs After: accuracy, Brier, log loss
- Writes CSV with pH_homebias, pD_homebias, pA_homebias
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


def apply_home_bias(p: np.ndarray, k: float) -> np.ndarray:
    """
    p columns: [pH, pD, pA]
    k > 0 pushes probability mass toward home (and away down).
    k < 0 pushes toward away.
    """
    p = safe_probs(p)
    p2 = p.copy()
    scale_h = float(np.exp(k))
    scale_a = float(np.exp(-k))
    p2[:, 0] = p2[:, 0] * scale_h
    p2[:, 2] = p2[:, 2] * scale_a
    p2 = p2 / p2.sum(axis=1, keepdims=True)
    return safe_probs(p2)


def nll(p: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    y_oh = np.zeros_like(p)
    y_oh[np.arange(len(y)), y] = 1.0
    return float(np.mean(np.sum((p - y_oh) ** 2, axis=1)))


def acc(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.argmax(p, axis=1) == y))


def grid_search_k(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Search k in a reasonable range.
    If model underestimates home edge, best k will be > 0.
    """
    ks = np.concatenate([
        np.linspace(-1.0, -0.2, 41),
        np.linspace(-0.2, 0.2, 81),
        np.linspace(0.2, 1.0, 41),
    ])
    best_k = 0.0
    best_nll = nll(p, y)

    for k in ks:
        p_adj = apply_home_bias(p, float(k))
        v = nll(p_adj, y)
        if v < best_nll:
            best_nll = v
            best_k = float(k)

    return best_k, best_nll


def main():
    csvs = sorted(EVAL_DIR.glob("match_level_eval_*.csv"))
    if not csvs:
        raise FileNotFoundError("No match_level_eval_*.csv files found.")

    src = csvs[-1]
    df = pd.read_csv(src)

    needed = {"model_version", "actual_winner", "pH", "pD", "pA"}
    if not needed.issubset(df.columns):
        raise ValueError(
            f"CSV missing required columns for home bias sweep: {needed}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["model_family"] = df["model_version"].apply(infer_model_family)
    dc = df[(df["model_family"] == "DC/Elo") & df["actual_winner"].isin(["H", "D", "A"])].copy()
    if dc.empty:
        print("No DC/Elo rows with probabilities + actual_winner found.")
        return

    p = dc[["pH", "pD", "pA"]].astype(float).to_numpy()
    p = safe_probs(p)

    y_map = {"H": 0, "D": 1, "A": 2}
    y = dc["actual_winner"].map(y_map).astype(int).to_numpy()

    before = {
        "accuracy": acc(p, y),
        "brier": brier(p, y),
        "logloss": nll(p, y),
        "mean_pH": float(np.mean(p[:, 0])),
        "mean_pD": float(np.mean(p[:, 1])),
        "mean_pA": float(np.mean(p[:, 2])),
    }

    best_k, _ = grid_search_k(p, y)
    p_bias = apply_home_bias(p, best_k)

    after = {
        "accuracy": acc(p_bias, y),
        "brier": brier(p_bias, y),
        "logloss": nll(p_bias, y),
        "mean_pH": float(np.mean(p_bias[:, 0])),
        "mean_pD": float(np.mean(p_bias[:, 1])),
        "mean_pA": float(np.mean(p_bias[:, 2])),
    }

    # Write adjusted probs back into full df (only DC/Elo rows)
    df["pH_homebias"] = np.nan
    df["pD_homebias"] = np.nan
    df["pA_homebias"] = np.nan
    df.loc[dc.index, "pH_homebias"] = p_bias[:, 0]
    df.loc[dc.index, "pD_homebias"] = p_bias[:, 1]
    df.loc[dc.index, "pA_homebias"] = p_bias[:, 2]

    out_path = src.with_name(src.stem + "_homebias.csv")
    df.to_csv(out_path, index=False)

    print("\n======================================")
    print("STEP 4B — HOME BIAS SWEEP (DC/Elo)")
    print("======================================")
    print(f"Rows adjusted: {len(dc)}")
    print(f"Best k: {best_k:.3f}  (k>0 pushes toward home, k<0 pushes toward away)")
    print("\nBefore:")
    print(f"  Accuracy : {before['accuracy']:.4f}")
    print(f"  Brier    : {before['brier']:.4f}")
    print(f"  Log loss : {before['logloss']:.4f}")
    print(f"  Mean pH/pD/pA: {before['mean_pH']:.4f} / {before['mean_pD']:.4f} / {before['mean_pA']:.4f}")
    print("\nAfter:")
    print(f"  Accuracy : {after['accuracy']:.4f}")
    print(f"  Brier    : {after['brier']:.4f}")
    print(f"  Log loss : {after['logloss']:.4f}")
    print(f"  Mean pH/pD/pA: {after['mean_pH']:.4f} / {after['mean_pD']:.4f} / {after['mean_pA']:.4f}")
    print("\nWrote file:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
