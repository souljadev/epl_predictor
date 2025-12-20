"""
STEP 1 — Aggregate evaluation properly.

Reads the match-level evaluation CSV produced by evaluate_vs_actuals.py
and aggregates metrics by model family (e.g., DC/Elo vs ChatGPT).

- Weighted aggregation (by match count)
- Read-only
- No DB writes
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
EVAL_DIR = Path("data/evaluation")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def infer_model_family(model_version: str) -> str:
    mv = str(model_version).lower()
    if "chatgpt" in mv:
        return "ChatGPT"
    if "dc_elo" in mv or "elo" in mv:
        return "DC/Elo"
    return "Other"


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    s = series.astype(float)
    w = weights.astype(float)
    mask = s.notna() & w.notna()
    if mask.sum() == 0:
        return np.nan
    return float((s[mask] * w[mask]).sum() / w[mask].sum())


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not EVAL_DIR.exists():
        raise FileNotFoundError(f"Evaluation dir not found: {EVAL_DIR.resolve()}")

    csvs = sorted(EVAL_DIR.glob("match_level_eval_*.csv"))
    if not csvs:
        raise FileNotFoundError("No match_level_eval_*.csv files found.")

    # Use the most recent evaluation snapshot
    path = csvs[-1]
    df = pd.read_csv(path)

    required = {
        "model_version",
        "winner_correct",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required}")

    df["model_family"] = df["model_version"].apply(infer_model_family)
    df["winner_correct"] = df["winner_correct"].astype(float)

    # Optional metrics if present
    has_brier = "brier_row" in df.columns
    has_logloss = "logloss_row" in df.columns

    # Aggregate
    rows = []
    for fam, g in df.groupby("model_family"):
        matches = len(g)
        acc = g["winner_correct"].mean()

        brier = (
            weighted_mean(g["brier_row"], pd.Series(np.ones(matches)))
            if has_brier else np.nan
        )
        logloss = (
            weighted_mean(g["logloss_row"], pd.Series(np.ones(matches)))
            if has_logloss else np.nan
        )

        rows.append({
            "model_family": fam,
            "matches": matches,
            "winner_accuracy": acc,
            "brier": brier,
            "logloss": logloss,
        })

    out = pd.DataFrame(rows).sort_values("matches", ascending=False)

    print("\n==============================")
    print("STEP 1 — AGGREGATED EVALUATION")
    print("==============================")
    print(out.to_string(index=False))

    print("\nInterpretation bands (3-way soccer):")
    print("- Accuracy: <40% bad | 40–50% OK | 50–58% good | 60%+ strong")
    print("- Brier:    >0.60 bad | 0.55–0.60 OK | 0.48–0.55 good | <0.48 strong")
    print("- LogLoss:  >1.0 bad | 0.90–1.0 OK | 0.75–0.90 good | <0.75 strong")

    print(f"\nSource file: {path.resolve()}")


if __name__ == "__main__":
    main()
