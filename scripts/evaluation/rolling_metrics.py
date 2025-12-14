"""
STEP 2 — Rolling performance metrics.

Reads the match-level evaluation CSV and computes rolling metrics
(accuracy, Brier, log loss) over time by model family.

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
ROLL_WINDOWS = [20, 50]

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


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    csvs = sorted(EVAL_DIR.glob("match_level_eval_*.csv"))
    if not csvs:
        raise FileNotFoundError("No match_level_eval_*.csv files found.")

    path = csvs[-1]
    df = pd.read_csv(path)

    required = {"date", "model_version", "winner_correct"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing required columns: {required}")

    df["model_family"] = df["model_version"].apply(infer_model_family)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    has_brier = "brier_row" in df.columns
    has_logloss = "logloss_row" in df.columns

    print("\n======================================")
    print("STEP 2 — ROLLING PERFORMANCE METRICS")
    print("======================================")
    print(f"Source file: {path.name}\n")

    for fam, g in df.groupby("model_family"):
        print(f"\n--- {fam} ---")
        g = g.copy()

        g["winner_correct"] = g["winner_correct"].astype(float)

        for w in ROLL_WINDOWS:
            g[f"acc_{w}"] = rolling_mean(g["winner_correct"], w)

            if has_brier:
                g[f"brier_{w}"] = rolling_mean(g["brier_row"], w)
            if has_logloss:
                g[f"logloss_{w}"] = rolling_mean(g["logloss_row"], w)

        # Show last available rolling snapshot
        tail_cols = ["date"]
        for w in ROLL_WINDOWS:
            tail_cols.append(f"acc_{w}")
            if has_brier:
                tail_cols.append(f"brier_{w}")
            if has_logloss:
                tail_cols.append(f"logloss_{w}")

        snapshot = g[tail_cols].dropna().tail(1)

        if snapshot.empty:
            print("Not enough data for rolling windows yet.")
        else:
            print(snapshot.to_string(index=False))

    print("\nInterpretation guide:")
    print("- Accuracy trending ↑ = better winner picks")
    print("- Brier trending ↓ = better probability calibration")
    print("- Log loss trending ↓ = less overconfidence")
    print("- Flat lines = model stagnation")
    print("- Sudden jumps = regime change or bad retrain")


if __name__ == "__main__":
    main()

