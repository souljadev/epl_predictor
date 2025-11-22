import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXP_PATH = ROOT / "models" / "history" / "backtest_expanding_matchday.csv"
ROLL_PATH = ROOT / "models" / "history" / "backtest_rolling_fast.csv"
OUT_DIR = ROOT / "models" / "evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_season_start_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["season_start_year"] = df["Date"].dt.year.where(
        df["Date"].dt.month >= 7,
        df["Date"].dt.year - 1,
    )
    return df


def expected_calibration_error(y_true, probs, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    pred_class = probs.argmax(axis=1)
    pred_conf = probs.max(axis=1)
    correct = (pred_class == y_true).astype(float)

    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_conf >= lo) & (pred_conf < hi)
        if mask.sum() == 0:
            continue
        avg_conf = pred_conf[mask].mean()
        avg_acc = correct[mask].mean()
        ece += abs(avg_conf - avg_acc) * (mask.sum() / N)

    return float(ece)


def compute_metrics(df: pd.DataFrame, label: str):
    required = ["pH", "pD", "pA"]
    for col in required:
        if col not in df.columns:
            return {"label": label, "n_matches": 0, "error": f"Missing {col}"}

    mapping = {"H": 0, "D": 1, "A": 2}

    if "Result" in df.columns:
        df = df.dropna(subset=["pH", "pD", "pA", "Result"]).copy()
        df = df[df["Result"].isin(mapping.keys())]
    else:
        df = df.dropna(subset=["pH", "pD", "pA"])

    if df.empty:
        return {"label": label, "n_matches": 0}

    probs = df[["pH", "pD", "pA"]].values

    metrics = {
        "label": label,
        "n_matches": len(df),
        "accuracy": np.nan,
        "acc_H": np.nan,
        "acc_D": np.nan,
        "acc_A": np.nan,
        "brier": np.nan,
        "log_loss": np.nan,
        "calibration_ece": np.nan,
        "mae_home": np.nan,
        "mae_away": np.nan,
        "mae_total": np.nan,
    }

    if "Result" in df.columns:
        y_true = df["Result"].map(mapping).values
        y_pred_idx = probs.argmax(axis=1)
        idx_to_label = {0: "H", 1: "D", 2: "A"}
        y_pred = np.vectorize(idx_to_label.get)(y_pred_idx)

        metrics["accuracy"] = float((y_pred == df["Result"]).mean())

        for label_val, key in [("H", "acc_H"), ("D", "acc_D"), ("A", "acc_A")]:
            mask = df["Result"] == label_val
            if mask.any():
                metrics[key] = float((y_pred[mask] == label_val).mean())

        y_onehot = np.eye(3)[y_true]
        metrics["brier"] = float(np.mean(((probs - y_onehot) ** 2).sum(axis=1)))

        eps = 1e-12
        metrics["log_loss"] = float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))

        metrics["calibration_ece"] = expected_calibration_error(y_true, probs)

    if {"FTHG", "FTAG", "ExpHomeGoals", "ExpAwayGoals"}.issubset(df.columns):
        metrics["mae_home"] = float(np.mean(np.abs(df["FTHG"] - df["ExpHomeGoals"])))
        metrics["mae_away"] = float(np.mean(np.abs(df["FTAG"] - df["ExpAwayGoals"])))
        metrics["mae_total"] = float(
            np.mean(np.abs((df["FTHG"] + df["FTAG"]) - (df["ExpHomeGoals"] + df["ExpAwayGoals"])))
        )

    return metrics


def run_evaluation():
    print("===== EVALUATING BACKTESTS (EXPANDING vs ROLLING_FAST) =====")

    exp = pd.read_csv(EXP_PATH)
    roll = pd.read_csv(ROLL_PATH)

    exp = add_season_start_year(exp)
    roll = add_season_start_year(roll)

    seasons = sorted(exp["season_start_year"].unique())
    last_two = seasons[-2:] if len(seasons) >= 2 else seasons

    print(f"Seasons found: {seasons}")
    print(f"Last two seasons used for focus: {last_two}")

    rows = []

    rows.append(compute_metrics(exp, "expanding_all"))
    rows.append(compute_metrics(roll, "rolling_fast_all"))

    exp_last2 = exp[exp["season_start_year"].isin(last_two)]
    roll_last2 = roll[roll["season_start_year"].isin(last_two)]

    rows.append(compute_metrics(exp_last2, "expanding_last2"))
    rows.append(compute_metrics(roll_last2, "rolling_fast_last2"))

    summary = pd.DataFrame(rows)
    out_path = OUT_DIR / "metrics_backtests_summary.csv"
    summary.to_csv(out_path, index=False)

    print("\n===== SUMMARY (key metrics) =====")
    cols = [
        "label",
        "n_matches",
        "accuracy",
        "acc_H",
        "acc_D",
        "acc_A",
        "brier",
        "log_loss",
        "calibration_ece",
        "mae_home",
        "mae_away",
        "mae_total",
    ]
    available_cols = [c for c in cols if c in summary.columns]
    print(summary[available_cols].to_string(index=False))

    print(f"\nSaved metrics summary -> {out_path}\n")


if __name__ == "__main__":
    run_evaluation()
