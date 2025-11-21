import pandas as pd
import numpy as np
from pathlib import Path
import os


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PRED_PATH = ROOT / "models" / "predictions" / "predictions_full.csv"
ACTUAL_PATH = ROOT / "data" / "raw" / "epl_combined.csv"  # Holds all historical results
OUT_DIR = ROOT / "models" / "evaluation"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_predictions():
    df = pd.read_csv(PRED_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def load_actuals():
    df = pd.read_csv(ACTUAL_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column in actuals: {col}")

    # Filter to rows where scores are known
    df = df.dropna(subset=["FTHG", "FTAG"])

    return df


# ============================================================
# MERGE ACTUALS + PREDICTIONS
# ============================================================

def merge_pred_actual(pred, actual):
    merged = pred.merge(
        actual,
        on=["Date", "HomeTeam", "AwayTeam"],
        how="inner"
    )

    if merged.empty:
        raise ValueError("❌ No matching actual results found for these predictions.")

    return merged


# ============================================================
# METRIC CALCULATIONS
# ============================================================

def compute_metrics(df):

    # Actual label (H/D/A)
    def label(row):
        if row["FTHG"] > row["FTAG"]:
            return "H"
        elif row["FTHG"] < row["FTAG"]:
            return "A"
        else:
            return "D"

    df["Actual"] = df.apply(label, axis=1)

    # Predicted label
    df["Predicted"] = df[
        ["home_win_prob", "draw_prob", "away_win_prob"]
    ].idxmax(axis=1)

    df["Predicted"] = df["Predicted"].map({
        "home_win_prob": "H",
        "draw_prob": "D",
        "away_win_prob": "A"
    })

    # Accuracy
    accuracy = (df["Predicted"] == df["Actual"]).mean()

    # Brier score
    mapping = {"H": 0, "D": 1, "A": 2}
    y_true = df["Actual"].map(mapping).values
    probs = df[["home_win_prob", "draw_prob", "away_win_prob"]].values
    y_onehot = np.eye(3)[y_true]
    brier = np.mean(np.sum((probs - y_onehot)**2, axis=1))

    # Log loss
    eps = 1e-12
    logloss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps))

    return accuracy, brier, logloss


# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnostic_row(row):
    issues = []

    # 1. Low confidence prediction
    if max(row["home_win_prob"], row["draw_prob"], row["away_win_prob"]) < 0.45:
        issues.append("Low model confidence")

    # 2. Big xG error
    if "exp_home_goals" in row and "exp_away_goals" in row:
        xg_home_err = abs(row["FTHG"] - row["exp_home_goals"])
        xg_away_err = abs(row["FTAG"] - row["exp_away_goals"])
        if xg_home_err > 1.5 or xg_away_err > 1.5:
            issues.append("Large xG deviation")

    # 3. Scoreline way off
    if row.get("most_likely_score_prob", 0) > 0 and (row["FTHG"] + row["FTAG"]) >= 3:
        ml_home, ml_away = 0, 0
        try:
            ml_home, ml_away = map(int, row["most_likely_score"].split("-"))
            if (ml_home + ml_away) <= 1 and (row["FTHG"] + row["FTAG"]) >= 3:
                issues.append("Model underestimated scoring environment")
        except:
            pass

    # 4. True randomness / upset
    if not issues:
        issues = ["Random variance / upset"]

    return "; ".join(issues)


def generate_diagnostics(df):
    df["Correct"] = df["Predicted"] == df["Actual"]
    df["Diagnostic"] = df.apply(diagnostic_row, axis=1)
    return df


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(df):
    full_path = OUT_DIR / "predictions_vs_actual.csv"
    wrong_path = OUT_DIR / "wrong_predictions.csv"
    diag_path = OUT_DIR / "diagnostics.csv"

    df.to_csv(full_path, index=False)
    df[df["Correct"] == False].to_csv(wrong_path, index=False)
    df.to_csv(diag_path, index=False)  # includes diagnostics

    print(f"\n📁 Saved evaluation outputs:")
    print(f" - Full merged results: {full_path}")
    print(f" - Wrong predictions:   {wrong_path}")
    print(f" - Diagnostics:         {diag_path}\n")


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_evaluation():

    print("\n===== RUNNING PREDICTION EVALUATION =====")

    pred = load_predictions()
    actual = load_actuals()

    # Only evaluate dates that we predicted
    dates_predicted = pred["Date"].unique()
    actual_eval = actual[actual["Date"].isin(dates_predicted)]

    merged = merge_pred_actual(pred, actual_eval)

    accuracy, brier, logloss = compute_metrics(merged)

    merged = generate_diagnostics(merged)

    save_outputs(merged)

    print("===== METRICS =====")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Brier Score:  {brier:.4f}")
    print(f"Log Loss:     {logloss:.4f}")

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    run_evaluation()
