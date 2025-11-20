import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def evaluate_backtest(backtest_path="models/history/backtest_fast.csv"):
    df = pd.read_csv(ROOT / backtest_path)

    required = ["pH", "pD", "pA", "Result", "FTHG", "FTAG",
                "ExpHomeGoals", "ExpAwayGoals", "ExpTotalGoals"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column in backtest file: {col}")

    # Drop bad rows
    df = df.dropna(subset=["pH", "pD", "pA", "Result"])

    # Convert results to numeric class labels
    mapping = {"H": 0, "D": 1, "A": 2}
    y_true = df["Result"].map(mapping).values
    y_pred = np.argmax(df[["pH", "pD", "pA"]].values, axis=1)

    # ---------- Accuracy ----------
    accuracy = (y_true == y_pred).mean()

    # ---------- Brier score ----------
    y_onehot = np.eye(3)[y_true]
    probs = df[["pH", "pD", "pA"]].values
    brier = np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

    # ---------- Log loss ----------
    eps = 1e-12
    log_loss = -np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps))

    # ---------- Goal MAE ----------
    mae_home = np.mean(np.abs(df["FTHG"] - df["ExpHomeGoals"]))
    mae_away = np.mean(np.abs(df["FTAG"] - df["ExpAwayGoals"]))
    mae_total = np.mean(np.abs((df["FTHG"] + df["FTAG"]) - df["ExpTotalGoals"]))

    # ---------- Draw accuracy ----------
    draw_mask = df["Result"] == "D"
    draw_acc = (np.argmax(probs[draw_mask], axis=1) == 1).mean() if draw_mask.sum() > 0 else None

    # ---------- Win/Loss accuracy ----------
    win_mask = df["Result"].isin(["H", "A"])
    wl_acc = (np.argmax(probs[win_mask], axis=1) == df["Result"][win_mask].map(mapping)).mean()

    # ---------- Over/Under model ----------
    df["Over2.5"] = (df["FTHG"] + df["FTAG"]) > 2.5
    df["Under2.5"] = (df["FTHG"] + df["FTAG"]) < 2.5

    df["Pred_Over2.5"] = df["ExpTotalGoals"] > 2.5
    df["Pred_Under2.5"] = df["ExpTotalGoals"] < 2.5

    over_acc = (df["Over2.5"] == df["Pred_Over2.5"]).mean()
    under_acc = (df["Under2.5"] == df["Pred_Under2.5"]).mean()

    # ---------- Print summary ----------
    print("\n================ MODEL EVALUATION ================\n")
    print(f"Matches evaluated:      {len(df):,}")
    print(f"Outcome accuracy:       {accuracy:.3f}")
    print(f"Brier score:            {brier:.3f}")
    print(f"Log loss:               {log_loss:.3f}")
    print()
    print(f"MAE Home Goals:         {mae_home:.3f}")
    print(f"MAE Away Goals:         {mae_away:.3f}")
    print(f"MAE Total Goals:        {mae_total:.3f}")
    print()
    print(f"Win/Loss Accuracy:      {wl_acc:.3f}")
    if draw_acc is not None:
        print(f"Draw Accuracy:          {draw_acc:.3f}")
    print()
    print(f"Over 2.5 Accuracy:      {over_acc:.3f}")
    print(f"Under 2.5 Accuracy:     {under_acc:.3f}")
    print("\n==================================================\n")

    return {
        "matches": len(df),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
        "wl_acc": wl_acc,
        "draw_acc": draw_acc,
        "over_acc": over_acc,
        "under_acc": under_acc
    }


if __name__ == "__main__":
    evaluate_backtest()
