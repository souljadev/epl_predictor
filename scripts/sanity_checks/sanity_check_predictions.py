import pandas as pd
import numpy as np
from pathlib import Path

PRED_PATH = Path("models/predictions/predictions_full.csv")


def sanity_check():
    print("\n=== SANITY CHECK: predictions_full.csv ===\n")

    if not PRED_PATH.exists():
        print(f"❌ File not found: {PRED_PATH}")
        return

    df = pd.read_csv(PRED_PATH)

    required_cols = [
        "HomeTeam", "AwayTeam",
        "home_win_prob", "draw_prob", "away_win_prob",
        "most_likely_score", "most_likely_score_prob",
        "exp_home_goals", "exp_away_goals", "exp_total_goals",
    ]

    # Check for missing columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"❌ Missing required columns: {missing}")
        return
    print("✔ All required columns found.")

    # 1) Basic shape
    print(f"✔ Rows: {len(df)}")

    # 2) Probability sanity tests
    prob_sum = df["home_win_prob"] + df["draw_prob"] + df["away_win_prob"]
    bad_rows = df[np.abs(prob_sum - 1) > 1e-6]
    if len(bad_rows):
        print(f"❌ {len(bad_rows)} rows do not sum to 1. Showing sample:")
        print(bad_rows.head())
    else:
        print("✔ All match probabilities sum to 1.")

    # 3) Expected goals sanity tests
    neg_exp = df[(df["exp_home_goals"] < 0) | (df["exp_away_goals"] < 0)]
    if len(neg_exp):
        print(f"❌ Negative expected goals found in {len(neg_exp)} rows.")
        print(neg_exp.head())
    else:
        print("✔ No negative expected goals.")

    high_lambda = df[df["exp_total_goals"] > 6]
    if len(high_lambda):
        print(f"⚠ Warning: unusually high total xG in {len(high_lambda)} rows.")
        print(high_lambda.head())
    else:
        print("✔ Expected total goals are within normal range.")

    # 4) Most likely score probability sanity
    bad_score_prob = df[df["most_likely_score_prob"] > 1]
    if len(bad_score_prob):
        print(f"❌ Invalid score probabilities >1 in {len(bad_score_prob)} rows.")
    else:
        print("✔ Most-likely score probabilities are valid.")

    # 5) Distribution sanity
    print("\n=== Probability Distribution Summary ===")
    print(df[["home_win_prob", "draw_prob", "away_win_prob"]].describe())

    print("\n=== Sample Output ===")
    print(df.head(5))

    print("\n=== SANITY CHECK COMPLETE ===")


if __name__ == "__main__":
    sanity_check()
