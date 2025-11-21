import pandas as pd
from pathlib import Path

FULL_PATH = Path("models/predictions/predictions_full.csv")
EXACT_PATH = Path("models/predictions/predictions_exact_scores.csv")


def sanity_check_exact_scores():
    print("\n=== SANITY CHECK: Exact Score Probabilities ===\n")

    if not FULL_PATH.exists() or not EXACT_PATH.exists():
        print("❌ Missing predictions files.")
        return

    full = pd.read_csv(FULL_PATH)
    exact = pd.read_csv(EXACT_PATH)

    required_full = {"HomeTeam", "AwayTeam", "most_likely_score", "most_likely_score_prob"}
    required_exact = {"HomeTeam", "AwayTeam", "score", "prob"}

    if missing := required_full - set(full.columns):
        print(f"❌ Missing columns in predictions_full: {missing}")
        return

    if missing := required_exact - set(exact.columns):
        print(f"❌ Missing columns in predictions_exact_scores: {missing}")
        return

    # Group exact scores by match
    grouped = exact.groupby(["HomeTeam", "AwayTeam"])

    issues = []
    for _, row in full.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        if (home, away) not in grouped.groups:
            issues.append(f"Missing exact scores for {home} vs {away}")
            continue

        sub = grouped.get_group((home, away))

        # Sum of score probabilities
        total_prob = sub["prob"].sum()

        if abs(total_prob - 1) > 1e-6:
            issues.append(f"{home} vs {away}: score prob sum = {total_prob:.6f}")

        # Negative or >1 probabilities
        bad = sub[(sub["prob"] < 0) | (sub["prob"] > 1)]
        if len(bad):
            issues.append(f"{home} vs {away}: invalid score probabilities detected.")

        # Check that stored best score matches max prob
        best_score = sub.loc[sub["prob"].idxmax(), "score"]
        if best_score != row["most_likely_score"]:
            issues.append(
                f"{home} vs {away}: mismatch best score. "
                f"matrix={best_score}, stored={row['most_likely_score']}"
            )

    if issues:
        print("\n❌ ISSUES FOUND:")
        for i in issues:
            print(" -", i)
    else:
        print("✔ All exact score probabilities pass sanity checks.")

    print("\n=== Score Distribution (Top 10 Most Common Scores) ===")
    print(
        exact.groupby("score")["prob"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n=== SAMPLE EXACT SCORES ===")
    print(exact.head(20))

    print("\n=== SANITY CHECK COMPLETE ===")


if __name__ == "__main__":
    sanity_check_exact_scores()
