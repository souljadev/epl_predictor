import pandas as pd
from pathlib import Path

# Paths
LIVE_FULL = Path("models/predictions/predictions_full.csv")
BACKTEST_EXP = Path("models/history/backtest_expanding_matchday.csv")
BACKTEST_FAST = Path("models/history/backtest_fast.csv")  # if present


def verify_alignment():
    print("\n=== VERIFY LIVE PREDICTIONS vs BACKTESTS ===\n")

    # Load live predictions
    if not LIVE_FULL.exists():
        print("❌ No live predictions found.")
        return
    live = pd.read_csv(LIVE_FULL)

    # Load expanding backtest
    if not BACKTEST_EXP.exists():
        print("❌ No expanding backtest found.")
        return
    exp = pd.read_csv(BACKTEST_EXP)

    print("✔ Loaded live predictions and expanding backtest.")

    # 1. Compare probability distribution
    print("\n=== Probability Distribution Comparison ===")
    print("\nLive Predictions:")
    print(live[["home_win_prob", "draw_prob", "away_win_prob"]].describe())

    print("\nExpanding Backtest:")
    print(exp[["pH", "pD", "pA"]].describe())

    # Check similarity in mean distributions
    live_mean = live[["home_win_prob", "draw_prob", "away_win_prob"]].mean()
    exp_mean = exp[["pH", "pD", "pA"]].mean()

    print("\nMean Difference (Live – Backtest):")
    print(live_mean - exp_mean)

    # Check calibration: probability mass of favorites
    live_fav = (live[["home_win_prob", "away_win_prob"]].max(axis=1)).mean()
    exp_fav = (exp[["pH", "pA"]].max(axis=1)).mean()

    print(f"\nFavorite Confidence:")
    print(f" Live: {live_fav:.3f}")
    print(f" Backtest: {exp_fav:.3f}")

    # Compare expected goals calibration
    if "ExpHomeGoals" in exp.columns:
        print("\n=== Expected Goals Calibration ===")
        print("\nLive:")
        print(live[["exp_home_goals", "exp_away_goals"]].describe())

        print("\nBacktest:")
        print(exp[["ExpHomeGoals", "ExpAwayGoals"]].describe())

    # If fast backtest exists
    if BACKTEST_FAST.exists():
        fast = pd.read_csv(BACKTEST_FAST)
        print("\n=== FAST Backtest Probability Distribution ===")
        print(fast[["pH", "pD", "pA"]].describe())

    print("\n=== ALIGNMENT CHECK COMPLETE ===")


if __name__ == "__main__":
    verify_alignment()