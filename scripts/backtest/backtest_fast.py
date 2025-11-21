import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.elo import EloModel
from src.models.dixon_coles import DixonColesModel

import pandas as pd
import numpy as np
import time
from math import exp, factorial


# -----------------------------------------------------
# Utility functions
# -----------------------------------------------------

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def poisson_pmf(k, lam):
    if lam is None or np.isnan(lam) or lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * lam**k / factorial(k)


def score_grid_probs(lam_home, lam_away, max_goals=6):
    home_p = [poisson_pmf(i, lam_home) for i in range(max_goals + 1)]
    away_p = [poisson_pmf(j, lam_away) for j in range(max_goals + 1)]

    grid = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            row.append(home_p[i] * away_p[j])
        grid.append(row)

    # Normalize
    total = sum(sum(r) for r in grid)
    if total > 0:
        grid = [[p / total for p in r] for r in grid]

    return grid


def extract_score_markets(grid, max_goals=6):
    best_h, best_a, best_p = 0, 0, -1.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if grid[i][j] > best_p:
                best_p = grid[i][j]
                best_h = i
                best_a = j

    btts_yes = sum(
        grid[i][j]
        for i in range(1, max_goals + 1)
        for j in range(1, max_goals + 1)
    )

    over_2_5 = sum(
        grid[i][j]
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
        if i + j >= 3
    )

    return {
        "most_likely_score": f"{best_h}-{best_a}",
        "most_likely_score_prob": best_p,
        "btts_yes": btts_yes,
        "btts_no": 1 - btts_yes,
        "over_2_5": over_2_5,
        "under_2_5": 1 - over_2_5,
    }


# -----------------------------------------------------
# MAIN BACKTEST
# -----------------------------------------------------

def backtest_fast():

    print("\n[START] FAST MATCHDAY backtest…\n")

    # Auto-load the correct file
    input_path = "data/raw/epl_combined.csv"

    print(f"Loading dataset: {input_path}")

    df = pd.read_csv(input_path)

    # Basic cleaning
    print("Checking data integrity…")

    # Drop duplicate rows
    before_dupes = len(df)
    df = df.drop_duplicates()
    after_dupes = len(df)
    if before_dupes != after_dupes:
        print(f"⚠ Removed {before_dupes - after_dupes} duplicate rows.")

    # Ensure correct datatypes
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Sort by date
    df = df.sort_values("Date").reset_index(drop=True)

    # Detect missing scores (fixtures)
    fixture_rows = df[df["FTHG"].isna() | df["FTAG"].isna()]
    if not fixture_rows.empty:
        print(f"⚠ Found {len(fixture_rows)} fixture rows with missing scores.")
        print("   These will be skipped and NOT used for training.\n")

    # Drop NaN score rows (fixtures)
    df = df.dropna(subset=["FTHG", "FTAG"])

    # Detect missing matchdays
    if "Matchday" not in df.columns:
        print("⚠ No Matchday column — generating sequential matchdays.")
        df["Matchday"] = df.groupby(df["Date"].dt.year).cumcount() + 1

    matchdays = sorted(df["Matchday"].unique())

    print(f"Total matches: {len(df)}")
    print(f"Total matchdays: {len(matchdays)}\n")

    # Initialize models
    dc = DixonColesModel()
    elo = EloModel()

    results = []

    overall_start = time.time()

    # MATCHDAY LOOP
    for md in matchdays:
        md_start = time.time()

        md_df = df[df["Matchday"] == md]

        for _, row in md_df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            fthg = int(row["FTHG"])
            ftag = int(row["FTAG"])

            # update models
            dc.update(home, away, fthg, ftag)
            elo.update(home, away, fthg, ftag)

            # ensemble
            dc_p = dc.predict(home, away)
            elo_p = elo.predict(home, away)

            h = 0.6 * dc_p[0] + 0.4 * elo_p[0]
            d = 0.6 * dc_p[1] + 0.4 * elo_p[1]
            a = 0.6 * dc_p[2] + 0.4 * elo_p[2]

            Z = h + d + a
            h, d, a = h / Z, d / Z, a / Z

            # expected goals
            lam_home, lam_away = dc.predict_expected_goals(home, away)

            # generate score grid
            grid = score_grid_probs(lam_home, lam_away)
            score_markets = extract_score_markets(grid)

            results.append({
                "Date": row["Date"],
                "HomeTeam": home,
                "AwayTeam": away,
                "home_win_prob": h,
                "draw_prob": d,
                "away_win_prob": a,
                "most_likely_score": score_markets["most_likely_score"],
                "most_likely_score_prob": score_markets["most_likely_score_prob"],
                "btts_yes": score_markets["btts_yes"],
                "btts_no": score_markets["btts_no"],
                "over_2_5": score_markets["over_2_5"],
                "under_2_5": score_markets["under_2_5"],
                "exp_home_goals": lam_home,
                "exp_away_goals": lam_away,
                "exp_total_goals": lam_home + lam_away,
            })

        elapsed = format_time(time.time() - md_start)
        print(f"Matchday {md} done in {elapsed}")

    # END
    total_time = format_time(time.time() - overall_start)
    print(f"\n[COMPLETE] Backtest finished in {total_time}\n")

    out = pd.DataFrame(results)
    output_path = "data/processed/backtest_fast_output.csv"
    out.to_csv(output_path, index=False)

    print(f"Saved predictions → {output_path}")


if __name__ == "__main__":
    backtest_fast()
