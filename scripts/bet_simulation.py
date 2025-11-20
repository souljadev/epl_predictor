import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# =============== CONFIG ============================
BACKTEST_FILE = "models/history/backtest_fast.csv"
ODDS_FILE = None   # Optional: data/raw/odds.csv
EDGECUTOFF = 0.05  # 5% edge required to bet (prob - implied)
STAKE = 100         # Flat stake (per bet)
KELLY_FRACTION = 0.5  # 50% Kelly
# ====================================================


def implied_prob(odds):
    """Convert decimal odds → implied probability."""
    return 1 / odds if odds > 0 else np.nan


def kelly_fraction(p, odds):
    """Full Kelly formula for match result bets."""
    b = odds - 1
    return (p * (b + 1) - 1) / b


def simulate_bets(df, use_kelly=False):
    bank = 10000  # starting bankroll
    results = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        # Skip if odds missing
        if any(pd.isna([row["OddsH"], row["OddsD"], row["OddsA"]])):
            continue

        # Model probabilities
        pH, pD, pA = row["pH"], row["pD"], row["pA"]

        # Implied odds probability
        impH = implied_prob(row["OddsH"])
        impD = implied_prob(row["OddsD"])
        impA = implied_prob(row["OddsA"])

        bets = {
            "H": (pH, impH, row["OddsH"]),
            "D": (pD, impD, row["OddsD"]),
            "A": (pA, impA, row["OddsA"])
        }

        for outcome, (p, imp, odds) in bets.items():
            edge = p - imp
            if edge < EDGECUTOFF:
                continue  # no bet

            if use_kelly:
                frac = max(0, kelly_fraction(p, odds))
                stake = bank * frac * KELLY_FRACTION
            else:
                stake = STAKE

            if stake < 1:
                continue

            # Determine if won
            won = (row["Result"] == outcome)
            pnl = stake * (odds - 1) if won else -stake

            bank += pnl

            results.append({
                "Date": row["Date"],
                "HomeTeam": home,
                "AwayTeam": away,
                "BetOn": outcome,
                "ModelProb": p,
                "ImpProb": imp,
                "Edge": edge,
                "Odds": odds,
                "Stake": stake,
                "PnL": pnl,
                "Bankroll": bank
            })

    return pd.DataFrame(results)


def load_backtest():
    df = pd.read_csv(ROOT / BACKTEST_FILE)

    # If no odds file is provided, create synthetic odds from probabilities
    # This allows simulation even without bookmaker data.
    df["OddsH"] = 1 / df["pH"] * 1.05
    df["OddsD"] = 1 / df["pD"] * 1.05
    df["OddsA"] = 1 / df["pA"] * 1.05

    return df


def evaluate_betting(df):
    total_pnl = df["PnL"].sum()
    n_bets = len(df)
    roi = total_pnl / (n_bets * STAKE) if n_bets > 0 else 0
    final_bankroll = df["Bankroll"].iloc[-1] if n_bets > 0 else 10000

    return {
        "bets": n_bets,
        "total_profit": total_pnl,
        "ROI": roi,
        "final_bankroll": final_bankroll
    }


def run():
    print("Loading backtest…")
    df = load_backtest()

    print("Running flat-stake simulation…")
    flat_df = simulate_bets(df, use_kelly=False)
    flat_metrics = evaluate_betting(flat_df)

    print("Running Kelly simulation…")
    kelly_df = simulate_bets(df, use_kelly=True)
    kelly_metrics = evaluate_betting(kelly_df)

    out_dir = ROOT / "models/history"
    out_dir.mkdir(exist_ok=True)

    flat_path = out_dir / "betting_flat.csv"
    kelly_path = out_dir / "betting_kelly.csv"
    flat_df.to_csv(flat_path, index=False)
    kelly_df.to_csv(kelly_path, index=False)

    print("\n================= BETTING RESULTS =================")
    print(f"Total Bets:          {flat_metrics['bets']}")
    print(f"Flat-Stake Profit:   {flat_metrics['total_profit']:.2f}")
    print(f"Flat-Stake ROI:      {flat_metrics['ROI']:.3f}")
    print(f"Flat Final Bankroll: {flat_metrics['final_bankroll']:.2f}")
    print("--------------------------------------------------")
    print(f"Kelly Profit:        {kelly_metrics['total_profit']:.2f}")
    print(f"Kelly Final Bankroll:{kelly_metrics['final_bankroll']:.2f}")
    print("===================================================\n")

    print("Saved:")
    print(" - Flat stake →", flat_path)
    print(" - Kelly      →", kelly_path)


if __name__ == "__main__":
    run()
