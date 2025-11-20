import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# =============== CONFIG ============================
BACKTEST_FILE = ROOT / "models/history/backtest_fast.csv"

# Optional odds file with real bookmaker odds.
# Expected at least: Date, HomeTeam, AwayTeam and odds columns.
# Example odds columns: OddsH, OddsD, OddsA OR home_odds, draw_odds, away_odds
ODDS_FILE = ROOT / "data/raw/odds_epl.csv"  # set to None if you don't have it

# Betting config
EDGECUTOFF = 0.01        # minimum edge (p_model - p_implied) to place a bet
CONF_PROB_MIN = 0.45     # only bet if model probability >= this (confidence filter)
STAKE = 100              # flat stake (per bet) used for flat strategy
KELLY_FRACTION = 0.5     # % of full Kelly
START_BANKROLL = 10_000  # starting bankroll for both strategies

# Odds sanity
MAX_OVERROUND = 0.25     # skip matches where sum(implied_probs) is too crazy
# ====================================================


def implied_prob(odds):
    """Convert decimal odds → implied probability."""
    return 1 / odds if odds and odds > 0 else np.nan


def kelly_fraction(p, odds):
    """Full Kelly formula for a single-outcome bet."""
    b = odds - 1
    if b <= 0:
        return 0.0
    return (p * (b + 1) - 1) / b


def normalize_backtest_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure df has pH, pD, pA, Result.
    This lets you be flexible with backtest_fast output names.
    """
    # Probabilities
    if "pH" not in df.columns:
        if "home_win_prob" in df.columns:
            df["pH"] = df["home_win_prob"]
        else:
            raise ValueError("Backtest file missing pH or home_win_prob")

    if "pD" not in df.columns:
        if "draw_prob" in df.columns:
            df["pD"] = df["draw_prob"]
        else:
            raise ValueError("Backtest file missing pD or draw_prob")

    if "pA" not in df.columns:
        if "away_win_prob" in df.columns:
            df["pA"] = df["away_win_prob"]
        else:
            raise ValueError("Backtest file missing pA or away_win_prob")

    # Result label (H/D/A)
    if "Result" not in df.columns:
        if "FTR" in df.columns:
            df["Result"] = df["FTR"]
        else:
            # Try numeric result label if you use 0/1/2
            if "true_label" in df.columns:
                mapping = {0: "H", 1: "D", 2: "A"}
                df["Result"] = df["true_label"].map(mapping)
            else:
                raise ValueError("Backtest file missing Result/FTR/true_label")

    return df


def load_real_odds() -> pd.DataFrame | None:
    """Load real odds if ODDS_FILE is set and exists."""
    if ODDS_FILE is None or not Path(ODDS_FILE).exists():
        return None

    odds = pd.read_csv(ODDS_FILE)

    # Expect at least Date, HomeTeam, AwayTeam
    required = {"Date", "HomeTeam", "AwayTeam"}
    if not required.issubset(odds.columns):
        raise ValueError(f"Odds file must contain columns: {required}")

    # Detect odds columns
    col_map = {}
    # Preferred names
    for k, candidates in {
        "OddsH": ["OddsH", "home_odds", "HomeOdds", "home_win_odds"],
        "OddsD": ["OddsD", "draw_odds", "DrawOdds", "draw_win_odds"],
        "OddsA": ["OddsA", "away_odds", "AwayOdds", "away_win_odds"],
    }.items():
        for c in candidates:
            if c in odds.columns:
                col_map[k] = c
                break

    if len(col_map) != 3:
        raise ValueError(
            "Could not infer odds columns. "
            "Expected something like OddsH/OddsD/OddsA or home_odds/draw_odds/away_odds."
        )

    odds = odds.rename(columns={
        col_map["OddsH"]: "OddsH",
        col_map["OddsD"]: "OddsD",
        col_map["OddsA"]: "OddsA"
    })

    return odds[["Date", "HomeTeam", "AwayTeam", "OddsH", "OddsD", "OddsA"]]


def load_backtest() -> pd.DataFrame:
    df = pd.read_csv(BACKTEST_FILE)
    df = normalize_backtest_columns(df)

    # If you store Date as string in different formats, you can normalize here
    # df["Date"] = pd.to_datetime(df["Date"]).dt.date

    real_odds = load_real_odds()
    if real_odds is not None:
        print("Merging real odds into backtest...")
        df = df.merge(
            real_odds,
            on=["Date", "HomeTeam", "AwayTeam"],
            how="left",
            suffixes=("", "_odds")
        )

    # If real odds missing, fall back to synthetic odds
    if "OddsH" not in df.columns or df["OddsH"].isna().all():
        print("No real odds found or all missing → using synthetic odds from model probs.")
        df["OddsH"] = 1 / df["pH"] * 1.05
        df["OddsD"] = 1 / df["pD"] * 1.05
        df["OddsA"] = 1 / df["pA"] * 1.05

    return df


def odds_row_sane(row) -> bool:
    """Basic sanity on odds / implied probabilities."""
    if any(pd.isna([row["OddsH"], row["OddsD"], row["OddsA"]])):
        return False

    impH = implied_prob(row["OddsH"])
    impD = implied_prob(row["OddsD"])
    impA = implied_prob(row["OddsA"])

    if any(pd.isna([impH, impD, impA])):
        return False

    s = impH + impD + impA
    # Overround in [0.9, 1+MAX_OVERROUND]
    if s < 0.9 or s > 1 + MAX_OVERROUND:
        return False

    return True


def simulate_bets(df: pd.DataFrame, use_kelly: bool = False, label: str = "flat") -> pd.DataFrame:
    bank = START_BANKROLL
    results = []

    for _, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]

        if not odds_row_sane(row):
            continue

        # Model probabilities
        pH, pD, pA = row["pH"], row["pD"], row["pA"]

        # Implied bookmaker probabilities
        impH = implied_prob(row["OddsH"])
        impD = implied_prob(row["OddsD"])
        impA = implied_prob(row["OddsA"])

        bets = {
            "H": (pH, impH, row["OddsH"]),
            "D": (pD, impD, row["OddsD"]),
            "A": (pA, impA, row["OddsA"]),
        }

        for outcome, (p, imp, odds) in bets.items():
            if pd.isna(p) or pd.isna(imp) or pd.isna(odds):
                continue

            # Confidence + edge filters
            edge = p - imp
            if p < CONF_PROB_MIN:
                continue
            if edge < EDGECUTOFF:
                continue

            # Stake size
            if use_kelly:
                frac = max(0, kelly_fraction(p, odds))
                stake = bank * frac * KELLY_FRACTION
            else:
                stake = STAKE

            if stake < 1:
                continue

            won = (row["Result"] == outcome)
            pnl = stake * (odds - 1) if won else -stake
            bank += pnl

            # Assign team for by-team breakdown: H bet → home team, A bet → away team, D → DRAW
            if outcome == "H":
                bet_team = home
            elif outcome == "A":
                bet_team = away
            else:
                bet_team = "DRAW"

            # Odds bucket
            odds_bucket = pd.cut(
                [odds],
                bins=[1.0, 1.5, 2.0, 3.0, 5.0, 100.0],
                labels=["1.00–1.49", "1.50–1.99", "2.00–2.99", "3.00–4.99", "5.00+"]
            )[0]

            results.append({
                "Date": row["Date"],
                "HomeTeam": home,
                "AwayTeam": away,
                "BetOn": outcome,
                "BetTeam": bet_team,
                "ModelProb": p,
                "ImpProb": imp,
                "Edge": edge,
                "Odds": odds,
                "Stake": stake,
                "Won": int(won),
                "PnL": pnl,
                "Bankroll": bank,
                "Strategy": label,
                "OddsBucket": odds_bucket
            })

    return pd.DataFrame(results)


def evaluate_betting(df: pd.DataFrame) -> dict:
    if df.empty:
        print("WARNING: no bets were placed for this strategy.")
        return {
            "bets": 0,
            "wins": 0,
            "total_profit": 0.0,
            "ROI": 0.0,
            "final_bankroll": START_BANKROLL,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "avg_edge": 0.0,
        }

    total_pnl = df["PnL"].sum()
    n_bets = len(df)
    wins = df["Won"].sum()

    # Use actual total stake (important for Kelly)
    total_staked = df["Stake"].sum()
    roi = total_pnl / total_staked if total_staked > 0 else 0.0

    final_bankroll = df["Bankroll"].iloc[-1]
    hit_rate = wins / n_bets
    avg_edge = df["Edge"].mean()

    # Max drawdown
    bankroll_series = df["Bankroll"].cummax() - df["Bankroll"]
    max_dd = bankroll_series.max()

    return {
        "bets": int(n_bets),
        "wins": int(wins),
        "total_profit": float(total_pnl),
        "ROI": float(roi),
        "final_bankroll": float(final_bankroll),
        "max_drawdown": float(max_dd),
        "hit_rate": float(hit_rate),
        "avg_edge": float(avg_edge),
    }


def breakdown_by_team(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("BetTeam").agg(
        bets=("BetTeam", "count"),
        wins=("Won", "sum"),
        profit=("PnL", "sum"),
        avg_edge=("Edge", "mean")
    )
    grp["hit_rate"] = grp["wins"] / grp["bets"]
    grp["roi"] = grp["profit"] / df.groupby("BetTeam")["Stake"].sum()
    return grp.sort_values("profit", ascending=False)


def breakdown_by_odds_bucket(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("OddsBucket").agg(
        bets=("Odds", "count"),
        wins=("Won", "sum"),
        profit=("PnL", "sum"),
        avg_edge=("Edge", "mean")
    )
    grp["hit_rate"] = grp["wins"] / grp["bets"]
    grp["roi"] = grp["profit"] / df.groupby("OddsBucket")["Stake"].sum()
    return grp.sort_values("OddsBucket")


def save_bankroll_curve(df_flat: pd.DataFrame, df_kelly: pd.DataFrame, out_path: Path):
    """Save bankroll evolution for both strategies as CSV for plotting."""
    if df_flat.empty and df_kelly.empty:
        return

    frames = []
    if not df_flat.empty:
        tmp = df_flat[["Date", "Bankroll"]].copy()
        tmp["Strategy"] = "flat"
        frames.append(tmp)
    if not df_kelly.empty:
        tmp = df_kelly[["Date", "Bankroll"]].copy()
        tmp["Strategy"] = "kelly"
        frames.append(tmp)

    curve = pd.concat(frames, ignore_index=True)
    curve.to_csv(out_path, index=False)


def run():
    print("Loading backtest…")
    df = load_backtest()

    print("Running flat-stake simulation…")
    flat_df = simulate_bets(df, use_kelly=False, label="flat")
    flat_metrics = evaluate_betting(flat_df)

    print("Running Kelly simulation…")
    kelly_df = simulate_bets(df, use_kelly=True, label="kelly")
    kelly_metrics = evaluate_betting(kelly_df)

    out_dir = ROOT / "models/history"
    out_dir.mkdir(exist_ok=True)

    flat_path = out_dir / "betting_flat.csv"
    kelly_path = out_dir / "betting_kelly.csv"
    flat_df.to_csv(flat_path, index=False)
    kelly_df.to_csv(kelly_path, index=False)

    # Bankroll curve CSV
    curve_path = out_dir / "betting_bankroll_curve.csv"
    save_bankroll_curve(flat_df, kelly_df, curve_path)

    # Breakdowns
    team_flat = breakdown_by_team(flat_df)
    team_kelly = breakdown_by_team(kelly_df)
    odds_flat = breakdown_by_odds_bucket(flat_df)
    odds_kelly = breakdown_by_odds_bucket(kelly_df)

    team_flat.to_csv(out_dir / "betting_team_flat.csv")
    team_kelly.to_csv(out_dir / "betting_team_kelly.csv")
    odds_flat.to_csv(out_dir / "betting_oddsbuckets_flat.csv")
    odds_kelly.to_csv(out_dir / "betting_oddsbuckets_kelly.csv")

    print("\n================= BETTING RESULTS =================")
    print("FLAT STAKE:")
    print(f"  Bets:            {flat_metrics['bets']}")
    print(f"  Wins:            {flat_metrics['wins']}")
    print(f"  Hit Rate:        {flat_metrics['hit_rate']:.3f}")
    print(f"  Profit:          {flat_metrics['total_profit']:.2f}")
    print(f"  ROI:             {flat_metrics['ROI']:.3f}")
    print(f"  Final Bankroll:  {flat_metrics['final_bankroll']:.2f}")
    print(f"  Max Drawdown:    {flat_metrics['max_drawdown']:.2f}")
    print(f"  Avg Edge:        {flat_metrics['avg_edge']:.4f}")
    print("--------------------------------------------------")
    print("KELLY:")
    print(f"  Bets:            {kelly_metrics['bets']}")
    print(f"  Wins:            {kelly_metrics['wins']}")
    print(f"  Hit Rate:        {kelly_metrics['hit_rate']:.3f}")
    print(f"  Profit:          {kelly_metrics['total_profit']:.2f}")
    print(f"  ROI:             {kelly_metrics['ROI']:.3f}")
    print(f"  Final Bankroll:  {kelly_metrics['final_bankroll']:.2f}")
    print(f"  Max Drawdown:    {kelly_metrics['max_drawdown']:.2f}")
    print(f"  Avg Edge:        {kelly_metrics['avg_edge']:.4f}")
    print("===================================================\n")

    print("Saved:")
    print(" - Flat stake      →", flat_path)
    print(" - Kelly           →", kelly_path)
    print(" - Bankroll curve  →", curve_path)
    print(" - Team breakdowns → betting_team_flat.csv / betting_team_kelly.csv")
    print(" - Odds buckets    → betting_oddsbuckets_flat.csv / betting_oddsbuckets_kelly.csv")


if __name__ == "__main__":
    run()
