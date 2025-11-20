import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.elo import EloModel
from models.ensemble import ensemble_win_probs


# ============================================================
# Parameters for FAST DC (Poisson) incremental updating
# ============================================================

ATTACK_DEFAULT = 1.0
DEFENSE_DEFAULT = 1.0
HOME_ADV = 0.15        # Slight home edge for expected goals
DC_DECAY = 0.005        # Smoothing factor: smaller = slower update, bigger = more reactive

# ============================================================

def load_config():
    return yaml.safe_load((ROOT / "config.yaml").read_text())


def initialize_team_strengths(teams):
    """Initialize attack & defense strengths for every team."""
    return {
        t: {"attack": ATTACK_DEFAULT, "defense": DEFENSE_DEFAULT}
        for t in teams
    }


def update_strengths(strengths, home, away, FTHG, FTAG):
    """Incremental Poisson-style update based on goals scored vs expected."""
    # Get old strengths
    ha = strengths[home]["attack"]
    hd = strengths[home]["defense"]
    aa = strengths[away]["attack"]
    ad = strengths[away]["defense"]

    # Expected goals (Poisson, simplified DC)
    exp_home = np.exp(np.log(ha) - np.log(ad) + HOME_ADV)
    exp_away = np.exp(np.log(aa) - np.log(hd))

    # Update rules: smoothed toward actual goals
    strengths[home]["attack"]  += DC_DECAY * (FTHG - exp_home)
    strengths[home]["defense"] += DC_DECAY * (exp_away - FTAG)
    strengths[away]["attack"]  += DC_DECAY * (FTAG - exp_away)
    strengths[away]["defense"] += DC_DECAY * (exp_home - FTHG)

    # Prevent negative values
    for t in (home, away):
        strengths[t]["attack"] = max(0.01, strengths[t]["attack"])
        strengths[t]["defense"] = max(0.01, strengths[t]["defense"])

    return exp_home, exp_away


def predict_exp(strengths, home, away):
    """Expected goals using the incremental DC model."""
    ha = strengths[home]["attack"]
    hd = strengths[home]["defense"]
    aa = strengths[away]["attack"]
    ad = strengths[away]["defense"]

    exp_home = np.exp(np.log(ha) - np.log(ad) + HOME_ADV)
    exp_away = np.exp(np.log(aa) - np.log(hd))

    return exp_home, exp_away


def backtest_fast():
    cfg = load_config()

    results_path = ROOT / cfg["data"]["results_csv"]
    df = pd.read_csv(results_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    # Collect unique teams
    teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))

    # Initialize strengths
    strengths = initialize_team_strengths(teams)

    # Initialize Elo incrementally
    elo_cfg = cfg["model"]["elo"]
    elo = EloModel(k_factor=elo_cfg["k_factor"], home_advantage=elo_cfg["home_advantage"])

    # But we don't fit Elo once — we update it match-by-match
    elo_ratings = elo.init_ratings(teams)

    eval_rows = []

    for idx, row in df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        FTHG, FTAG = row["FTHG"], row["FTAG"]
        result = row["Result"]
        date = row["Date"]

        # FAST DC prediction (before update)
        exp_home, exp_away = predict_exp(strengths, home, away)

        # Elo prediction (before update)
        elo_probs = elo.predict_win_probs_raw(elo_ratings, home, away)

        # Weighted ensemble
        w_dc = cfg["model"]["ensemble"]["w_dc"]
        w_elo = cfg["model"]["ensemble"]["w_elo"]

        # Poisson-based win probabilities (fast version)
        p_home_dc = exp_home / (exp_home + exp_away + 1e-8)
        p_away_dc = exp_away / (exp_home + exp_away + 1e-8)
        p_draw_dc = 1 - (p_home_dc + p_away_dc)
        p_draw_dc = max(0, min(1, p_draw_dc))

        dc_probs = {"H": p_home_dc, "D": p_draw_dc, "A": p_away_dc}

        # Final ensemble
        pH, pD, pA = ensemble_win_probs(dc_probs, elo_probs, w_dc=w_dc, w_elo=w_elo)

        # Store prediction
        eval_rows.append({
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": FTHG,
            "FTAG": FTAG,
            "Result": result,
            "pH": pH,
            "pD": pD,
            "pA": pA,
            "ExpHomeGoals": exp_home,
            "ExpAwayGoals": exp_away,
            "ExpTotalGoals": exp_home + exp_away
        })

        # AFTER PREDICTING: update strengths based on actuals
        update_strengths(strengths, home, away, FTHG, FTAG)

        # Update Elo ratings incrementally
        elo_ratings = elo.update_ratings_match(elo_ratings, home, away, FTHG, FTAG)

    # Save results
    out_dir = ROOT / "models/history"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest_fast.csv"
    pd.DataFrame(eval_rows).to_csv(out_path, index=False)

    print(f"\n⚡ FAST BACKTEST COMPLETE → {out_path}")
    print("Runtime: ~5–10 seconds on your machine.")


if __name__ == "__main__":
    backtest_fast()
