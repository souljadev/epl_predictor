import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.dixon_coles import DixonColesModel
from models.elo import EloModel
from models.ensemble import ensemble_win_probs

CONFIG_PATH = ROOT / "config.yaml"
OUT_DIR = ROOT / "models" / "history"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "backtest_rolling_fast.csv"

BACKTEST_START_DATE = pd.Timestamp("2016-08-01")
ROLLING_WINDOW_DAYS = 730
MIN_TRAIN_MATCHES = 100


def load_config():
    import yaml
    return yaml.safe_load(CONFIG_PATH.read_text())


def outcome_label(hg, ag):
    if pd.isna(hg) or pd.isna(ag):
        return None
    hg, ag = int(hg), int(ag)
    if hg > ag:
        return "H"
    if hg < ag:
        return "A"
    return "D"


def backtest_rolling_fast(config=None):
    start = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting FAST ROLLING backtest")

    cfg = config or load_config()
    results_path = ROOT / cfg["data"]["results_csv"]
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    df = pd.read_csv(results_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    before = len(df)
    df = df.dropna(subset=["FTHG", "FTAG"])
    dropped = before - len(df)
    if dropped:
        print(f"WARNING: Dropped {dropped} rows with missing FTHG/FTAG")

    if "Result" not in df.columns:
        df["Result"] = df.apply(lambda r: outcome_label(r["FTHG"], r["FTAG"]), axis=1)

    df = df[df["Date"] >= BACKTEST_START_DATE].copy()
    if df.empty:
        print("No matches after BACKTEST_START_DATE")
        return

    unique_dates = sorted(df["Date"].unique())
    print(f"Total match dates (rolling_fast): {len(unique_dates)}")

    dc_cfg = cfg.get("model", {}).get("dc", {})
    elo_cfg = cfg.get("model", {}).get("elo", {})

    rows = []

    for i, date in enumerate(unique_dates):
        day_matches = df[df["Date"] == date]
        window_start = date - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
        history = df[(df["Date"] < date) & (df["Date"] >= window_start)]
        history = history.dropna(subset=["FTHG", "FTAG"])
        if len(history) < MIN_TRAIN_MATCHES:
            continue

        print(
            f"[{i+1}/{len(unique_dates)}] {date.date()} - "
            f"training: {len(history)}, predicting: {len(day_matches)}"
        )

        dc = DixonColesModel(
            rho_init=dc_cfg.get("rho_init", 0.0),
            home_adv_init=dc_cfg.get("home_adv_init", 0.15),
            lr=dc_cfg.get("lr", 0.05),
        )
        use_xg = "Home_xG" in history.columns and "Away_xG" in history.columns
        dc.fit(history, use_xg=use_xg)

        elo = EloModel(
            k_factor=elo_cfg.get("k_factor", 18.0),
            home_advantage=elo_cfg.get("home_advantage", 55.0),
            base_rating=elo_cfg.get("base_rating", 1500.0),
            draw_base=elo_cfg.get("draw_base", 0.25),
            draw_max_extra=elo_cfg.get("draw_max_extra", 0.10),
            draw_scale=elo_cfg.get("draw_scale", 400.0),
        )
        elo.fit(history)

        for _, row in day_matches.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            try:
                lamH, lamA = dc.predict_expected_goals(home, away)
                pH_dc, pD_dc, pA_dc = dc.predict(home, away)
            except Exception:
                continue

            try:
                pH_elo, pD_elo, pA_elo = elo.predict(home, away)
            except Exception:
                continue

            pH, pD, pA = ensemble_win_probs(
                (pH_dc, pD_dc, pA_dc),
                (pH_elo, pD_elo, pA_elo),
                w_dc=0.6,
                w_elo=0.4,
            )

            rows.append(
                {
                    "Date": row["Date"],
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": row["FTHG"],
                    "FTAG": row["FTAG"],
                    "Result": row["Result"],
                    "pH": pH,
                    "pD": pD,
                    "pA": pA,
                    "ExpHomeGoals": lamH,
                    "ExpAwayGoals": lamA,
                    "ExpTotalGoals": lamH + lamA,
                    "TrainMatches": len(history),
                }
            )

    if not rows:
        print("WARNING: Rolling FAST backtest produced no rows.")
        return

    out_df = pd.DataFrame(rows).sort_values("Date")
    out_df.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - start
    print(f"Saved FAST rolling backtest -> {OUT_PATH}")
    print(f"Total runtime: {elapsed/60:.2f} minutes")
    print(f"Total matches evaluated: {len(out_df)}")


if __name__ == "__main__":
    backtest_rolling_fast()
