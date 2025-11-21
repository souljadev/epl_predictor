# scripts/backtest/backtest_rolling.py

import pandas as pd
import numpy as np
import yaml
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

from tqdm import tqdm

# ----------------- PATH SETUP -----------------
ROOT = Path(__file__).resolve().parents[2]  # soccer_agent_local/
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.poisson_dc import DixonColes
from models.elo import EloModel
from models.ensemble import ensemble_win_probs

CONFIG_PATH = ROOT / "config.yaml"
HISTORY_DIR = ROOT / "models" / "history"
HISTORY_DIR.mkdir(exist_ok=True)

# Rolling window length (days)
ROLLING_WINDOW_DAYS = 730  # ~2 seasons
MIN_TRAIN_MATCHES = 200    # ignore early dates with too little data


# ----------------- METRICS (same as expanding) -----------------
def compute_metrics(eval_df: pd.DataFrame):
    mapping = {"H": 0, "D": 1, "A": 2}
    eval_df = eval_df.dropna(subset=["pH", "pD", "pA", "Result"])

    y_true = eval_df["Result"].map(mapping).values
    probs = eval_df[["pH", "pD", "pA"]].values

    y_pred_idx = probs.argmax(axis=1)
    idx_to_label = {0: "H", 1: "D", 2: "A"}
    y_pred = np.vectorize(idx_to_label.get)(y_pred_idx)

    accuracy = (y_pred == eval_df["Result"].values).mean()

    # Brier Score
    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # Log Loss
    eps = 1e-12
    ll = np.log(probs[np.arange(len(y_true)), y_true] + eps)
    log_loss = float(-np.mean(ll))

    # Goal MAE (from DC xG)
    mae_home  = float(np.mean(np.abs(eval_df["FTHG"] - eval_df["ExpHomeGoals"])))
    mae_away  = float(np.mean(np.abs(eval_df["FTAG"] - eval_df["ExpAwayGoals"])))
    mae_total = float(np.mean(np.abs(
        (eval_df["FTHG"] + eval_df["FTAG"]) -
        (eval_df["ExpHomeGoals"] + eval_df["ExpAwayGoals"])
    )))

    return {
        "n_matches": len(eval_df),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
    }


def load_config(config_path: Path):
    return yaml.safe_load(config_path.read_text())


# ----------------- ROLLING BACKTEST -----------------
def backtest_rolling(config: dict | None = None,
                     config_path: Path | None = None,
                     run_tag: str | None = None):

    start_global = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting ROLLING-WINDOW backtest…")

    # Load config
    if config is None:
        if config_path is None:
            config_path = CONFIG_PATH
        cfg = load_config(config_path)
    else:
        cfg = config

    # Load results data
    results_path = ROOT / cfg["data"]["results_csv"]
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    df = pd.read_csv(results_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column from results: {col}")

    # Filter to rows with actual scores
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()

    # Unique match dates
    dates = sorted(df["Date"].unique().tolist())
    print(f"Total match dates (with results): {len(dates)}")

    eval_rows = []

    for i, d in enumerate(tqdm(dates, desc="Dates")):
        window_start = d - pd.Timedelta(days=ROLLING_WINDOW_DAYS)

        # Train on window (strictly before d)
        train_mask = (df["Date"] < d) & (df["Date"] >= window_start)
        test_mask  = df["Date"] == d

        train_df = df.loc[train_mask]
        test_df  = df.loc[test_mask]

        if len(train_df) < MIN_TRAIN_MATCHES:
            # Not enough history yet
            continue

        try:
            # Fit Dixon–Coles on xG or goals (depending on your model)
            dc = DixonColes(
                rho_init=cfg["model"]["dc"].get("rho_init", 0.0),
                max_iter=cfg["model"]["dc"].get("max_iter", 200),
                tol=cfg["model"]["dc"].get("tol", 1e-6),
            ).fit(train_df)

            # Fit Elo on same training window
            elo = EloModel(
                k_factor=cfg["model"]["elo"].get("k_factor", 18.0),
                home_advantage=cfg["model"]["elo"].get("home_advantage", 55.0),
            ).fit(train_df)

        except Exception as e:
            print(f"\n⚠ Training error on date {d.date()}: {e}")
            continue

        # Predict for that date
        for _, row in test_df.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            try:
                dc_out = dc.match_probs(home, away)
                lamH, lamA = dc_out["lambdas"]
                dc_wp = dc_out["win_probs"]

                elo_wp = elo.predict_win_probs(home, away)

                pH, pD, pA = ensemble_win_probs(
                    dc_wp, elo_wp,
                    w_dc=cfg["model"]["ensemble"]["w_dc"],
                    w_elo=cfg["model"]["ensemble"]["w_elo"],
                )
            except Exception:
                continue

            eval_rows.append({
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
                "TrainMatches": len(train_df),
                "WindowStart": window_start,
                "WindowEnd": d - pd.Timedelta(days=1),
            })

    if not eval_rows:
        print("⚠ No evaluation rows generated. Check data and window settings.")
        return None, None

    eval_df = pd.DataFrame(eval_rows)

    suffix = f"_{run_tag}" if run_tag else ""
    out_path = HISTORY_DIR / f"backtest_rolling_matchday{suffix}.csv"
    eval_df.to_csv(out_path, index=False)

    print(f"\nSaved rolling-window backtest → {out_path}")

    metrics = compute_metrics(eval_df)

    print("\n===== ROLLING-WINDOW BACKTEST SUMMARY =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    total_elapsed = time.time() - start_global
    print(f"\n⏱ Total runtime: {total_elapsed/60:.2f} minutes")

    return metrics, out_path


if __name__ == "__main__":
    backtest_rolling(config_path=CONFIG_PATH)
