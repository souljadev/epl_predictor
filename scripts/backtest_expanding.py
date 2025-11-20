import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
from math import log

# ---------- Paths / imports ----------
ROOT = Path(__file__).resolve().parents[1]   # project root: soccer_agent_local
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))  # so we can import from src.models etc.

from models.poisson_dc import DixonColes
from models.elo import EloModel
from models.ensemble import ensemble_win_probs

# ---------- Config ----------
CONFIG_PATH = ROOT / "config.yaml"

# You can tweak this to limit how far back the backtest goes
BACKTEST_START_DATE = pd.Timestamp("2025-01-01")   # only backtest from this date onward
MIN_TRAIN_MATCHES = 500                            # skip very early days with tiny sample


def load_config(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text())
    return cfg


def compute_metrics(eval_df: pd.DataFrame):
    """Compute basic backtest metrics from eval_df."""
    mapping = {"H": 0, "D": 1, "A": 2}

    # Filter out rows without proper probs
    eval_df = eval_df.dropna(subset=["pH", "pD", "pA", "Result"])

    y_true = eval_df["Result"].map(mapping).values
    probs = eval_df[["pH", "pD", "pA"]].values

    # predicted class = argmax(prob)
    y_pred_idx = probs.argmax(axis=1)
    idx_to_label = {0: "H", 1: "D", 2: "A"}
    y_pred = np.vectorize(idx_to_label.get)(y_pred_idx)

    accuracy = (y_pred == eval_df["Result"].values).mean()

    # Brier score
    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    # Log loss
    eps = 1e-12
    log_probs_true = np.log(probs[np.arange(len(y_true)), y_true] + eps)
    log_loss = float(-np.mean(log_probs_true))

    # Score MAE
    mae_home = float(np.mean(np.abs(eval_df["FTHG"] - eval_df["ExpHomeGoals"])))
    mae_away = float(np.mean(np.abs(eval_df["FTAG"] - eval_df["ExpAwayGoals"])))
    mae_total = float(
        np.mean(
            np.abs(
                (eval_df["FTHG"] + eval_df["FTAG"])
                - (eval_df["ExpHomeGoals"] + eval_df["ExpAwayGoals"])
            )
        )
    )

    return {
        "n_matches": len(eval_df),
        "accuracy": accuracy,
        "brier": brier,
        "log_loss": log_loss,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "mae_total": mae_total,
    }


def backtest_expanding(config_path: Path):
    cfg = load_config(config_path)

    results_csv = cfg["data"]["results_csv"]
    ens_cfg = cfg.get("model", {}).get("ensemble", {})
    w_dc = ens_cfg.get("w_dc", 0.6)
    w_elo = ens_cfg.get("w_elo", 0.4)

    results_path = ROOT / results_csv
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    df = pd.read_csv(results_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure required columns exist
    needed_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"results_csv is missing columns: {missing}")

    # Filter to backtest window
    df_bt = df[df["Date"] >= BACKTEST_START_DATE].copy()
    unique_dates = sorted(df_bt["Date"].unique())

    eval_rows = []

    for d in unique_dates:
        train_mask = df["Date"] < d
        test_mask = df["Date"] == d

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if len(train_df) < MIN_TRAIN_MATCHES:
            # not enough history yet; skip this day
            continue

        print(f"Backtesting date {d.date()} | train: {len(train_df)} matches, test: {len(test_df)} matches")

        # Train fresh models on all data before this date
        dc = DixonColes(
            rho_init=cfg["model"]["dc"].get("rho_init", 0.0),
            max_iter=cfg["model"]["dc"].get("max_iter", 300),
            tol=cfg["model"]["dc"].get("tol", 1e-6),
        ).fit(train_df)

        elo = EloModel(
            k_factor=cfg["model"]["elo"].get("k_factor", 18.0),
            home_advantage=cfg["model"]["elo"].get("home_advantage", 55.0),
        ).fit(train_df)

        # Predict today’s matches
        for _, row in test_df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            try:
                dc_out = dc.match_probs(home, away)
                dc_wp = dc_out["win_probs"]
                lamH, lamA = dc_out["lambdas"]
                elo_wp = elo.predict_win_probs(home, away)
                pH, pD, pA = ensemble_win_probs(dc_wp, elo_wp, w_dc=w_dc, w_elo=w_elo)
            except KeyError:
                # unseen team or model issue; skip match
                continue

            eval_rows.append({
                "Date": d,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": row["FTHG"],
                "FTAG": row["FTAG"],
                "Result": row["Result"],
                "pH": float(pH),
                "pD": float(pD),
                "pA": float(pA),
                "ExpHomeGoals": float(lamH),
                "ExpAwayGoals": float(lamA),
                "ExpTotalGoals": float(lamH + lamA),
            })

    if not eval_rows:
        print("No evaluation rows generated. Check BACKTEST_START_DATE and data coverage.")
        return

    eval_df = pd.DataFrame(eval_rows)
    history_dir = ROOT / "models" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    out_path = history_dir / "backtest_expanding.csv"
    eval_df.to_csv(out_path, index=False)

    metrics = compute_metrics(eval_df)

    print("\n=== Backtest summary (expanding window) ===")
    print(f"Matches evaluated: {metrics['n_matches']}")
    print(f"Outcome accuracy: {metrics['accuracy']:.3f}")
    print(f"Brier score:      {metrics['brier']:.3f}")
    print(f"Log loss:         {metrics['log_loss']:.3f}")
    print(f"MAE home goals:   {metrics['mae_home']:.3f}")
    print(f"MAE away goals:   {metrics['mae_away']:.3f}")
    print(f"MAE total goals:  {metrics['mae_total']:.3f}")
    print(f"\nDetailed per-match results saved to: {out_path}")


if __name__ == "__main__":
    backtest_expanding(CONFIG_PATH)
