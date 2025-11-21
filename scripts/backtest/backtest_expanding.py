import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import sys
import time
from datetime import datetime



# ---------- Paths / imports ----------
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from models.dixon_coles import DixonColesModel
from models.elo import EloModel
from models.ensemble import ensemble_win_probs
import inspect
print("DC loaded from:", inspect.getfile(DixonColesModel))


# ---------- Config ----------
CONFIG_PATH = ROOT / "config.yaml"
RESUME_PATH = ROOT / "models" / "history" / "backtest_expanding_resume.txt"

BACKTEST_START_DATE = pd.Timestamp("2015-01-01")
MIN_TRAIN_MATCHES = 200


def load_config(config_path: Path):
    return yaml.safe_load(config_path.read_text())


# ---------- Metrics ----------
def compute_metrics(eval_df: pd.DataFrame):
    mapping = {"H": 0, "D": 1, "A": 2}
    eval_df = eval_df.dropna(subset=["pH", "pD", "pA", "Result"])

    y_true = eval_df["Result"].map(mapping).values
    probs = eval_df[["pH", "pD", "pA"]].values

    y_pred_idx = probs.argmax(axis=1)
    idx_to_label = {0: "H", 1: "D", 2: "A"}
    y_pred = np.vectorize(idx_to_label.get)(y_pred_idx)

    accuracy = (y_pred == eval_df["Result"].values).mean()

    y_onehot = np.eye(3)[y_true]
    brier = float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))

    eps = 1e-12
    ll = np.log(probs[np.arange(len(y_true)), y_true] + eps)
    log_loss = float(-np.mean(ll))

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


# ---------- Backtest ----------

def backtest_expanding(config=None, config_path=None, run_tag=None):

    start_global = time.time()
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting EXPANDING MATCHDAY backtest…")

    # Load config
    if config is None:
        cfg = load_config(config_path or CONFIG_PATH)
    else:
        cfg = config

    results_path = ROOT / cfg["data"]["results_csv"]
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    df = pd.read_csv(results_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    required_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in results: {col}")

    # Filter window
    df_bt = df[df["Date"] >= BACKTEST_START_DATE].copy()

    # Drop fixtures (NaN)
    fixtures = df_bt[df_bt["FTHG"].isna() | df_bt["FTAG"].isna()]
    if not fixtures.empty:
        print(f"⚠ Dropping {len(fixtures)} fixture rows with missing scores.")
    df_bt = df_bt.dropna(subset=["FTHG", "FTAG"])

    # Handle Seasons
    if "Season" not in df_bt.columns:
        df_bt["Season"] = df_bt["Date"].dt.year

    # Matchday logic
    df_bt["Matchday"] = df_bt.groupby("Season").cumcount() // 10
    matchdays = sorted(df_bt["Matchday"].unique())
    print(f"Total matchdays: {len(matchdays)}\n")

    # Resume support
    last_completed_md = None
    if RESUME_PATH.exists():
        try:
            last_completed_md = int(RESUME_PATH.read_text().strip())
            print(f"🔄 Resuming from matchday {last_completed_md}")
        except:
            print("⚠ Resume file unreadable — starting fresh.")

    eval_rows = []

    # Main Loop
    for md in matchdays:

        if last_completed_md is not None and md <= last_completed_md:
            continue

        md_start = time.time()

        train_df = df_bt[df_bt["Matchday"] < md]
        test_df  = df_bt[df_bt["Matchday"] == md]

        if len(train_df) < MIN_TRAIN_MATCHES:
            continue

        # Train models fresh
        try:
            dc = DixonColesModel(
                rho_init=cfg["model"]["dc"].get("rho_init", 0.0),
                home_adv_init=cfg["model"]["dc"].get("home_adv_init", 0.1),
            )
            dc.fit(train_df)

            elo = EloModel(
                k_factor=cfg["model"]["elo"].get("k_factor", 18.0),
                home_advantage=cfg["model"]["elo"].get("home_advantage", 55.0),
            ).fit(train_df)

        except Exception as e:
            print(f"⚠ Training error on matchday {md}: {e}")
            continue

        # Predict
        for _, row in test_df.iterrows():
            home, away = row["HomeTeam"], row["AwayTeam"]
            try:
                dc_out = dc.match_probs(home, away)
                lamH, lamA = dc_out["lambdas"]
                dc_wp = (dc_out["win_probs"]["H"],
                         dc_out["win_probs"]["D"],
                         dc_out["win_probs"]["A"])

                elo_wp = elo.predict(home, away)

                pH, pD, pA = ensemble_win_probs(
                    dc_wp, elo_wp,
                    w_dc=cfg["model"]["ensemble"]["w_dc"],
                    w_elo=cfg["model"]["ensemble"]["w_elo"]
                )

            except Exception:
                continue

            eval_rows.append({
                "Matchday": md,
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
                "TrainMatches": len(train_df)
            })

        elapsed = time.time() - md_start
        print(f"Matchday {md} done in {elapsed:.2f}s")

        # Save resume checkpoint
        RESUME_PATH.write_text(str(md))

    # End Loop

    if not eval_rows:
        print("⚠ No evaluation rows. Check data coverage.")
        return None, None

    eval_df = pd.DataFrame(eval_rows)

    history_dir = ROOT / "models" / "history"
    history_dir.mkdir(exist_ok=True)

    suffix = f"_{run_tag}" if run_tag else ""
    out_path = history_dir / f"backtest_expanding_matchday{suffix}.csv"
    eval_df.to_csv(out_path, index=False)

    print(f"\nSaved expanding matchday backtest → {out_path}")

    metrics = compute_metrics(eval_df)

    print("\n===== EXPANDING BACKTEST SUMMARY =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    if RESUME_PATH.exists():
        RESUME_PATH.unlink()
        print("🧹 Resume checkpoint cleared.")

    total_elapsed = time.time() - start_global
    print(f"\n⏱ Total runtime: {total_elapsed/60:.2f} minutes")

    return metrics, out_path


if __name__ == "__main__":
    backtest_expanding(config_path=CONFIG_PATH)
