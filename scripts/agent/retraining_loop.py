"""
Agentic Retraining Loop

This script ties the full EPL pipeline together:

1. Runs the FAST rolling backtest.
2. Runs evaluation (expanding vs rolling_fast).
3. Reads metrics from metrics_backtests_summary.csv.
4. Decides whether to retrain based on thresholds.
5. If triggered, retrains production DC + Elo models on all data.
6. Saves model artifacts and a retraining log entry.

Intended usage:
    python scripts/agent/retraining_loop.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Adjust these imports to match your actual model module paths
from models.dixon_coles import DixonColesModel
from models.elo import EloModel

CONFIG_PATH = ROOT / "config.yaml"

HISTORY_DIR = ROOT / "models" / "history"
EVAL_DIR = ROOT / "models" / "evaluation"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
LOG_DIR = ROOT / "models" / "logs"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV_DEFAULT = ROOT / "data" / "raw" / "epl_results_sample.csv"
METRICS_CSV = EVAL_DIR / "metrics_backtests_summary.csv"

BACKTEST_EXPANDING_SCRIPT = ROOT / "scripts" / "backtest" / "backtest_expanding.py"
BACKTEST_ROLLING_FAST_SCRIPT = ROOT / "scripts" / "backtest" / "backtest_rolling_fast.py"
EVALUATE_SCRIPT = ROOT / "scripts" / "evaluation" / "evaluate_backtest.py"


def load_config():
    import yaml
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def run_subprocess(cmd, desc: str):
    """Run a subprocess command and print status."""
    print(f"\n===== Running: {desc} =====")
    print(" ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠ {desc} FAILED with return code {result.returncode}")
        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
    else:
        print(f"{desc} completed successfully.")
        if result.stdout.strip():
            print("STDOUT:\n", result.stdout)
    return result.returncode


def run_backtests_and_evaluation():
    """Run rolling FAST backtest + expanding (optional) + evaluation."""
    # Optional: you can comment out expanding if you only care about rolling_fast
    run_subprocess([sys.executable, str(BACKTEST_EXPANDING_SCRIPT)], "Expanding backtest")
    run_subprocess([sys.executable, str(BACKTEST_ROLLING_FAST_SCRIPT)], "FAST rolling backtest")
    run_subprocess([sys.executable, str(EVALUATE_SCRIPT)], "Evaluate backtests")


def load_latest_metrics():
    """Load metrics_backtests_summary.csv and return as DataFrame."""
    if not METRICS_CSV.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_CSV}")
    df = pd.read_csv(METRICS_CSV)
    return df


def select_key_row(df: pd.DataFrame, label: str) -> pd.Series | None:
    """Return the row for a given label or None if missing."""
    matches = df[df["label"] == label]
    if matches.empty:
        return None
    return matches.iloc[0]


def should_retrain(metrics_row: pd.Series, cfg: dict) -> bool:
    """
    Decide whether to retrain based on thresholds in config or defaults.

    Config (config.yaml) example:

    retraining:
      target_label: "rolling_fast_last2"
      min_accuracy: 0.52
      max_log_loss: 1.05
      max_brier: 0.60

    """
    retrain_cfg = cfg.get("retraining", {})
    min_acc = retrain_cfg.get("min_accuracy", 0.52)
    max_ll = retrain_cfg.get("max_log_loss", 1.05)
    max_brier = retrain_cfg.get("max_brier", 0.60)

    acc = metrics_row.get("accuracy", float("nan"))
    ll = metrics_row.get("log_loss", float("nan"))
    br = metrics_row.get("brier", float("nan"))

    print("\n===== RETRAIN DECISION METRICS =====")
    print(f"Accuracy: {acc:.4f} (min target {min_acc:.4f})")
    print(f"Log loss: {ll:.4f} (max target {max_ll:.4f})")
    print(f"Brier:    {br:.4f} (max target {max_brier:.4f})")

    # If any metric is NaN, be conservative and retrain
    if any(pd.isna(v) for v in [acc, ll, br]):
        print("⚠ Some metrics are NaN — defaulting to retrain=True")
        return True

    # Simple rule: retrain if accuracy < min or losses exceed max threshold
    if acc < min_acc:
        print("→ Retrain: accuracy below threshold")
        return True
    if ll > max_ll:
        print("→ Retrain: log loss above threshold")
        return True
    if br > max_brier:
        print("→ Retrain: Brier score above threshold")
        return True

    print("→ No retrain needed based on current thresholds.")
    return False


def train_production_models(cfg: dict):
    """
    Train final production DC + Elo models on all available data.
    Saves artifacts to models/artifacts/.
    """
    retrain_cfg = cfg.get("retraining", {})
    results_csv = ROOT / cfg.get("data", {}).get("results_csv", RESULTS_CSV_DEFAULT)
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    df = pd.read_csv(results_csv, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["FTHG", "FTAG"])

    # Basic outcome label if missing
    if "Result" not in df.columns:
        def outcome_label(hg, ag):
            if pd.isna(hg) or pd.isna(ag):
                return None
            hg, ag = int(hg), int(ag)
            if hg > ag:
                return "H"
            if hg < ag:
                return "A"
            return "D"
        df["Result"] = df.apply(lambda r: outcome_label(r["FTHG"], r["FTAG"]), axis=1)

    dc_cfg = cfg.get("model", {}).get("dc", {})
    elo_cfg = cfg.get("model", {}).get("elo", {})

    print("\n===== TRAINING PRODUCTION MODELS =====")
    print(f"Training on {len(df)} matches from {df['Date'].min().date()} to {df['Date'].max().date()}")

    # Dixon–Coles
    dc = DixonColesModel(
        rho_init=dc_cfg.get("rho_init", 0.0),
        home_adv_init=dc_cfg.get("home_adv_init", 0.15),
        lr=dc_cfg.get("lr", 0.05),
    )
    use_xg = "Home_xG" in df.columns and "Away_xG" in df.columns
    dc.fit(df, use_xg=use_xg)

    # Elo
    elo = EloModel(
        k_factor=elo_cfg.get("k_factor", 18.0),
        home_advantage=elo_cfg.get("home_advantage", 55.0),
        base_rating=elo_cfg.get("base_rating", 1500.0),
        draw_base=elo_cfg.get("draw_base", 0.25),
        draw_max_extra=elo_cfg.get("draw_max_extra", 0.10),
        draw_scale=elo_cfg.get("draw_scale", 400.0),
    )
    elo.fit(df)

    # Save artifacts
    import pickle
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dc_path = ARTIFACT_DIR / f"dc_model_{ts}.pkl"
    elo_path = ARTIFACT_DIR / f"elo_model_{ts}.pkl"

    with open(dc_path, "wb") as f:
        pickle.dump(dc, f)
    with open(elo_path, "wb") as f:
        pickle.dump(elo, f)

    print(f"Saved DC model → {dc_path}")
    print(f"Saved Elo model → {elo_path}")

    # Log retraining event
    retrain_log_path = LOG_DIR / "retraining_log.csv"
    entry = {
        "timestamp_utc": ts,
        "n_matches": len(df),
        "date_min": df["Date"].min(),
        "date_max": df["Date"].max(),
        "dc_model_path": dc_path.name,
        "elo_model_path": elo_path.name,
    }
    if retrain_log_path.exists():
        log_df = pd.read_csv(retrain_log_path)
        log_df = pd.concat([log_df, pd.DataFrame([entry])], ignore_index=True)
    else:
        log_df = pd.DataFrame([entry])
    log_df.to_csv(retrain_log_path, index=False)
    print(f"Updated retraining log → {retrain_log_path}")


def main():
    print("\n================ AGENTIC RETRAINING LOOP ================")

    cfg = load_config()

    # 1) Run backtests + evaluation to get fresh metrics
    run_backtests_and_evaluation()

    # 2) Load metrics and pull the key row for decision
    df_metrics = load_latest_metrics()

    # Choose label to make decision on (default: rolling_fast_last2)
    target_label = cfg.get("retraining", {}).get("target_label", "rolling_fast_last2")
    row = select_key_row(df_metrics, target_label)
    if row is None:
        print(f"⚠ No metrics row found for label='{target_label}'. Cannot decide. Exiting.")
        return

    # 3) Decide whether to retrain
    if not should_retrain(row, cfg):
        print("\nNo retraining performed.")
        return

    # 4) Train production models on all data
    train_production_models(cfg)

    print("\n================ END AGENTIC LOOP RUN ================")


if __name__ == "__main__":
    main()
