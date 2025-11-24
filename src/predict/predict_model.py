import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]  # soccer_agent_local/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from predictor import train_models, predict_fixtures  # noqa: E402
from db import (  # noqa: E402
    init_db,
    get_upcoming_fixtures,
    insert_model_predictions,
)

CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def parse_args():
    days_ahead = 7
    run_id = None

    if len(sys.argv) >= 2:
        try:
            days_ahead = int(sys.argv[1])
        except ValueError:
            print(f"Warning: invalid days_ahead '{sys.argv[1]}', using default 7.")

    if len(sys.argv) >= 3:
        run_id = sys.argv[2]

    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    return days_ahead, run_id


# ============================================================
#  WRAPPER FUNCTION USED BY THE AGENT SYSTEM
# ============================================================
def run_model_predictions(days_ahead: int = 7, run_id: str | None = None) -> pd.DataFrame:
    """
    Run full model prediction pipeline and return predictions DataFrame.

    This is the function used by run_agent.py.
    Predictions are written directly into the DB (predictions table).
    """

    init_db()

    if run_id is None:
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    cfg = load_config()

    data_cfg = cfg.get("data", {})
    results_csv = ROOT / data_cfg.get("results_csv", "data/raw/epl_combined.csv")
    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    # Load historical results for training
    results_df = pd.read_csv(results_csv, parse_dates=["Date"])
    results_df = results_df.dropna(subset=["FTHG", "FTAG"])

    # Get fixtures to predict from DB
    fixtures_df = get_upcoming_fixtures(days_ahead=days_ahead)
    if fixtures_df.empty:
        raise RuntimeError("No fixtures in DB to predict for the requested window.")

    min_fixture_date = fixtures_df["Date"].min()
    train_df = results_df[results_df["Date"] < min_fixture_date].copy()

    if train_df.empty:
        raise RuntimeError("No training matches before earliest fixture.")

    # Load model configs
    dc_cfg = cfg.get("model", {}).get("dc", {})
    elo_cfg = cfg.get("model", {}).get("elo", {})
    ensemble_cfg = cfg.get("model", {}).get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    # Train models
    dc_model, elo_model = train_models(train_df, dc_cfg, elo_cfg)

    # Predict
    preds_df = predict_fixtures(
        fixtures_df,
        dc_model,
        elo_model,
        w_dc=w_dc,
        w_elo=w_elo,
    )

    if preds_df.empty:
        raise RuntimeError("Prediction DataFrame is empty.")

    # Ensure ExpTotalGoals is present
    if "ExpTotalGoals" not in preds_df.columns:
        preds_df["ExpTotalGoals"] = preds_df["ExpHomeGoals"] + preds_df["ExpAwayGoals"]

    # Write predictions into DB
    model_version = "dc_elo_ensemble_v1"
    insert_model_predictions(
        df=preds_df,
        run_id=run_id,
        run_ts=model_version,
    )

    return preds_df


# ============================================================
# CLI ENTRYPOINT
# ============================================================
def main():
    init_db()
    days_ahead, run_id = parse_args()

    print("====================================================")
    print("Starting live model predictions from DB fixtures...")
    print(f"Days ahead window: 0–{days_ahead}")
    print(f"Assigned run_id: {run_id}")
    print("====================================================\n")

    preds_df = run_model_predictions(days_ahead=days_ahead, run_id=run_id)

    print(f"\nPredictions complete. {len(preds_df)} rows written to DB.\n")


if __name__ == "__main__":
    main()
