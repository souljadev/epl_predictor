from pathlib import Path
import sys
from datetime import datetime

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]  # .../scripts -> project root
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from predictor import train_models, predict_fixtures  # noqa: E402
from db import init_db, insert_model_predictions  # noqa: E402

CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> dict:
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
    return {}


def main():
    print("\n==========================================")
    print("   Historical Backtest (FAST, expanding)  ")
    print("==========================================\n")

    init_db()

    cfg = load_config()
    data_cfg = cfg.get("data", {})
    results_rel = data_cfg.get("results_csv", "data/raw/epl_combined.csv")
    results_csv = ROOT / results_rel

    if not results_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_csv}")

    print(f"Loading historical results from: {results_csv}")
    results_df = pd.read_csv(results_csv, parse_dates=["Date"])
    results_df = results_df.dropna(subset=["FTHG", "FTAG"])
    results_df = results_df.sort_values("Date")

    # Fixtures for backtest = all matches that have results
    fixtures_df = (
        results_df[["Date", "HomeTeam", "AwayTeam"]]
        .drop_duplicates()
        .sort_values("Date")
    )

    print(f"Total matches for fast backtest: {len(fixtures_df)}")

    # Model configs
    dc_cfg = cfg.get("model", {}).get("dc", {})
    elo_cfg = cfg.get("model", {}).get("elo", {})
    ensemble_cfg = cfg.get("model", {}).get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    # Train once on all historical results (expanding type)
    print("Training models on full historical dataset...")
    dc_model, elo_model = train_models(results_df, dc_cfg, elo_cfg)
    print("✓ Models trained.")

    print("Predicting all historical fixtures (expanding backtest)...")
    preds_df = predict_fixtures(
        fixtures_df,
        dc_model,
        elo_model,
        w_dc=w_dc,
        w_elo=w_elo,
    )

    if preds_df.empty:
        raise RuntimeError("Prediction DataFrame is empty after fast backtest.")

    # Ensure ExpTotalGoals is present
    if "ExpTotalGoals" not in preds_df.columns:
        preds_df["ExpTotalGoals"] = preds_df["ExpHomeGoals"] + preds_df["ExpAwayGoals"]

    # Write all predictions into DB under a fixed model_version
    model_version_tag = "hist_fast_v1"
    print(f"Writing predictions to DB with model_version='{model_version_tag}'...")
    insert_model_predictions(
        df=preds_df,
        run_id=None,          # ensures model_version = run_ts
        run_ts=model_version_tag,
    )

    print(f"\n✓ Fast backtest complete. {len(preds_df)} predictions written to DB.\n")
    print("You can now run:  python scripts/evaluation/compare_models.py\n")


if __name__ == "__main__":
    main()
