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
    print("   Historical Backtest (ROLLING, matchday)")
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

    # Unique matchdays (by calendar date)
    matchdays = results_df["Date"].drop_duplicates().sort_values().tolist()
    print(f"Total distinct matchdays: {len(matchdays)}")

    dc_cfg = cfg.get("model", {}).get("dc", {})
    elo_cfg = cfg.get("model", {}).get("elo", {})
    ensemble_cfg = cfg.get("model", {}).get("ensemble", {})
    w_dc = ensemble_cfg.get("w_dc", 0.6)
    w_elo = ensemble_cfg.get("w_elo", 0.4)

    all_preds = []

    for i, md in enumerate(matchdays, start=1):
        # Training data: all matches strictly before this matchday
        train_df = results_df[results_df["Date"] < md].copy()
        test_df = results_df[results_df["Date"] == md][["Date", "HomeTeam", "AwayTeam"]].copy()

        if train_df.empty:
            # Not enough history yet (early seasons) — skip this matchday
            continue

        print(f"[{i}/{len(matchdays)}] Matchday: {md.date()} | "
              f"Train size: {len(train_df)} | Test matches: {len(test_df)}")

        # Train models on data up to this matchday
        dc_model, elo_model = train_models(train_df, dc_cfg, elo_cfg)

        # Predict this matchday's fixtures
        day_preds = predict_fixtures(
            test_df,
            dc_model,
            elo_model,
            w_dc=w_dc,
            w_elo=w_elo,
        )

        if day_preds.empty:
            continue

        # Ensure ExpTotalGoals column
        if "ExpTotalGoals" not in day_preds.columns:
            day_preds["ExpTotalGoals"] = (
                day_preds["ExpHomeGoals"] + day_preds["ExpAwayGoals"]
            )

        all_preds.append(day_preds)

    if not all_preds:
        raise RuntimeError("No rolling predictions produced — check data coverage.")

    preds_df = pd.concat(all_preds, ignore_index=True)
    print(f"\nTotal rolling predictions generated: {len(preds_df)}")

    # Write rolling predictions into DB with a distinct model_version
    model_version_tag = "hist_roll_v1"
    print(f"Writing rolling backtest predictions to DB with model_version='{model_version_tag}'...")
    insert_model_predictions(
        df=preds_df,
        run_id=None,
        run_ts=model_version_tag,
    )

    print("\n✓ Rolling backtest complete.")
    print("You can now run:  python scripts/evaluation/compare_models.py\n")


if __name__ == "__main__":
    main()
