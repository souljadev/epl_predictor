import argparse
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime

from ..training.train import train_models
from .auto_tuner_structural import tune_config

def ingest_and_retrain(config_path: str, new_results_csv: str, predictions_csv: str):
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text())

    results_path = Path(cfg["data"]["results_csv"])
    results = pd.read_csv(results_path, parse_dates=["Date"])

    new_results = pd.read_csv(new_results_csv, parse_dates=["Date"])
    needed_cols = ["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]
    missing = [c for c in needed_cols if c not in new_results.columns]
    if missing:
        raise ValueError(f"new_results_csv is missing columns: {missing}")

    combined = pd.concat([results, new_results], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"]).sort_values("Date")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(results_path, index=False)

    preds = pd.read_csv(predictions_csv, parse_dates=["Date"])
    eval_df = pd.merge(
        new_results,
        preds,
        on=["Date", "HomeTeam", "AwayTeam"],
        how="inner",
        suffixes=("", "_pred"),
    )
    if eval_df.empty:
        raise ValueError("No overlapping rows between new_results and predictions; check Date/Team names.")

    history_dir = Path("models/history")
    history_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_path = history_dir / f"eval_{ts}.csv"
    eval_df.to_csv(eval_path, index=False)
    print(f"Saved evaluation merge to {eval_path}")

    metrics, _ = tune_config(config_path, str(eval_path), None)
    print("Auto-tuning metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    dc, elo = train_models(config_path)
    print("Retraining completed with updated config.")
    return eval_path

def main():
    parser = argparse.ArgumentParser(description="Ingest new EPL results, auto-tune, and retrain models.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--new_results", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    ingest_and_retrain(args.config, args.new_results, args.predictions)

if __name__ == "__main__":
    main()
