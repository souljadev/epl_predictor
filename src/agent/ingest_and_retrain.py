"""
Agent ingestion + retraining module.

This version is simplified to:
- Use the main config.yaml
- Retrain models on the full epl_combined.csv (or whatever results_csv points to)
- Be callable from src/agent/run_agent.py as run_structural_tuning()
"""

from pathlib import Path
from datetime import datetime
import yaml

from src.training.train import train_models


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yaml"


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_structural_tuning(config_path: str | None = None):
    """
    Retrain the core models using the current config.yaml.

    Flow:
      1. Load config (to know which results CSV to use)
      2. Call train_models(config)
      3. Print basic info and return a small summary dict
    """
    header("STRUCTURAL TUNING — RETRAIN MODELS")

    cfg_file = Path(config_path) if config_path else CONFIG_PATH
    if not cfg_file.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_file}")

    cfg = yaml.safe_load(cfg_file.read_text())
    results_csv = cfg["data"]["results_csv"]

    print(f"Using config: {cfg_file}")
    print(f"Training data: {results_csv}")
    print(f"Started at: {datetime.utcnow()} UTC\n")

    # This builds features, splits, and fits DC + Elo
    dc_model, elo_model = train_models(str(cfg_file))

    print("\nRetraining completed.")
    print("Models trained:")
    print(f"  Dixon-Coles: {type(dc_model).__name__}")
    print(f"  Elo model:   {type(elo_model).__name__}")

    return {
        "config": str(cfg_file),
        "results_csv": results_csv,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }


# Optional CLI entrypoint if you want to run this module directly
if __name__ == "__main__":
    info = run_structural_tuning()
    print("\nRun info:", info)
