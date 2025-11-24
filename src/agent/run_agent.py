"""
EPL Agent — Full Automated Pipeline (DB-driven, no orchestrator.py)

Steps:
  1) Scrape latest FBref EPL data  (scripts/scrape_fbref_epl.py)
  2) Convert futures -> fixtures   (scripts/convert_futures_to_fixtures.py)
  3) Auto-tune structural params   (src/agent/auto_tuner_structural.py)
  4) Ingest + retrain models       (src/agent/ingest_and_retrain.py)
  5) Model predictions from DB     (src/predict/predict_model.py)
  6) ChatGPT predictions           (src/predict/predict_chatgpt.py)
  7) Model vs ChatGPT comparison   (src/evaluation/compare_models.py)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # ./soccer_agent_local
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"

# Ensure we can import src/*
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# ------------------------------------------------------------
# Import pipeline components
# ------------------------------------------------------------
from src.agent.auto_tuner_structural import run_auto_tuner
from src.agent.ingest_and_retrain import run_structural_tuning
from src.predict.predict_model import run_model_predictions
from src.predict.chatgpt_predictions import run_chatgpt_predictions


# Script paths for scrape + convert
SCRAPE_SCRIPT = SCRIPTS / "scrape_fbref_epl.py"
CONVERT_SCRIPT = SCRIPTS / "convert_futures_to_fixtures.py"


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_script(script_path: Path, desc: str | None = None):
    """Run a Python script as a subprocess and stream output."""
    if desc:
        header(desc)
    else:
        header(f"Running script: {script_path}")

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"FAILED: {script_path} (exit code {result.returncode})")


def main():
    header("EPL Agent — FULL PIPELINE START")

    # ------------------------------------------------------------
    # STEP 1 — SCRAPE FBREF
    # ------------------------------------------------------------
    try:
        run_script(SCRAPE_SCRIPT, "STEP 1 — Scraping FBref EPL data...")
    except Exception as e:
        print(f"⚠ Scrape step failed: {e}")
        # You can choose to return here if scrape is mandatory:
        # return

    # ------------------------------------------------------------
    # STEP 2 — CONVERT FUTURES → FIXTURES (DB + fixtures_today.csv)
    # ------------------------------------------------------------
    try:
        run_script(CONVERT_SCRIPT, "STEP 2 — Converting futures → fixtures (DB + fixtures_today.csv)")
    except Exception as e:
        print(f"⚠ Convert futures→fixtures step failed: {e}")
        # Again, you can choose to stop here if no fixtures means no pipeline:
        # return

    # ------------------------------------------------------------
    # STEP 3 — AUTO-TUNER (based on backtest metrics)
    # ------------------------------------------------------------
    header("[3/7] AUTO-TUNER (Structural Model Tuning)")
    try:
        run_auto_tuner()
    except Exception as e:
        print("⚠ Auto-tuner failed:", e)

    # ------------------------------------------------------------
    # STEP 4 — INGEST NEW DATA + RETRAIN
    # ------------------------------------------------------------
    header("[4/7] INGEST + RETRAIN")
    try:
        run_structural_tuning()
    except Exception as e:
        print("⚠ Structural tuning / retrain failed:", e)

    # ------------------------------------------------------------
    # STEP 5 — MODEL PREDICTIONS (DB fixtures → predictions_model)
    # ------------------------------------------------------------
    header("[5/7] MODEL PREDICTIONS (DB fixtures → predictions_model)")
    try:
        run_model_predictions()
    except Exception as e:
        print("⚠ Model prediction failed:", e)

    # ------------------------------------------------------------
    # STEP 6 — CHATGPT PREDICTIONS (DB fixtures → predictions_chatgpt)
    # ------------------------------------------------------------
    header("[6/7] CHATGPT PREDICTIONS (DB fixtures → predictions_chatgpt)")
    try:
        run_chatgpt_predictions()
    except Exception as e:
        print("⚠ ChatGPT prediction failed:", e)

    # ------------------------------------------------------------
    # STEP 7 — MODEL vs CHATGPT COMPARISON (chatgpt_vs_model.csv)
    # ------------------------------------------------------------
    header("[7/7] MODEL vs CHATGPT — COMPARISON REPORT")
    try:
        run_comparison()
    except Exception as e:
        print("⚠ Comparison failed:", e)

    header("EPL Agent — ALL TASKS COMPLETE")
    print("Finished at:", datetime.utcnow(), "UTC")


if __name__ == "__main__":
    main()
