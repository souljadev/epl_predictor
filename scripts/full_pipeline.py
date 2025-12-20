"""
Full EPL Pipeline — DB-driven

Workflow:
  1) Scrape FBref EPL data
  2) Convert futures → fixtures
  3) Auto-tune DC/Elo structural parameters
  4) Ingest + retrain structural models
  5) Generate model predictions
  6) Generate ChatGPT predictions
  7) Compare model vs ChatGPT
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# ------------------------------------------------------------
# Detect project root and fix sys.path
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

# Add project root so `src` and `scripts` are importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ------------------------------------------------------------
# Imports from src/*
# ------------------------------------------------------------
from src.agent.auto_tuner_structural import run_auto_tuner
from src.agent.ingest_and_retrain import run_structural_tuning
from src.predict.predict_model import run_model_predictions
from src.predict.chatgpt_predictions import run_chatgpt_predictions

# ------------------------------------------------------------
# Import comparison step from scripts/evaluation/*
# ------------------------------------------------------------
from scripts.evaluation.compare_models import run_comparison
from scripts.ingest_results_from_score import run_results_ingest

# Scripts for scrape + futures→fixtures
SCRAPE_SCRIPT = SCRIPTS / "scrape_fbref_epl.py"
CONVERT_SCRIPT = SCRIPTS / "convert_futures_to_fixtures.py"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def header(title: str):
    print("\n" + "=" * 75)
    print(title)
    print("=" * 75 + "\n")


def run_script(path: Path, label: str):
    header(label)

    if not path.exists():
        raise FileNotFoundError(f"Missing script: {path}")

    result = subprocess.run(
        [sys.executable, str(path)],
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"FAILED: {path} (exit code {result.returncode})")


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------
def main():
    start = datetime.now(timezone.utc)

    header("EPL Agent — FULL PIPELINE START")
    print("Started at:", start)

    # 1) Scrape FBref
    try:
        run_script(SCRAPE_SCRIPT, "STEP 1 — Scraping FBref EPL data")
    except Exception as e:
        print(f"⚠ Scrape failed: {e}")

    # 1.5) Ingest recent results (last 7 days)
    header("STEP 1.5 — INGEST RECENT RESULTS (LAST 7 DAYS)")
    try:
        run_results_ingest()
    except Exception as e:
        print(f"⚠ Results ingest failed (best-effort): {e}")


    # 2) Convert futures → fixtures
    try:
        run_script(CONVERT_SCRIPT, "STEP 2 — Converting futures → fixtures")
    except Exception as e:
        print(f"⚠ Convert futures→fixtures failed: {e}")

    # 3) Auto-tune structural params
    header("STEP 3 — AUTO-TUNER (DC/Elo Structural Search)")
    try:
        run_auto_tuner()
    except Exception as e:
        print("⚠ Auto-tuner failed:", e)

    # 4) Ingest new data + retrain
    header("STEP 4 — INGEST + RETRAIN STRUCTURAL MODELS")
    try:
        run_structural_tuning()
    except Exception as e:
        print("⚠ Structural ingestion/retrain failed:", e)

    # 5) Model predictions
    header("STEP 5 — MODEL PREDICTIONS (DC/Elo/Ensemble)")
    try:
        run_model_predictions()
    except Exception as e:
        print("⚠ Model predictions failed:", e)

    # 6) ChatGPT predictions
    header("STEP 6 — CHATGPT SCORE PREDICTIONS")
    try:
        run_chatgpt_predictions()
    except Exception as e:
        print("⚠ ChatGPT predictions failed:", e)

    # 7) Comparison
    header("STEP 7 — MODEL vs CHATGPT COMPARISON")
    try:
        run_comparison()
    except Exception as e:
        print("⚠ Comparison failed:", e)

    end = datetime.now(timezone.utc)
    header("EPL Agent — PIPELINE COMPLETE")
    print("Started at:", start)
    print("Finished at:", end)
    print("Duration:", end - start)


if __name__ == "__main__":
    main()
