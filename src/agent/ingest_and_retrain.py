"""
ingest_and_retrain.py

DB-first structural model training (Option C)
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import (
    init_db,
    get_training_matches,   # ✔ THIS NOW EXISTS
    save_dc_params,
    save_elo_params,
)

from src.models.train_dc import train_dc_model
from src.models.train_elo import train_elo_model


def header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def run_structural_tuning():
    start = datetime.now(timezone.utc)

    header("STRUCTURAL TUNING — RETRAIN MODELS")

    print(f"Using config: {ROOT / 'config.yaml'}")
    print(f"Started at: {start} UTC\n")

    # ----------------------------------------------------------------------
    # STEP 1 — Load full match history
    # ----------------------------------------------------------------------
    print("Loading training data from DB...")
    df = get_training_matches()

    if df.empty:
        raise ValueError("❌ No match history in DB — cannot retrain models.")

    print(f"✔ Loaded {len(df)} matches")

    # ----------------------------------------------------------------------
    # STEP 2 — Save debug copy
    # ----------------------------------------------------------------------
    debug_path = ROOT / "data" / "debug" / "training_export.csv"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(debug_path, index=False)
    print(f"✔ Training export saved: {debug_path}")

    # ----------------------------------------------------------------------
    # STEP 3 — Train DC
    # ----------------------------------------------------------------------
    print("\nTraining Dixon–Coles model...")
    try:
        dc_params = train_dc_model(df)
        save_dc_params(dc_params)
        print("✔ Dixon–Coles model saved.")
    except Exception as e:
        print("⚠ DC training failed:", e)

    # ----------------------------------------------------------------------
    # STEP 4 — Train ELO
    # ----------------------------------------------------------------------
    print("\nTraining Elo model...")
    try:
        elo_params = train_elo_model(df)
        save_elo_params(elo_params)
        print("✔ Elo model saved.")
    except Exception as e:
        print("⚠ Elo training failed:", e)

    end = datetime.now(timezone.utc)

    print("\n" + "=" * 70)
    print("STRUCTURAL TRAINING COMPLETE")
    print("Started:", start)
    print("Finished:", end)
    print("Duration:", end - start)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_structural_tuning()
