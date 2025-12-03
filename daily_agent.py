"""
daily_agent.py

Entry point for the daily EPL agent.
Runs the full DB-driven pipeline once per day.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Project root (this file lives in the root)
ROOT = Path(__file__).resolve().parent
FULL_PIPELINE = ROOT / "scripts" / "full_pipeline.py"


def main():
    start = datetime.now(timezone.utc)
    print("=" * 75)
    print(f" DAILY EPL AGENT START — {start} ")
    print("=" * 75)

    if not FULL_PIPELINE.exists():
        raise FileNotFoundError(f"❌ full_pipeline.py not found at: {FULL_PIPELINE}")

    # Run the full pipeline as a subprocess
    result = subprocess.run(
        [sys.executable, str(FULL_PIPELINE)],
        text=True
    )

    if result.returncode != 0:
        print(f"⚠ full_pipeline.py FAILED (exit code {result.returncode})")
    else:
        print("✅ full_pipeline.py completed successfully.")

    end = datetime.now(timezone.utc)
    print("=" * 75)
    print(f" DAILY EPL AGENT END   — {end} ")
    print(f" Duration: {end - start}")
    print("=" * 75)


if __name__ == "__main__":
    main()
