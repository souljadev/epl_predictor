"""
Convert FUTURES → FIXTURES (DB-first, Option C)

- Loads futures_today.csv created by scrape_fbref_epl.py
- Converts to the format expected by insert_fixtures()
- Inserts fixtures into the SQLite database
- Exports a debug copy to data/debug/fixtures_today.csv
"""

import sys
from pathlib import Path
import pandas as pd

# Fix path so src imports work when run as a subprocess
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.db import init_db, insert_fixtures   # ✔ existing functions


def header(msg: str):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70 + "\n")


def main():
    header("Convert FUTURES → FIXTURES (DB-only)")

    # Ensure DB initialized
    init_db()

    futures_csv = ROOT / "data" / "futures" / "futures_today.csv"
    if not futures_csv.exists():
        print(f"⚠ No futures_today.csv found at: {futures_csv}")
        return

    # Load the futures CSV from scraper
    df = pd.read_csv(futures_csv)

    # Normalize column names to match DB loader
    rename_map = {
        "Date": "Date",
        "Home": "HomeTeam",
        "Away": "AwayTeam",
        "HomeTeam": "HomeTeam",
        "AwayTeam": "AwayTeam"
    }
    df = df.rename(columns=rename_map)

    required = {"Date", "HomeTeam", "AwayTeam"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in futures_today.csv: {missing}")

    # Insert fixtures into DB
    insert_fixtures(df)
    print(f"✔ Inserted / upserted {len(df)} fixtures into DB")

    # Write a debugging CSV copy
    debug_path = ROOT / "data" / "debug" / "fixtures_today.csv"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(debug_path, index=False)
    print(f"✔ Debug CSV exported to: {debug_path}")


if __name__ == "__main__":
    main()
