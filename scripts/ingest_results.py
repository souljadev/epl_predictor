from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from db import insert_fixtures, insert_results, init_db  # noqa: E402


def main():
    print("\n===============================")
    print("   Ingesting EPL Results CSV   ")
    print("===============================\n")

    init_db()

    csv_path = ROOT / "data" / "raw" / "epl_combined.csv"
    print("CSV Path:", csv_path)

    if not csv_path.exists():
        print("✗ ERROR: epl_combined.csv not found.")
        return

    print("\nLoading CSV into memory...")
    df = pd.read_csv(csv_path)
    print(f"Rows loaded: {len(df)}")

    # Clean: drop rows with missing scores (NaN)
    before = len(df)
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()
    after = len(df)

    print(f"Dropped {before - after} unfinished matches with NaN FTHG/FTAG.")
    print(f"Remaining complete matches: {after}")

    # Normalize date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Insert fixtures (dedup automatically)
    print("\nInserting fixtures...")
    insert_fixtures(df)
    print("✓ Fixtures inserted into DB.")

    # Insert results (upsert automatically)
    print("Inserting results...")
    insert_results(df)
    print("✓ Results inserted into DB.")

    print("\nDone. Fixtures + Results loaded successfully.\n")


if __name__ == "__main__":
    main()
