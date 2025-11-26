import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "soccer_agent.db"

def main():
    print(f"DB Path: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # 1) How many results?
    df_results_count = pd.read_sql_query(
        "SELECT COUNT(*) AS results_count FROM results;",
        conn,
    )
    print("\n=== RESULTS COUNT ===")
    print(df_results_count)

    # 2) Sample results
    df_results_sample = pd.read_sql_query(
        "SELECT * FROM results ORDER BY date DESC LIMIT 10;",
        conn,
    )
    print("\n=== SAMPLE RESULTS (LATEST 10) ===")
    print(df_results_sample)

    # 3) How many fixtures?
    df_fixtures_count = pd.read_sql_query(
        "SELECT COUNT(*) AS fixtures_count FROM fixtures;",
        conn,
    )
    print("\n=== FIXTURES COUNT ===")
    print(df_fixtures_count)

    # 4) Sample fixtures
    df_fixtures_sample = pd.read_sql_query(
        "SELECT * FROM fixtures ORDER BY date LIMIT 20;",
        conn,
    )
    print("\n=== SAMPLE FIXTURES (NEXT 20) ===")
    print(df_fixtures_sample)

    # 5) Predictions count + sample
    df_preds_count = pd.read_sql_query(
        "SELECT COUNT(*) AS preds_count FROM predictions;",
        conn,
    )
    print("\n=== PREDICTIONS COUNT ===")
    print(df_preds_count)

    df_preds_sample = pd.read_sql_query(
        "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;",
        conn,
    )
    print("\n=== SAMPLE PREDICTIONS (LATEST 10) ===")
    print(df_preds_sample)


    conn.close()


if __name__ == "__main__":
    main()
