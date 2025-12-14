import sqlite3
from pathlib import Path

# ------------------------------------------------------------
# CONFIG — adjust path if needed
# ------------------------------------------------------------
DB_PATH = Path("data/soccer_agent.db")  # or soccer.db if that's the name

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH.resolve()}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("\n==============================")
    print("DATABASE PATH")
    print("==============================")
    print(DB_PATH.resolve())

    # ------------------------------------------------------------
    # 1. List tables
    # ------------------------------------------------------------
    print("\n==============================")
    print("TABLES")
    print("==============================")
    cur.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        ORDER BY name;
    """)
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        print(f"- {t}")

    # ------------------------------------------------------------
    # 2. Inspect results table (if it exists)
    # ------------------------------------------------------------
    if "results" in tables:
        print("\n==============================")
        print("RESULTS TABLE SCHEMA")
        print("==============================")
        cur.execute("PRAGMA table_info(results);")
        for row in cur.fetchall():
            cid, name, col_type, notnull, dflt, pk = row
            print(f"{cid:2d} | {name:15s} | {col_type:10s} | PK={pk}")

        print("\n==============================")
        print("RESULTS TABLE COUNTS")
        print("==============================")
        cur.execute("SELECT COUNT(*) FROM results;")
        print("Row count:", cur.fetchone()[0])

        cur.execute("SELECT MIN(date), MAX(date) FROM results;")
        print("Date range:", cur.fetchone())

    else:
        print("\n⚠️  No `results` table found.")

    conn.close()


if __name__ == "__main__":
    main()
