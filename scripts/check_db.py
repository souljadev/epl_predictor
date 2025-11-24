from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "soccer_agent.db"

print("DB Path:", DB_PATH, "\n")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("TABLES:")
for (name,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table';"):
    print(" -", name)

print("\nSCHEMA:")

for (name,) in cur.execute("SELECT name FROM sqlite_master WHERE type='table';"):
    print(f"\n=== {name} ===")
    for col in cur.execute(f"PRAGMA table_info({name});"):
        print(col)

conn.close()
