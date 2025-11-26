import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "soccer_agent.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("Deleting all rows from fixtures table...")
cur.execute("DELETE FROM fixtures;")

conn.commit()
conn.close()

print("✓ fixtures table cleared.")
