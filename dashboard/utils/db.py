from pathlib import Path
import sqlite3
from contextlib import contextmanager

# Project root: .../soccer_agent_local
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "soccer_agent.db"


@contextmanager
def get_conn():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
