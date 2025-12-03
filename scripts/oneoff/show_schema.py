import sqlite3
conn = sqlite3.connect("data/soccer_agent.db")
cur = conn.cursor()
rows = cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='predictions'").fetchall()
print(rows[0][0] if rows else "No predictions table found")
