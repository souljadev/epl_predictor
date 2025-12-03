import sqlite3
from pathlib import Path

DB_PATH = Path("data/soccer_agent.db")

def fix_predictions_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 1. Read the existing table into memory
    cur.execute("SELECT * FROM predictions")
    rows = cur.fetchall()

    cur.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cur.fetchall()]

    print("Existing columns:", columns)

    # 2. Backup old table
    cur.execute("ALTER TABLE predictions RENAME TO predictions_old")

    # 3. Create new table with correct PK
    cur.execute("""
        CREATE TABLE predictions (
            date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            model_version TEXT NOT NULL,

            dixon_coles_probs TEXT,
            elo_probs TEXT,
            ensemble_probs TEXT,

            home_win_prob REAL,
            draw_prob REAL,
            away_win_prob REAL,

            exp_goals_home REAL,
            exp_goals_away REAL,
            exp_total_goals REAL,

            score_pred TEXT,
            chatgpt_pred TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            PRIMARY KEY (date, home_team, away_team, model_version)
        );
    """)

    print("New predictions table created with proper PRIMARY KEY.")

    # 4. Insert all old rows back into new table
    placeholders = ",".join(["?"] * len(columns))

    cur.executemany(
        f"INSERT OR IGNORE INTO predictions ({','.join(columns)}) VALUES ({placeholders})",
        rows
    )

    print(f"Restored {len(rows)} rows.")

    # 5. Drop old backup table
    cur.execute("DROP TABLE predictions_old")

    conn.commit()
    conn.close()

    print("Migration complete.")

if __name__ == "__main__":
    fix_predictions_table()
