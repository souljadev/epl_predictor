import sqlite3
from pathlib import Path
from typing import Dict, Optional

# ----------------------------------
# DB PATH (reuse your existing DB)
# ----------------------------------
DB_PATH = Path("data/soccer_agent.db")  # adjust if needed


# ----------------------------------
# CONNECTION
# ----------------------------------
def get_conn():
    return sqlite3.connect(DB_PATH)


# ----------------------------------
# TABLE SETUP
# ----------------------------------
def ensure_table():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS gemini_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date DATE,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            predicted_score TEXT NOT NULL,
            predicted_winner TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.commit()


# ----------------------------------
# INSERT (light dedupe)
# ----------------------------------
def insert_prediction(
    prediction: Dict[str, str],
    match_date: Optional[str] = None
):
    ensure_table()

    with get_conn() as conn:
        # Dedupe: same teams + date + score
        existing = conn.execute("""
            SELECT 1 FROM gemini_predictions
            WHERE home_team = ?
              AND away_team = ?
              AND predicted_score = ?
              AND (match_date IS ? OR match_date = ?)
            LIMIT 1
        """, (
            prediction["home_team"],
            prediction["away_team"],
            prediction["predicted_score"],
            match_date,
            match_date
        )).fetchone()

        if existing:
            return False  # already stored

        conn.execute("""
            INSERT INTO gemini_predictions (
                match_date,
                home_team,
                away_team,
                predicted_score,
                predicted_winner
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            match_date,
            prediction["home_team"],
            prediction["away_team"],
            prediction["predicted_score"],
            prediction["predicted_winner"]
        ))

        conn.commit()
        return True


# ----------------------------------
# CLI TEST
# ----------------------------------
if __name__ == "__main__":
    sample = {
        "home_team": "Arsenal",
        "away_team": "Brentford",
        "predicted_score": "2-0",
        "predicted_winner": "Arsenal"
    }

    inserted = insert_prediction(sample, match_date="2026-01-05")
    print("Inserted:", inserted)
