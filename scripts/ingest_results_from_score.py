"""
Ingest match results from fixtures_fbref.csv into the `results` table
using the `score` column.

Behavior:
- Best-effort (never crashes pipeline)
- Looks back N days (default: 7)
- Parses score like '2–1' or '2-1'
- Inserts only missing rows
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re

DB_PATH = Path("data/soccer_agent.db")
CSV_PATH = Path("data/fixtures_fbref.csv")

LOOKBACK_DAYS = 7
SCORE_RE = re.compile(r"^\s*(\d+)\s*[–-]\s*(\d+)\s*$")


def run_results_ingest():
    try:
        cutoff = (datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)).date().isoformat()

        df = pd.read_csv(CSV_PATH)

        required = {"match_date", "home_team", "away_team", "score"}
        if not required.issubset(df.columns):
            print("⚠ Results ingest skipped — CSV missing required columns")
            return

        # Only consider recent matches
        df = df[df["match_date"] >= cutoff]

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        inserted = 0

        for _, r in df.iterrows():
            score = str(r["score"]).strip()
            m = SCORE_RE.match(score)
            if not m:
                continue

            fthg, ftag = map(int, m.groups())
            result = "H" if fthg > ftag else "A" if fthg < ftag else "D"

            try:
                cur.execute(
                    """
                    INSERT INTO results
                    (date, home_team, away_team, FTHG, FTAG, Result)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["match_date"],
                        r["home_team"],
                        r["away_team"],
                        fthg,
                        ftag,
                        result,
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        conn.close()

        print(f"✅ Results ingest complete — {inserted} new results inserted")

    except Exception as e:
        print(f"⚠ Results ingest failed (best-effort): {e}")
