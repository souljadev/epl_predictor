import sqlite3
from pathlib import Path
from datetime import date

from gemini_predict import get_gemini_prediction
from gemini_store import insert_prediction

# ----------------------------------
# DB PATH
# ----------------------------------
DB_PATH = Path("data/soccer_agent.db")


# ----------------------------------
# FETCH TODAY'S FIXTURES
# ----------------------------------
def get_todays_fixtures(target_date: date):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT
                date,
                home_team,
                away_team
            FROM fixtures
            WHERE date = ?
        """, (target_date.isoformat(),)).fetchall()

    return rows


# ----------------------------------
# RUNNER
# ----------------------------------
def run_gemini_for_date(target_date: date):
    fixtures = get_todays_fixtures(target_date)

    if not fixtures:
        print(f"No fixtures found for {target_date}")
        return

    print(f"Running Gemini for {len(fixtures)} matches on {target_date}")

    for match_date, home, away in fixtures:
        try:
            prediction = get_gemini_prediction(home, away)

            inserted = insert_prediction(
                prediction,
                match_date=match_date
            )

            status = "INSERTED" if inserted else "SKIPPED"
            print(f"[{status}] {home} vs {away} → {prediction['predicted_score']}")

        except Exception as e:
            print(f"[ERROR] {home} vs {away}: {e}")


# ----------------------------------
# CLI ENTRYPOINT
# ----------------------------------
from datetime import timedelta

def run_gemini_next_days(days_ahead: int = 1):
    for i in range(1, days_ahead + 1):
        target_date = date.today() + timedelta(days=i)
        run_gemini_for_date(target_date)


if __name__ == "__main__":
    run_gemini_next_days(days_ahead=3)

