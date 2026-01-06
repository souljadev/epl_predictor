from .gemini_predict import get_gemini_prediction
from .gemini_store import insert_prediction
from src.db import get_conn


def run_gemini_predictions(days_ahead: int = 3):
    """
    Fetch upcoming fixtures, call Gemini, store results.
    """

    # get_conn is a context manager
    with get_conn() as conn:
        fixtures = conn.execute(
            """
            SELECT date, home_team, away_team
            FROM fixtures
            WHERE date >= DATE('now')
              AND date <= DATE('now', ?)
            ORDER BY date, home_team, away_team
            """,
            (f"+{days_ahead} days",),
        ).fetchall()

    if not fixtures:
        print("⚠ Gemini: no fixtures found")
        return

    for match_date, home, away in fixtures:
        try:
            # 🔥 ACTUAL GEMINI API CALL
            prediction = get_gemini_prediction(home, away)

            inserted = insert_prediction(
                prediction,
                match_date=match_date
            )

            status = "INSERTED" if inserted else "SKIPPED"
            print(
                f"✅ Gemini [{status}]: {home} vs {away} → "
                f"{prediction['predicted_score']}"
            )

        except Exception as e:
            print(f"❌ Gemini failed for {home} vs {away}: {e}")
