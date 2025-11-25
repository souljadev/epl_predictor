from datetime import date, timedelta
import pandas as pd

from utils.db import get_conn


def load_upcoming_predictions(days_ahead: int = 7) -> pd.DataFrame:
    """
    Load upcoming fixtures (today..today+days_ahead) with latest model
    predictions + optional ChatGPT score from DB.
    """
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)

    with get_conn() as conn:
        query = """
        WITH latest_predictions AS (
            SELECT *
            FROM predictions p
            WHERE created_at = (
                SELECT MAX(created_at)
                FROM predictions p2
                WHERE p2.date = p.date
                  AND p2.home_team = p.home_team
                  AND p2.away_team = p.away_team
            )
        )
        SELECT
            p.date,
            p.home_team,
            p.away_team,
            p.home_win_prob,
            p.draw_prob,
            p.away_win_prob,
            p.exp_goals_home,
            p.exp_goals_away,
            p.exp_total_goals,
            p.score_pred,
            p.chatgpt_pred
        FROM latest_predictions p
        JOIN fixtures f
          ON f.date = p.date
         AND f.home_team = p.home_team
         AND f.away_team = p.away_team
        WHERE f.date >= ? AND f.date <= ?
        ORDER BY f.date, f.home_team, f.away_team;
        """
        df = pd.read_sql_query(
            query,
            conn,
            params=(today.isoformat(), cutoff.isoformat()),
        )

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def load_comparison_from_db() -> pd.DataFrame:
    """
    Build full model vs ChatGPT vs actual comparison
    LIVE from the SQLite DB.
    """
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            WITH latest_predictions AS (
                SELECT *
                FROM predictions p
                WHERE created_at = (
                    SELECT MAX(created_at)
                    FROM predictions p2
                    WHERE p2.date = p.date
                      AND p2.home_team = p.home_team
                      AND p2.away_team = p.away_team
                )
            )
            SELECT
                p.date,
                p.home_team,
                p.away_team,
                p.model_version,
                p.home_win_prob,
                p.draw_prob,
                p.away_win_prob,
                p.exp_goals_home,
                p.exp_goals_away,
                p.exp_total_goals,
                p.score_pred,
                p.chatgpt_pred,
                r.FTHG,
                r.FTAG,
                r.Result AS actual_result
            FROM latest_predictions p
            LEFT JOIN results r
              ON p.date = r.date
             AND p.home_team = r.home_team
             AND p.away_team = r.away_team
            ORDER BY p.date DESC, p.home_team;
            """,
            conn,
        )

    if df.empty:
        return df

    # Normalize Date
    df["Date"] = pd.to_datetime(df["date"], errors="coerce")

    # ------------------------------
    # Winner helpers
    # ------------------------------
    def resolve_winner(h, a):
        if pd.isna(h) or pd.isna(a):
            return None
        if h > a:
            return "H"
        if a > h:
            return "A"
        return "D"

    df["actual_winner"] = df.apply(lambda r: resolve_winner(r["FTHG"], r["FTAG"]), axis=1)

    def model_winner(row):
        if pd.isna(row["home_win_prob"]):
            return None
        probs = [row["home_win_prob"], row["draw_prob"], row["away_win_prob"]]
        idx = probs.index(max(probs))
        return ["H", "D", "A"][idx]

    df["model_winner_pred"] = df.apply(model_winner, axis=1)

    def chatgpt_winner(score):
        if not isinstance(score, str) or "-" not in score:
            return None
        try:
            h, a = score.split("-")
            h, a = int(h), int(a)
        except Exception:
            return None
        if h > a:
            return "H"
        if a > h:
            return "A"
        return "D"

    df["chatgpt_winner_pred"] = df["chatgpt_pred"].apply(chatgpt_winner)

    # ------------------------------
    # Correctness columns
    # ------------------------------
    df["actual_score"] = df.apply(
        lambda r: f"{int(r['FTHG'])}-{int(r['FTAG'])}" if pd.notna(r["FTHG"]) else None,
        axis=1,
    )

    df["correct_winner_model"] = (
        df["model_winner_pred"] == df["actual_winner"]
    ).astype(int)

    df["correct_winner_chatgpt"] = (
        df["chatgpt_winner_pred"] == df["actual_winner"]
    ).astype(int)

    df["correct_score_model"] = (
        df["score_pred"] == df["actual_score"]
    ).astype(int)

    df["correct_score_chatgpt"] = (
        df["chatgpt_pred"] == df["actual_score"]
    ).astype(int)

    # ------------------------------
    # Prediction miss (abs diff total goals vs xG total)
    # ------------------------------
    def pred_miss(row):
        if pd.isna(row["FTHG"]) or pd.isna(row["FTAG"]) or pd.isna(row["exp_total_goals"]):
            return None
        actual_total = row["FTHG"] + row["FTAG"]
        return abs(actual_total - row["exp_total_goals"])

    df["model_xg_error"] = df.apply(pred_miss, axis=1)

    return df
