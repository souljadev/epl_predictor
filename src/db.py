import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
import json

import pandas as pd

# ------------------------------------------------------------
# DB PATH
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # .../soccer_agent_local/src
DB_PATH = ROOT / "data" / "soccer_agent.db"


# ------------------------------------------------------------
# CONNECTION HANDLER
# ------------------------------------------------------------
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


# ------------------------------------------------------------
# INITIALIZE DATABASE
# ------------------------------------------------------------
def init_db():
    with get_conn() as conn:
        cur = conn.cursor()

        # Fixtures table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fixtures (
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                PRIMARY KEY (date, home_team, away_team)
            );
        """)

        # Results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS results (
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                FTHG INTEGER,
                FTAG INTEGER,
                Result TEXT,
                PRIMARY KEY (date, home_team, away_team)
            );
        """)

        # Predictions table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
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

        conn.commit()


# ------------------------------------------------------------
# INSERT FIXTURES
# ------------------------------------------------------------
def insert_fixtures(df: pd.DataFrame):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    rows = [
        (row["Date"], row["HomeTeam"], row["AwayTeam"])
        for _, row in df.iterrows()
    ]

    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO fixtures (date, home_team, away_team)
            VALUES (?, ?, ?)
            ON CONFLICT(date, home_team, away_team) DO NOTHING;
            """,
            rows,
        )
        conn.commit()


# ------------------------------------------------------------
# INSERT RESULTS
# ------------------------------------------------------------
def insert_results(df: pd.DataFrame):
    """
    Expects: Date, HomeTeam, AwayTeam, FTHG, FTAG
    """
    df = df.copy()
    df = df.dropna(subset=["FTHG", "FTAG"])

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    rows = []
    for _, row in df.iterrows():
        FTHG = int(row["FTHG"])
        FTAG = int(row["FTAG"])

        if FTHG > FTAG:
            result = "H"
        elif FTAG > FTHG:
            result = "A"
        else:
            result = "D"

        rows.append(
            (row["Date"], row["HomeTeam"], row["AwayTeam"], FTHG, FTAG, result)
        )

    with get_conn() as conn:
        cur = conn.cursor()
        cur.executemany(
            """
            INSERT INTO results (date, home_team, away_team, FTHG, FTAG, Result)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, home_team, away_team) DO UPDATE SET
                FTHG=excluded.FTHG,
                FTAG=excluded.FTAG,
                Result=excluded.Result;
            """,
            rows,
        )
        conn.commit()


# ------------------------------------------------------------
# INSERT A SINGLE PREDICTION
# ------------------------------------------------------------
def insert_predictions(row_dict: dict):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (
                date, home_team, away_team, model_version,
                dixon_coles_probs, elo_probs, ensemble_probs,
                home_win_prob, draw_prob, away_win_prob,
                exp_goals_home, exp_goals_away, exp_total_goals,
                score_pred, chatgpt_pred
            ) VALUES (
                :date, :home_team, :away_team, :model_version,
                :dixon_coles_probs, :elo_probs, :ensemble_probs,
                :home_win_prob, :draw_prob, :away_win_prob,
                :exp_goals_home, :exp_goals_away, :exp_total_goals,
                :score_pred, :chatgpt_pred
            )
            ON CONFLICT(date, home_team, away_team, model_version)
            DO UPDATE SET
                dixon_coles_probs=excluded.dixon_coles_probs,
                elo_probs=excluded.elo_probs,
                ensemble_probs=excluded.ensemble_probs,
                home_win_prob=excluded.home_win_prob,
                draw_prob=excluded.draw_prob,
                away_win_prob=excluded.away_win_prob,
                exp_goals_home=excluded.exp_goals_home,
                exp_goals_away=excluded.exp_goals_away,
                exp_total_goals=excluded.exp_total_goals,
                score_pred=excluded.score_pred,
                chatgpt_pred=excluded.chatgpt_pred;
            """,
            row_dict,
        )
        conn.commit()


# ------------------------------------------------------------
# UPSERT RESULTS FROM epl_combined.csv
# ------------------------------------------------------------
def upsert_results_from_epl_combined(csv_path):
    df = pd.read_csv(csv_path)

    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    insert_fixtures(df)
    insert_results(df)

    return True


# ------------------------------------------------------------
# GET UPCOMING FIXTURES (USED BY predict_model.py)
# ------------------------------------------------------------
def get_upcoming_fixtures(days_ahead: int = 7) -> pd.DataFrame:
    """
    Returns fixtures between today and today + days_ahead as:
        Date, HomeTeam, AwayTeam
    """
    today = datetime.utcnow().date()
    end_date = today + timedelta(days=days_ahead)

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT date AS Date,
                   home_team AS HomeTeam,
                   away_team AS AwayTeam
            FROM fixtures
            WHERE date >= ? AND date <= ?
            ORDER BY date, home_team, away_team
            """,
            conn,
            params=(today.isoformat(), end_date.isoformat()),
        )

    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


# ------------------------------------------------------------
# BULK INSERT MODEL PREDICTIONS (USED BY predict_model.py)
# ------------------------------------------------------------
def insert_model_predictions(df: pd.DataFrame, run_id: str, run_ts: str):
    """
    Takes the preds_df from predict_model.py and stores it in predictions.

    Expected core columns:
      - Date
      - HomeTeam
      - AwayTeam
      - ExpHomeGoals
      - ExpAwayGoals
      - ExpTotalGoals (or will be added in caller)
      - home_win_prob / draw_prob / away_win_prob   OR pH / pD / pA
      - most_likely_score (optional)
    """

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Build a version label that can distinguish runs
    model_version = run_ts if run_id is None else f"{run_ts}_{run_id}"

    for _, row in df.iterrows():
        date_str = row["Date"]
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # Try both naming conventions
        home_prob = row.get("home_win_prob", row.get("pH", None))
        draw_prob = row.get("draw_prob", row.get("pD", None))
        away_prob = row.get("away_win_prob", row.get("pA", None))

        # Prob vectors (if present) could be stored as JSON; for now, None
        dc_probs = None
        elo_probs = None
        ensemble_probs = None

        row_dict = {
            "date": date_str,
            "home_team": home,
            "away_team": away,
            "model_version": model_version,
            "dixon_coles_probs": dc_probs,
            "elo_probs": elo_probs,
            "ensemble_probs": ensemble_probs,
            "home_win_prob": home_prob,
            "draw_prob": draw_prob,
            "away_win_prob": away_prob,
            "exp_goals_home": row.get("ExpHomeGoals"),
            "exp_goals_away": row.get("ExpAwayGoals"),
            "exp_total_goals": row.get("ExpTotalGoals"),
            "score_pred": row.get("most_likely_score"),
            "chatgpt_pred": None,
        }

        insert_predictions(row_dict)

# ------------------------------------------------------------
# GET TRAINING MATCHES (fixtures + results joined)
# ------------------------------------------------------------
def get_training_matches() -> pd.DataFrame:
    """
    Returns historical match results joined with fixture metadata.

    Output columns:
        Date, HomeTeam, AwayTeam, FTHG, FTAG, Result
    """

    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT 
                f.date AS Date,
                f.home_team AS HomeTeam,
                f.away_team AS AwayTeam,
                r.FTHG,
                r.FTAG,
                r.Result
            FROM fixtures f
            JOIN results r
              ON f.date = r.date
             AND f.home_team = r.home_team
             AND f.away_team = r.away_team
            ORDER BY f.date
            """,
            conn,
        )

    # Convert Date to datetime
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df

# ------------------------------------------------------------
# SAVE / LOAD MODEL PARAMETERS (DC + ELO)
# ------------------------------------------------------------
def save_dc_params(params: dict):
    """
    Saves Dixon–Coles parameters to models/dc_params.json
    """
    path = ROOT / "models" / "dc_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(params, f)

    return True


def save_elo_params(params: dict):
    """
    Saves Elo parameters to models/elo_params.json
    """
    path = ROOT / "models" / "elo_params.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(params, f)

    return True


def load_dc_params():
    """
    Loads DC params from stored JSON.
    """
    path = ROOT / "models" / "dc_params.json"
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_elo_params():
    """
    Loads Elo params from stored JSON.
    """
    path = ROOT / "models" / "elo_params.json"
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)
