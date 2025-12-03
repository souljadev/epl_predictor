import sys
import json
import sqlite3
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import yaml
from openai import OpenAI

# ============================================================
# FIX IMPORT PATHS (CRITICAL)
# ============================================================
ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local/
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from db import init_db, insert_predictions  # noqa: E402

DB_PATH = ROOT / "data" / "soccer_agent.db"

# Choose ChatGPT model
CHATGPT_MODEL = "gpt-4o-mini"


# ============================================================
# CONFIG LOADER
# ============================================================
def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ============================================================
# DB HELPERS
# ============================================================
def get_conn():
    return sqlite3.connect(DB_PATH)


def get_upcoming_model_predictions(days_ahead: int = 7) -> pd.DataFrame:
    """
    Load upcoming fixtures WITH their latest model predictions from DB.predictions.
    """

    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at: {DB_PATH}")

    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    conn = get_conn()

    query = """
    WITH latest AS (
        SELECT date, home_team, away_team, MAX(created_at) AS max_created_at
        FROM predictions
        WHERE model_version LIKE 'dc_elo_ensemble_live%'
        GROUP BY date, home_team, away_team
    )
    SELECT p.*
    FROM predictions p
    JOIN latest l
      ON p.date = l.date
     AND p.home_team = l.home_team
     AND p.away_team = l.away_team
     AND p.created_at = l.max_created_at
    WHERE p.date >= ? AND p.date <= ?
    ORDER BY p.date, p.home_team, p.away_team;
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=(today.isoformat(), cutoff.isoformat()),
    )
    conn.close()

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    return df


# ============================================================
# CHATGPT MESSAGE GENERATION
# ============================================================
def build_chatgpt_messages(fixtures_df: pd.DataFrame) -> list[dict]:

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert football prediction assistant analyzing Premier League matches. "
            "Predict realistic final scorelines based on model probabilities and xG.\n\n"

            "Rules:\n"
            "- You MUST consider draws.\n"
            "- If draw probability is similar to win probability, pick draw.\n"
            "- Use realistic scorelines.\n"
            "- Return ONLY valid JSON.\n\n"

            "Output format:\n"
            "[ {\n"
            '  "date": "YYYY-MM-DD",\n'
            '  "home_team": "Team",\n'
            '  "away_team": "Team",\n'
            '  "score": "H-A"\n'
            "} ]"
        ),
    }

    lines = []
    for _, row in fixtures_df.iterrows():
        d = row["date"].date().isoformat()
        home = row["home_team"]
        away = row["away_team"]
        pH = float(row["home_win_prob"])
        pD = float(row["draw_prob"])
        pA = float(row["away_win_prob"])
        xH = float(row["exp_goals_home"])
        xA = float(row["exp_goals_away"])

        lines.append(
            f"Date: {d} | {home} vs {away} | Prob(H/D/A) = {pH:.3f}/{pD:.3f}/{pA:.3f} | xG(H/A) = {xH:.2f}/{xA:.2f}"
        )

    return [
        system_msg,
        {
            "role": "user",
            "content": (
                "Upcoming Premier League fixtures:\n\n"
                + "\n".join(lines)
                + "\n\nPredict one realistic score per match. JSON only."
            ),
        },
    ]


# ============================================================
# CHATGPT CALL
# ============================================================
def call_chatgpt_for_fixtures(fixtures_df: pd.DataFrame) -> list[dict]:
    if fixtures_df.empty:
        return []

    client = OpenAI()
    messages = build_chatgpt_messages(fixtures_df)

    response = client.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=messages,
        temperature=0.4,
        max_tokens=800,
    )

    raw = response.choices[0].message.content

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        s = raw.find("[")
        e = raw.rfind("]")
        if s != -1 and e != -1:
            return json.loads(raw[s:e+1])
        raise ValueError(f"ChatGPT returned invalid JSON:\n\n{raw}")


# ============================================================
# PIPELINE
# ============================================================
def run_chatgpt_predictions(days_ahead: int = 7) -> pd.DataFrame:

    df = get_upcoming_model_predictions(days_ahead=days_ahead)
    if df.empty:
        print("No upcoming fixtures with model predictions.")
        return pd.DataFrame()

    print(f"Sending {len(df)} fixtures to ChatGPT...")

    chat_outputs = call_chatgpt_for_fixtures(df)

    chat_map = {(x["date"], x["home_team"], x["away_team"]): x["score"] for x in chat_outputs}

    out_rows = []
    upserted = 0

    for _, row in df.iterrows():
        date_str = row["date"].date().isoformat()
        home = row["home_team"]
        away = row["away_team"]

        key = (date_str, home, away)
        if key not in chat_map:
            continue

        chat_score = chat_map[key]

        row_dict = {
            "date": date_str,
            "home_team": home,
            "away_team": away,

            # IMPORTANT — ChatGPT gets its own model_version
            "model_version": "chatgpt",

            "dixon_coles_probs": row.get("dixon_coles_probs", ""),
            "elo_probs": row.get("elo_probs", ""),
            "ensemble_probs": row.get("ensemble_probs", ""),

            "home_win_prob": float(row["home_win_prob"]),
            "draw_prob": float(row["draw_prob"]),
            "away_win_prob": float(row["away_win_prob"]),

            "exp_goals_home": float(row["exp_goals_home"]),
            "exp_goals_away": float(row["exp_goals_away"]),
            "exp_total_goals": float(row.get("exp_total_goals", row["exp_goals_home"] + row["exp_goals_away"])),

            "score_pred": row.get("score_pred"),
            "chatgpt_pred": chat_score,
        }

        insert_predictions(row_dict)
        upserted += 1
        out_rows.append(row_dict)

    print(f"ChatGPT predictions written for {upserted} fixtures.")
    return pd.DataFrame(out_rows)


# ============================================================
# CLI ENTRYPOINT
# ============================================================
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ChatGPT predictions for upcoming matches.")
    parser.add_argument("days_ahead", nargs="?", type=int, default=7)
    return parser.parse_args().days_ahead


def main():
    init_db()

    days_ahead = parse_args()

    print("===================================================")
    print("   Generating ChatGPT Predictions (DB-only)")
    print(f"   Model: {CHATGPT_MODEL}")
    print(f"   Window: Today → Today+{days_ahead} days")
    print("===================================================\n")

    df = run_chatgpt_predictions(days_ahead=days_ahead)

    print(f"\nDone. ChatGPT predictions stored for {len(df)} fixtures.\n")


if __name__ == "__main__":
    main()
