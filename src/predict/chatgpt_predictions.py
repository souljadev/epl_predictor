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

    Filters:
    - model_version LIKE 'dc_elo_ensemble_live%'
    - date >= today AND date <= today+days_ahead
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
    """
    Balanced, draw-aware ChatGPT prompt (Option B).
    """

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert football prediction assistant analyzing Premier League matches. "
            "You must predict realistic final scorelines based on model probabilities and xG.\n\n"

            "RULES:\n"
            "- You MUST consider draws when probabilities suggest they are likely.\n"
            "- If draw probability is similar to home/away win probabilities, choose a draw.\n"
            "- Use reasonable scorelines (0–0, 1–0, 1–1, 2–1, 2–2, 3–1, etc.).\n"
            "- Use the expected goals (xG) to drive your predictions.\n"
            "- ALWAYS return valid JSON ONLY — no text, no commentary.\n\n"

            "OUTPUT FORMAT:\n"
            "Return a JSON array of objects. Each must have:\n"
            "{\n"
            '  "date": "YYYY-MM-DD",\n'
            '  "home_team": "Team",\n'
            '  "away_team": "Team",\n'
            '  "score": "H-A"\n'
            "}\n"
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
            f"Date: {d} | {home} vs {away} | "
            f"Prob(H/D/A) = {pH:.3f}/{pD:.3f}/{pA:.3f} | "
            f"xG(H/A) = {xH:.2f}/{xA:.2f}"
        )

    user_msg = {
        "role": "user",
        "content": (
            "Upcoming Premier League fixtures with model probabilities and xG:\n\n"
            + "\n".join(lines)
            + "\n\nPredict a single realistic final score for EACH match.\n"
            "Respond ONLY with the JSON array described."
        ),
    }

    return [system_msg, user_msg]


# ============================================================
# CHATGPT CALL
# ============================================================
def call_chatgpt_for_fixtures(fixtures_df: pd.DataFrame) -> list[dict]:
    """
    Calls ChatGPT and returns a list of dicts:
    { "date": str, "home_team": str, "away_team": str, "score": "H-A" }
    """
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

    # Try to parse JSON directly
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting substring
        s = raw.find("[")
        e = raw.rfind("]")
        if s != -1 and e != -1:
            try:
                data = json.loads(raw[s:e+1])
            except Exception:
                raise ValueError(f"ChatGPT returned invalid JSON:\n\n{raw}")
        else:
            raise ValueError(f"ChatGPT returned invalid JSON:\n\n{raw}")

    # Validate
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not all(k in item for k in ("date", "home_team", "away_team", "score")):
            continue
        out.append(
            {
                "date": item["date"].strip(),
                "home_team": item["home_team"].strip(),
                "away_team": item["away_team"].strip(),
                "score": item["score"].strip(),
            }
        )

    return out


# ============================================================
# PIPELINE
# ============================================================
def run_chatgpt_predictions(days_ahead: int = 7) -> pd.DataFrame:
    """
    1. Load latest model predictions from DB
    2. Ask ChatGPT for score predictions
    3. Upsert ChatGPT predictions back into DB.predictions
    """

    df = get_upcoming_model_predictions(days_ahead=days_ahead)
    if df.empty:
        print("No upcoming fixtures with model predictions.")
        return pd.DataFrame()

    print(f"Sending {len(df)} fixtures to ChatGPT...")

    chat_outputs = call_chatgpt_for_fixtures(df)
    if not chat_outputs:
        print("ChatGPT returned no predictions.")
        return pd.DataFrame()

    # Map (date, home, away) → score
    chat_map = {(x["date"], x["home_team"], x["away_team"]): x["score"] for x in chat_outputs}

    # Insert predictions back into DB
    upserted = 0
    out_rows = []

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
            "model_version": row["model_version"],

            "dixon_coles_probs": row.get("dixon_coles_probs", "") or "",
            "elo_probs": row.get("elo_probs", "") or "",
            "ensemble_probs": row.get("ensemble_probs", "") or "",

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
