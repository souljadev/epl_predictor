import sys
import json
import sqlite3
from pathlib import Path
from datetime import date, timedelta

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

# Debug settings
DEBUG_TRUNCATE_PROMPT = 1000
DEBUG_TRUNCATE_RESPONSE = 2000

# ChatGPT batching
CHATGPT_BATCH_SIZE = 5


# ============================================================
# DB HELPERS
# ============================================================
def get_conn():
    return sqlite3.connect(DB_PATH)


def get_upcoming_model_predictions(days_ahead: int = 7) -> pd.DataFrame:
    """
    Load upcoming fixtures WITH their latest model predictions from DB.predictions.
    Option C: ignore model_version, select latest by created_at.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB not found at: {DB_PATH}")

    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    conn = get_conn()

    query = """
    WITH latest AS (
        SELECT
            date,
            home_team,
            away_team,
            MAX(created_at) AS max_created_at
        FROM predictions
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
        lines.append(
            f"Date: {row['date'].date().isoformat()} | "
            f"{row['home_team']} vs {row['away_team']} | "
            f"Prob(H/D/A) = "
            f"{row['home_win_prob']:.3f}/"
            f"{row['draw_prob']:.3f}/"
            f"{row['away_win_prob']:.3f} | "
            f"xG(H/A) = "
            f"{row['exp_goals_home']:.2f}/"
            f"{row['exp_goals_away']:.2f}"
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
# CHATGPT CALL (BATCHED)
# ============================================================
def call_chatgpt_for_fixtures(
    fixtures_df: pd.DataFrame, debug: bool = False
) -> list[dict]:

    if fixtures_df.empty:
        return []

    client = OpenAI()
    all_outputs: list[dict] = []

    batches = [
        fixtures_df.iloc[i : i + CHATGPT_BATCH_SIZE]
        for i in range(0, len(fixtures_df), CHATGPT_BATCH_SIZE)
    ]

    for idx, batch in enumerate(batches, start=1):
        messages = build_chatgpt_messages(batch)

        if debug:
            print(f"\n=== CHATGPT BATCH {idx}/{len(batches)} ===")
            print(messages[-1]["content"][:DEBUG_TRUNCATE_PROMPT])
            print("... [truncated]" if len(messages[-1]["content"]) > DEBUG_TRUNCATE_PROMPT else "")
            print("=====================================\n")

        response = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=800,
        )

        raw = response.choices[0].message.content

        if debug:
            print("--- RAW RESPONSE (truncated) ---")
            print(raw[:DEBUG_TRUNCATE_RESPONSE])
            print("... [truncated]" if len(raw) > DEBUG_TRUNCATE_RESPONSE else "")
            print("-------------------------------")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            s = raw.find("[")
            e = raw.rfind("]")
            if s != -1 and e != -1:
                parsed = json.loads(raw[s : e + 1])
            else:
                raise ValueError(f"Invalid JSON from ChatGPT batch {idx}")

        all_outputs.extend(parsed)

    return all_outputs


# ============================================================
# PIPELINE
# ============================================================
def run_chatgpt_predictions(days_ahead: int = 7, debug: bool = False) -> pd.DataFrame:

    df = get_upcoming_model_predictions(days_ahead=days_ahead)
    if df.empty:
        print("No upcoming fixtures with model predictions.")
        return pd.DataFrame()

    print(f"Sending {len(df)} fixtures to ChatGPT in batches of {CHATGPT_BATCH_SIZE}...")

    chat_outputs = call_chatgpt_for_fixtures(df, debug=debug)

    chat_map = {
        (x["date"], x["home_team"], x["away_team"]): x["score"]
        for x in chat_outputs
    }

    out_rows = []
    upserted = 0

    for _, row in df.iterrows():
        key = (
            row["date"].date().isoformat(),
            row["home_team"],
            row["away_team"],
        )

        if key not in chat_map:
            continue

        row_dict = {
            "date": key[0],
            "home_team": key[1],
            "away_team": key[2],
            "model_version": "chatgpt",

            # REQUIRED DB FIELDS — PRESERVED
            "dixon_coles_probs": row.get("dixon_coles_probs", ""),
            "elo_probs": row.get("elo_probs", ""),
            "ensemble_probs": row.get("ensemble_probs", ""),

            "home_win_prob": float(row["home_win_prob"]),
            "draw_prob": float(row["draw_prob"]),
            "away_win_prob": float(row["away_win_prob"]),

            "exp_goals_home": float(row["exp_goals_home"]),
            "exp_goals_away": float(row["exp_goals_away"]),
            "exp_total_goals": float(
                row.get(
                    "exp_total_goals",
                    row["exp_goals_home"] + row["exp_goals_away"],
                )
            ),

            "score_pred": row.get("score_pred"),
            "chatgpt_pred": chat_map[key],
        }

        insert_predictions(row_dict)
        upserted += 1
        out_rows.append(row_dict)

    print(f"ChatGPT predictions written for {upserted} fixtures.")
    return pd.DataFrame(out_rows)


# ============================================================
# CLI
# ============================================================
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ChatGPT predictions for upcoming matches."
    )
    parser.add_argument("days_ahead", nargs="?", type=int, default=7)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args.days_ahead, args.debug


def main():
    init_db()
    days_ahead, debug = parse_args()

    print("===================================================")
    print("   Generating ChatGPT Predictions (DB-only)")
    print(f"   Model: {CHATGPT_MODEL}")
    print(f"   Window: Today → Today+{days_ahead} days")
    print(f"   Debug: {debug}")
    print("===================================================\n")

    df = run_chatgpt_predictions(days_ahead=days_ahead, debug=debug)
    print(f"\nDone. ChatGPT predictions stored for {len(df)} fixtures.\n")


if __name__ == "__main__":
    main()
