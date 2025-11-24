import sys
from pathlib import Path
import re
from datetime import datetime
import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from db import get_conn, init_db  # noqa: E402


def fetch_upcoming_matches_missing_chatgpt():
    """
    Get upcoming matches (date >= today) that:
      - have model predictions in `predictions`
      - do NOT have a final result yet
      - do NOT yet have a ChatGPT prediction

    This ensures we ONLY use ChatGPT for live / future games,
    never for historical ones.
    """
    today = datetime.utcnow().date().isoformat()

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT p.date,
                   p.home_team,
                   p.away_team,
                   p.model_version
            FROM predictions p
            LEFT JOIN results r
              ON p.date = r.date
             AND p.home_team = r.home_team
             AND p.away_team = r.away_team
            WHERE p.date >= ?
              AND r.date IS NULL
              AND (p.chatgpt_pred IS NULL OR p.chatgpt_pred = '')
            ORDER BY p.date ASC, p.home_team, p.away_team;
            """,
            conn,
            params=(today,),
        )

    return df


def build_prompt(df: pd.DataFrame) -> str:
    """
    Turn fixtures into a ChatGPT prompt.
    """
    lines = [
        "Predict the final score of each of the following upcoming matches.",
        "Respond ONLY in this exact format per line:",
        "HomeTeam AwayTeam Score",
        "For example: Arsenal Chelsea 2-1",
        "",
    ]
    for _, row in df.iterrows():
        lines.append(f"{row['home_team']} {row['away_team']}")
    return "\n".join(lines)


def parse_chatgpt_output(raw_text: str):
    """
    Parse lines like:
      Arsenal Chelsea 2-1
      Man City Tottenham 3-0

    Returns: list of (home_team, away_team, score_str)
    """
    pattern = re.compile(r"^(.+?)\s+(.+?)\s+(\d+-\d+)$")
    out = []

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if m:
            home, away, score = m.groups()
            out.append((home, away, score))

    return out


def update_db_with_chatgpt(rows):
    """
    rows: list of (home_team, away_team, score_str)

    Updates predictions.chatgpt_pred for ALL model_versions
    of those fixtures (date constraint isn't needed here
    because team pairs are unique on upcoming fixtures set).
    """
    with get_conn() as conn:
        cur = conn.cursor()

        for home, away, score in rows:
            cur.execute(
                """
                UPDATE predictions
                SET chatgpt_pred = ?
                WHERE home_team = ?
                  AND away_team = ?
                  AND (chatgpt_pred IS NULL OR chatgpt_pred = '');
                """,
                (score, home, away),
            )

        conn.commit()


def main():
    init_db()

    df = fetch_upcoming_matches_missing_chatgpt()
    if df.empty:
        print("✓ No upcoming matches needing ChatGPT predictions.")
        return

    print(f"Found {len(df)} upcoming matches needing ChatGPT predictions.")

    prompt = build_prompt(df)

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.choices[0].message.content

    parsed = parse_chatgpt_output(raw)
    if not parsed:
        print("✗ Could not parse ChatGPT output:")
        print(raw)
        return

    print(f"Parsed {len(parsed)} ChatGPT predictions.")
    update_db_with_chatgpt(parsed)
    print("✓ ChatGPT predictions stored in DB for upcoming matches.")


if __name__ == "__main__":
    main()
