import sys
from pathlib import Path
import re
from datetime import datetime
import pandas as pd
from openai import OpenAI
from difflib import get_close_matches

ROOT = Path(__file__).resolve().parents[2]  # .../soccer_agent_local
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from db import get_conn, init_db  # noqa: E402


# ============================================================
# TEAM NORMALIZATION & FUZZY MATCHING
# ============================================================
def normalize_team(s: str) -> str:
    """Normalize text for matching."""
    return (
        s.lower()
        .replace(".", "")
        .replace("'", "")
        .replace("’", "")
        .replace("fc", "")
        .replace("afc", "")
        .replace("-", " ")
        .strip()
    )


def fuzzy_team_match(target: str, valid_teams: list[str], cutoff: float = 0.8):
    """
    Return best fuzzy match using difflib.
    Used ONLY when strict matching fails.
    """
    matches = get_close_matches(
        normalize_team(target),
        [normalize_team(t) for t in valid_teams],
        n=1,
        cutoff=cutoff,
    )
    if not matches:
        return None

    # Recover original team name with exact spelling
    norm_map = {normalize_team(t): t for t in valid_teams}
    return norm_map.get(matches[0], None)


# ============================================================
# FETCH UPCOMING MATCHES MISSING CHATGPT
# ============================================================
def fetch_upcoming_matches_missing_chatgpt():
    """
    Returns upcoming matches lacking ChatGPT predictions.

    Conditions:
      - match date >= today
      - exists in predictions table
      - NOT in results table
      - chatgpt_pred IS NULL
    """
    today = datetime.utcnow().date().isoformat()

    with get_conn() as conn:
        df = pd.read_sql(
            """
            SELECT p.date, p.home_team, p.away_team
            FROM predictions p
            LEFT JOIN results r
              ON p.date = r.date
             AND p.home_team = r.home_team
             AND p.away_team = r.away_team
            WHERE p.date >= ?
              AND r.date IS NULL
              AND (p.chatgpt_pred IS NULL OR p.chatgpt_pred = '')
            ORDER BY p.date, p.home_team, p.away_team;
            """,
            conn,
            params=(today,),
        )

    return df


# ============================================================
# COMPILE SYSTEM TEAMS (for matching)
# ============================================================
def get_all_team_names():
    with get_conn() as conn:
        teams = pd.read_sql(
            "SELECT DISTINCT home_team AS team FROM fixtures "
            "UNION SELECT DISTINCT away_team FROM fixtures;",
            conn,
        )
    return sorted(teams["team"].tolist())


# ============================================================
# BUILD PROMPT
# ============================================================
def build_prompt(df: pd.DataFrame):
    lines = [
        "Predict the final score for each match below.",
        "Respond ONLY in this format per line:",
        "HomeTeam AwayTeam Score",
        "Examples:",
        "Arsenal Chelsea 2-1",
        "Man United Liverpool 1-0",
        "",
        "Matches:",
    ]
    for _, row in df.iterrows():
        lines.append(f"{row['home_team']} {row['away_team']}")
    return "\n".join(lines)


# ============================================================
# PARSE CHATGPT OUTPUT — FLEXIBLE PATTERNS
# ============================================================
def parse_chatgpt_output(raw: str):
    """
    Accepts formats like:
      Arsenal Chelsea 2-1
      Man Utd vs Chelsea 1–1
      Brentford - Burnley: 3-1
      Aston Villa 2 - 1 Wolves
      "Tottenham 3-0 Fulham"

    Returns: list of (home, away, score)
    """
    lines = raw.splitlines()
    cleaned = []

    # Normalize Unicode dashes
    raw = raw.replace("–", "-").replace("—", "-")

    # Generic score pattern
    score_re = r"(\d+)\s*-\s*(\d+)"

    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Remove separators
        line = line.replace(" vs ", " ").replace(" VS ", " ").replace(":", " ")

        # Find score first
        score_match = re.search(score_re, line)
        if not score_match:
            continue

        score = f"{score_match.group(1)}-{score_match.group(2)}"

        # Remove score from line, leaving team names
        team_part = re.sub(score_re, "", line).strip()
        parts = team_part.split()

        if len(parts) < 2:
            continue

        # Attempt naive split:
        # Everything except last word = home team
        # last word = away team
        # But some clubs have 2–3 words
        # Example: "Aston Villa Wolves"
        home_candidate = " ".join(parts[:-1])
        away_candidate = parts[-1]

        results.append((home_candidate, away_candidate, score))

    return results


# ============================================================
# UPDATE DATABASE WITH CHATGPT RESULTS
# ============================================================
def update_db_with_chatgpt(rows, valid_teams):
    """
    rows: list of (home_raw, away_raw, score)

    Uses strict → fuzzy matching to determine correct DB team names.
    """
    updates = []

    for raw_home, raw_away, score in rows:
        home = None
        away = None

        # STRICT MATCH FIRST
        for t in valid_teams:
            if normalize_team(t) == normalize_team(raw_home):
                home = t
            if normalize_team(t) == normalize_team(raw_away):
                away = t

        # FUZZY FALLBACK — warn user
        if home is None:
            home = fuzzy_team_match(raw_home, valid_teams)
            print(f"[FUZZY MATCH] '{raw_home}' → '{home}'")

        if away is None:
            away = fuzzy_team_match(raw_away, valid_teams)
            print(f"[FUZZY MATCH] '{raw_away}' → '{away}'")

        if not home or not away:
            print(f"[SKIP] Could not match teams for line: {raw_home} vs {raw_away}")
            continue

        updates.append((home, away, score))

    # Perform DB updates
    with get_conn() as conn:
        cur = conn.cursor()

        for home, away, score in updates:
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


# ============================================================
# MAIN
# ============================================================
def main():
    init_db()

    df = fetch_upcoming_matches_missing_chatgpt()
    if df.empty:
        print("✓ No upcoming matches missing ChatGPT predictions.")
        return

    print(f"Found {len(df)} upcoming matches needing ChatGPT predictions.")

    # All valid team names for fuzzy matching
    valid_teams = get_all_team_names()

    # Build prompt
    prompt = build_prompt(df)

    # Call model
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    print("\n=== RAW CHATGPT OUTPUT ===\n", raw)

    parsed = parse_chatgpt_output(raw)
    print(f"\nParsed {len(parsed)} predictions.")

    update_db_with_chatgpt(parsed, valid_teams)

    print("\n✓ ChatGPT predictions saved to DB.")


if __name__ == "__main__":
    main()
