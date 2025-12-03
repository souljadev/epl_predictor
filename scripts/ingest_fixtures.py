import sys
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # .../soccer_agent_local
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from db import init_db, insert_fixtures  # noqa: E402

FBREF_XG = ROOT / "data" / "raw" / "fbref_epl_xg.csv"

TEAM_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Brighton and Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Sheffield United": "Sheffield Utd",
    # add any others as needed
}


def normalize_team(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


def load_fbref_fixtures() -> pd.DataFrame:
    if not FBREF_XG.exists():
        raise FileNotFoundError(f"fbref_epl_xg.csv not found at {FBREF_XG}")

    df = pd.read_csv(FBREF_XG)

    possible_date = ["Date", "date"]
    possible_home = ["Home", "HomeTeam", "home_team"]
    possible_away = ["Away", "AwayTeam", "away_team"]
    possible_hg = ["HG", "FTHG", "home_goals", "HomeGoals"]
    possible_ag = ["AG", "FTAG", "away_goals", "AwayGoals"]

    def find_col(options):
        for col in options:
            if col in df.columns:
                return col
        return None

    col_date = find_col(possible_date)
    col_home = find_col(possible_home)
    col_away = find_col(possible_away)
    col_hg = find_col(possible_hg)
    col_ag = find_col(possible_ag)

    for name, col in [
        ("Date", col_date),
        ("Home", col_home),
        ("Away", col_away),
        ("HG", col_hg),
        ("AG", col_ag),
    ]:
        if col is None:
            raise ValueError(f"fbref_epl_xg.csv missing a detectable '{name}' column")

    df = df[[col_date, col_home, col_away, col_hg, col_ag]].copy()
    df.columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

    # Fixtures = no scores yet
    fixtures = df[df["FTHG"].isna() & df["FTAG"].isna()].copy()
    if fixtures.empty:
        return fixtures

    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
    fixtures = fixtures.dropna(subset=["Date"])

    # Only keep fixtures today or later
    today = pd.Timestamp(date.today())
    fixtures = fixtures[fixtures["Date"] >= today].copy()

    fixtures["HomeTeam"] = fixtures["HomeTeam"].apply(normalize_team)
    fixtures["AwayTeam"] = fixtures["AwayTeam"].apply(normalize_team)

    return fixtures[["Date", "HomeTeam", "AwayTeam"]]


def main():
    print("\n===========================================")
    print("   Ingest FUTURE fixtures into DB.fixtures")
    print("===========================================\n")

    init_db()

    fixtures_df = load_fbref_fixtures()
    if fixtures_df.empty:
        print("No future fixtures found in fbref_epl_xg.csv.")
        return

    print(f"Loaded {len(fixtures_df)} fixtures from fbref_epl_xg.csv")
    insert_fixtures(fixtures_df)

    print("✓ Fixtures ingested/upserted into DB.fixtures.\n")


if __name__ == "__main__":
    main()
