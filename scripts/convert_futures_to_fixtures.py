import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

# ============================================================
# FIX: ensure the src/ folder is on PYTHONPATH
# This allows: from db import init_db, upsert_fixtures_from_df
# ============================================================
ROOT = Path(__file__).resolve().parents[1]   # soccer_agent_local/
SRC = ROOT / "src"
sys.path.append(str(SRC))

from db import init_db, upsert_fixtures_from_df  # noqa: E402


# Football-Data-style team names
VALID_TEAMS = {
    "Arsenal", "Aston Villa", "Birmingham", "Blackburn", "Bolton", "Bournemouth",
    "Brentford", "Brighton", "Burnley", "Cardiff", "Charlton", "Chelsea",
    "Crystal Palace", "Everton", "Fulham", "Huddersfield", "Ipswich", "Leeds",
    "Leicester", "Liverpool", "Luton", "Man City", "Man United", "Middlesbrough",
    "Newcastle", "Norwich", "Nott'm Forest", "Sheffield United", "Southampton",
    "Stoke", "Sunderland", "Swansea", "Tottenham", "Watford", "West Brom",
    "West Ham", "Wolves",
}

TEAM_MAP = {
    "Manchester City": "Man City",
    "Man. City": "Man City",
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Man Utd": "Man United",
    "Leeds United": "Leeds",
    "Newcastle United": "Newcastle",
    "Newcastle Utd": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nott'ham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "Wolverhampton": "Wolves",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "West Bromwich Albion": "West Brom",
    "West Bromwich": "West Brom",
    "Sheffield Utd": "Sheffield United",
}


def normalize(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\xa0", " ", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def map_team_name(name: str):
    if name in VALID_TEAMS:
        return name
    if name in TEAM_MAP:
        mapped = TEAM_MAP[name]
        if mapped in VALID_TEAMS:
            return mapped
    return None


def convert_futures_to_fixtures(
    input_path: str = "data/raw/futures.csv",
    output_path: str = "data/raw/fixtures_today.csv",
):
    init_db()

    input_path = ROOT / input_path
    output_path = ROOT / output_path

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = ["Date", "HomeTeam", "AwayTeam"]
    fixtures = df[required_cols].copy()

    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
    fixtures["HomeTeam"] = normalize(fixtures["HomeTeam"])
    fixtures["AwayTeam"] = normalize(fixtures["AwayTeam"])

    fixtures = fixtures.dropna(subset=["Date"])

    today = pd.Timestamp.now().normalize()
    cutoff = today + timedelta(days=7)

    before_count = len(fixtures)
    fixtures = fixtures[(fixtures["Date"] >= today) & (fixtures["Date"] <= cutoff)]
    after_count = len(fixtures)

    print(f"Date filter: {before_count} → {after_count}")
    print(f"Keeping fixtures between: {today.date()} and {cutoff.date()}")

    if fixtures.empty:
        print("⚠ No fixtures within 7 days. No output file created or DB upserted.")
        return

    unmatched = []

    def validate(name: str):
        mapped = map_team_name(name)
        if mapped is None:
            unmatched.append(name)
            return name
        return mapped

    fixtures["HomeTeam"] = fixtures["HomeTeam"].apply(validate)
    fixtures["AwayTeam"] = fixtures["AwayTeam"].apply(validate)

    if unmatched:
        print("\n⚠ WARNING: Unrecognized team names:")
        for name in sorted(set(unmatched)):
            print(" -", name)
        print("Fix TEAM_MAP before running predictions.\n")
    else:
        print("✓ All team names mapped successfully")

    # Save rolling view CSV (not historical)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)
    print(f"\n✓ Fixtures saved to CSV → {output_path}")

    # Insert new fixtures into DB (real storage)
    upsert_fixtures_from_df(fixtures, source="futures")
    print("✓ Fixtures upserted into SQLite 'fixtures' table.")



if __name__ == "__main__":
    convert_futures_to_fixtures()
