import pandas as pd
from pathlib import Path
import sys

# Football-Data-style team names (from your epl_master.csv)
VALID_TEAMS = {
    'Arsenal', 'Aston Villa', 'Birmingham', 'Blackburn', 'Bolton', 'Bournemouth',
    'Brentford', 'Brighton', 'Burnley', 'Cardiff', 'Charlton', 'Chelsea',
    'Crystal Palace', 'Everton', 'Fulham', 'Huddersfield', 'Ipswich', 'Leeds',
    'Leicester', 'Liverpool', 'Luton', 'Man City', 'Man United', 'Middlesbrough',
    'Newcastle', 'Norwich', "Nott'm Forest", 'Sheffield United', 'Southampton',
    'Stoke', 'Sunderland', 'Swansea', 'Tottenham', 'Watford', 'West Brom',
    'West Ham', 'Wolves'
}

# All known mappings from common variations → Football-Data names
TEAM_MAP = {
    "Manchester City": "Man City",
    "Man City": "Man City",
    "Man. City": "Man City",

    "Manchester United": "Man United",
    "Manchester Utd": "Man United",   # then mapped below
    "Man Utd": "Man United",

    "Man United": "Man United",

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
    "Sheffield United": "Sheffield United",
}

def normalize(s):
    """Removes unicode, trims, collapses spaces."""
    return (
        s.astype(str)
         .str.replace(r"\xa0", " ", regex=True)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )


def map_team_name(name):
    """Maps scraped team names to Football-Data names."""
    if name in VALID_TEAMS:
        return name
    if name in TEAM_MAP:
        mapped = TEAM_MAP[name]
        if mapped in VALID_TEAMS:
            return mapped
    return None  # means unrecognized


def convert_futures_to_fixtures(
    input_path="data/raw/futures.csv",
    output_path="data/raw/fixtures_today.csv"
):
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)

    # Keep only required columns
    required_cols = ["Date", "HomeTeam", "AwayTeam"]
    fixtures = df[required_cols].copy()

    # Normalize fields
    fixtures["Date"] = pd.to_datetime(fixtures["Date"], errors="coerce")
    fixtures["HomeTeam"] = normalize(fixtures["HomeTeam"])
    fixtures["AwayTeam"] = normalize(fixtures["AwayTeam"])

    # Remove invalid date rows
    fixtures = fixtures.dropna(subset=["Date"])

    # Validate team names
    unmatched = []

    def validate(name):
        mapped = map_team_name(name)
        if mapped is None:
            unmatched.append(name)
            return name  # keep original so user can inspect it
        return mapped

    fixtures["HomeTeam"] = fixtures["HomeTeam"].apply(validate)
    fixtures["AwayTeam"] = fixtures["AwayTeam"].apply(validate)

    # Report issues
    if unmatched:
        print("\n⚠️ WARNING: Unrecognized team names found:")
        for name in sorted(set(unmatched)):
            print(" -", name)
        print("\nFix mapping in TEAM_MAP before running predictions.\n")
    else:
        print("✓ All team names validated and mapped successfully")

    # Save final file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(output_path, index=False)

    print(f"\nSaved fixtures file → {output_path}")


if __name__ == "__main__":
    convert_futures_to_fixtures()
