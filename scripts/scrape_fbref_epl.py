import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

# ------------------------------------------------------------------
# Ensure project root + src is importable
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import DB helpers
from db import insert_fixtures, insert_results

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

log_file = LOGS / "football_data_scrape.log"

# ------------------------------------------------------------------
# football-data.co.uk EPL CSV (2025–26)
# ------------------------------------------------------------------
EPL_CSV = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"

# ------------------------------------------------------------------
# TEAM NAME NORMALIZATION (AUTHORITATIVE)
# These MUST match names used by models / ChatGPT / dashboards
# ------------------------------------------------------------------
TEAM_FIX = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Brighton & Hove Albion": "Brighton",
    "Sheffield United": "Sheffield Utd",
    "Luton Town": "Luton",
}

EPL_TEAMS_2024 = {
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nott'm Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
}

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("football-data EPL scraper started")

# ------------------------------------------------------------------
# UPSERT helpers
# ------------------------------------------------------------------
def upsert_to_main_db(df: pd.DataFrame) -> int:
    """
    Upserts fixtures (all matches) and results (only played matches).
    Returns number of rows processed (not necessarily newly inserted).
    """
    df_epl = pd.DataFrame(
        {
            "Date": df["Date"],
            "HomeTeam": df["HomeTeam"],
            "AwayTeam": df["AwayTeam"],
            "FTHG": df["FTHG"],
            "FTAG": df["FTAG"],
        }
    )

    # Fixtures: include future matches
    insert_fixtures(df_epl)

    # Results: only completed matches
    results_df = df_epl.dropna(subset=["FTHG", "FTAG"])
    insert_results(results_df)

    return len(df_epl)

# ------------------------------------------------------------------
# Main scrape function
# ------------------------------------------------------------------
def scrape_football_data():
    print("Scraping EPL data from football-data.co.uk…")

    try:
        df = pd.read_csv(EPL_CSV)
    except Exception as e:
        logging.error(f"Failed to download football-data CSV: {e}")
        raise RuntimeError("❌ Could not download football-data CSV")

    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"❌ Unexpected CSV structure: {df.columns}")

    # ------------------------------------------------------------------
    # Normalize dates (DD/MM/YYYY)
    # ------------------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df[df["Date"].notna()].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Normalize team names (CRITICAL)
    # ------------------------------------------------------------------
    df["HomeTeam"] = df["HomeTeam"].replace(TEAM_FIX)
    df["AwayTeam"] = df["AwayTeam"].replace(TEAM_FIX)

    # ------------------------------------------------------------------
    # Sanity check: unknown teams
    # ------------------------------------------------------------------
    all_teams = set(df["HomeTeam"]) | set(df["AwayTeam"])
    unknown = all_teams - EPL_TEAMS_2024
    if unknown:
        logging.warning(f"Unknown teams detected: {sorted(unknown)}")

    # ------------------------------------------------------------------
    # Save CSV snapshot for debugging
    # ------------------------------------------------------------------
    csv_path = DATA / "fixtures_football_data.csv"
    df.to_csv(csv_path, index=False)

    print(f"📄 Saved CSV → {csv_path}")
    print(f"🔢 Rows scraped: {len(df)}")

    # ------------------------------------------------------------------
    # UPSERT into DB
    # ------------------------------------------------------------------
    processed = upsert_to_main_db(df)

    print(f"✅ Processed {processed} rows (upsert attempted)")
    logging.info(f"Processed {processed} rows")

    print("✔ Scrape complete.")

# ------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------
if __name__ == "__main__":
    scrape_football_data()
