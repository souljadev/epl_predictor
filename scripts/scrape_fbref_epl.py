"""
scrape_fbref_epl.py
Scrapes EPL Scores & Fixtures from FBref and writes them into the main DB
via db.insert_fixtures() and db.insert_results().

Fully integrated with:
    - data/soccer_agent.db
    - db.py UPSERT helpers
"""

import sys
import re
import pandas as pd
from io import StringIO
from pathlib import Path
import cloudscraper
from datetime import datetime
import logging

# ------------------------------------------------------------------
# Ensure project root + src is importable
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import DB helpers (uses the correct soccer_agent.db path)
from db import insert_fixtures, insert_results

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

FBREF_URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"
log_file = LOGS / "fbref_scrape.log"

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("FBref scraper started")


# ------------------------------------------------------------------
# Extract all tables, including commented ones
# ------------------------------------------------------------------
def extract_all_tables(html: str):
    dfs = []

    # FBref hides the real tables inside HTML comments
    commented = re.findall(r"<!--(.*?)-->", html, flags=re.DOTALL)
    for block in commented:
        try:
            dfs.extend(pd.read_html(StringIO(block), flavor="lxml"))
        except Exception:
            pass

    # Also try visible tables
    try:
        dfs.extend(pd.read_html(StringIO(html), flavor="lxml"))
    except Exception:
        pass

    return dfs


# ------------------------------------------------------------------
# Identify fixture table
# ------------------------------------------------------------------
def find_fixture_table(tables):
    required_cols = {"Date", "Home", "Away"}

    for t in tables:
        if required_cols.issubset(t.columns):
            return t

    logging.error("FBref fixtures table not found — HTML structure changed.")
    raise ValueError("❌ Could not locate Fixtures table on FBref")


# ------------------------------------------------------------------
# Date parsing
# ------------------------------------------------------------------
def parse_date(x):
    try:
        return datetime.strptime(x, "%Y-%m-%d")
    except Exception:
        return None


# ------------------------------------------------------------------
# UPSERT into main DB
# ------------------------------------------------------------------
def upsert_to_main_db(df: pd.DataFrame):
    """
    Convert FBref format → db.py-friendly DataFrame:
        Date, HomeTeam, AwayTeam, FTHG, FTAG
    Then pass to:
        insert_fixtures()
        insert_results()
    which perform correct UPSERTs in soccer_agent.db.
    """

    df_epl = pd.DataFrame(
        {
            "Date": df["match_date"],
            "HomeTeam": df["home_team"],
            "AwayTeam": df["away_team"],
            "FTHG": df["home_goals"],
            "FTAG": df["away_goals"],
        }
    )

    # Insert fixtures (future + past)
    insert_fixtures(df_epl)

    # Insert results (only rows with scores)
    insert_results(df_epl)

    return len(df_epl)


# ------------------------------------------------------------------
# Main scrape function
# ------------------------------------------------------------------
def scrape_fbref():
    print("Scraping FBref EPL fixtures…")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
    }

    scraper = cloudscraper.create_scraper()
    response = scraper.get(FBREF_URL, headers=headers)
    html = response.text

    # --------------------------------------------------------------
    # Blocked detection
    # --------------------------------------------------------------
    if "<table" not in html and "<!--" not in html:
        print("❌ Blocked by FBref / Cloudflare")
        debug_path = ROOT / "blocked_fbref.html"
        debug_path.write_text(html, encoding="utf-8")
        logging.error("FBref BLOCKED — saved blocked HTML.")
        raise RuntimeError("FBref returned no table content — likely bot protection")

    # --------------------------------------------------------------
    # Extract all tables
    # --------------------------------------------------------------
    tables = extract_all_tables(html)
    if not tables:
        logging.error("No tables extracted — layout changed?")
        raise RuntimeError("❌ No tables extracted from FBref")

    fixtures = find_fixture_table(tables).copy()

    # Remove duplicated header rows
    fixtures = fixtures[fixtures["Date"] != "Date"]

    # Warn if FBref changes structure
    expected = {"Date", "Home", "Away", "Score"}
    actual = set(fixtures.columns)
    if not expected.issubset(actual):
        logging.warning(f"Unexpected FBref column structure: {actual}")

    # --------------------------------------------------------------
    # Parse dates & scores
    # --------------------------------------------------------------
    fixtures["match_date"] = fixtures["Date"].astype(str).apply(parse_date)

    def split_score(x):
        if isinstance(x, str) and "-" in x:
            h, a = x.split("-")
            try:
                return int(h), int(a)
            except ValueError:
                return None, None
        return None, None

    fixtures["home_goals"], fixtures["away_goals"] = zip(*fixtures["Score"].apply(split_score))

    out = fixtures[
        ["match_date", "Home", "Away", "home_goals", "away_goals", "Score"]
    ].rename(
        columns={
            "Home": "home_team",
            "Away": "away_team",
            "Score": "score",
        }
    )

    out = out[out["match_date"].notnull()].reset_index(drop=True)
    out = out.sort_values("match_date")

    # Save CSV for debugging
    csv_path = DATA / "fixtures_fbref.csv"
    out.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV → {csv_path}")
    print(f"🔢 Rows scraped: {len(out)}")

    # --------------------------------------------------------------
    # UPSERT into main DB
    # --------------------------------------------------------------
    inserted = upsert_to_main_db(out)

    print(f"✅ Upserted {inserted} rows into data/soccer_agent.db")
    logging.info(f"Upserted {inserted} fixture/results rows")

    print("\n✔ Scrape complete.")
    print(f"📌 Log written to: {log_file}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    scrape_fbref()
