import pandas as pd
import cloudscraper
from pathlib import Path

BASE = Path("data/raw")
BASE.mkdir(parents=True, exist_ok=True)

URL = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

def scrape_fbref():
    print("Scraping FBref with Cloudscraper…")

    # Create a scraper session that bypasses Cloudflare
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "windows",
            "mobile": False
        }
    )

    # Get page HTML
    html = scraper.get(URL).text

    # Extract the fixtures table
    tables = pd.read_html(html, match="Scores & Fixtures", flavor="lxml")

    if len(tables) == 0:
        print("ERROR: Could not find fixtures table.")
        return

    df = tables[0]

    # Rename core columns
    df = df.rename(columns={
        "Home": "HomeTeam",
        "Away": "AwayTeam",
        "xG": "Home_xG",
        "xG.1": "Away_xG",
        "Score": "Score"
    })

    # Extract goals from Score (e.g., "2–1" or "3-0")
    df["FTHG"] = df["Score"].astype(str).str.extract(r'(\d+)[–-]').astype(float)
    df["FTAG"] = df["Score"].astype(str).str.extract(r'[–-](\d+)').astype(float)

    # Compute match result
    def result(r):
        if pd.isna(r["FTHG"]) or pd.isna(r["FTAG"]):
            return None
        if r["FTHG"] > r["FTAG"]:
            return "H"
        if r["FTHG"] < r["FTAG"]:
            return "A"
        return "D"

    df["Result"] = df.apply(result, axis=1)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Keep only relevant columns
    df_clean = df[[
        "Date", "HomeTeam", "AwayTeam",
        "FTHG", "FTAG", "Result",
        "Home_xG", "Away_xG"
    ]]

    out = BASE / "fbref_epl_xg.csv"
    df_clean.to_csv(out, index=False)

    print("Saved:", out)


if __name__ == "__main__":
    scrape_fbref()
