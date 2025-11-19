import pandas as pd
import requests
from io import BytesIO
from pathlib import Path

BASE = Path("data/raw")
BASE.mkdir(parents=True, exist_ok=True)

# Football-Data EPL season codes
SEASON_CODES = {
    1993: "9394", 1994: "9495", 1995: "9596", 1996: "9697",
    1997: "9798", 1998: "9899", 1999: "9900", 2000: "0001",
    2001: "0102", 2002: "0203", 2003: "0304", 2004: "0405",
    2005: "0506", 2006: "0607", 2007: "0708", 2008: "0809",
    2009: "0910", 2010: "1011", 2011: "1112", 2012: "1213",
    2013: "1314", 2014: "1415", 2015: "1516", 2016: "1617",
    2017: "1718", 2018: "1819", 2019: "1920", 2020: "2021",
    2021: "2122", 2022: "2223", 2023: "2324", 2024: "2425"
}

def download_and_read(url: str):
    """Download CSV with robust encoding fallbacks."""
    r = requests.get(url, timeout=10)
    if r.status_code != 200:
        return None

    raw = r.content

    # Try UTF-8 first, then fall back to Latin-1 / cp1252
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(
                BytesIO(raw),
                encoding=enc,
                on_bad_lines="skip",
                engine="python"
            )
            return df
        except Exception:
            continue

    return None  # Completely unreadable file

def clean(df):
    """Extract and clean only the columns we need."""
    col_map = {
        "HomeTeam": "HomeTeam",
        "AwayTeam": "AwayTeam",
        "FTHG": "FTHG",
        "FTAG": "FTAG",
        "FTR": "Result",
        "Date": "Date"
    }

    usable = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=usable)
    df = df[list(usable.values())]

    # Parse date with multiple formats
    for fmt in ["%d/%m/%Y", "%d/%m/%y", None]:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format=fmt)
        if df["Date"].notna().sum() > 0:
            break

    df = df.dropna(subset=["Date"])

    return df

def build_master():
    all_dfs = []

    for year, code in SEASON_CODES.items():
        url = f"https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
        print("Downloading:", url)

        df = download_and_read(url)
        if df is None:
            print("FAILED:", url)
            continue

        df["Season"] = f"{year}-{year+1}"
        df = clean(df)

        if df.empty:
            print("WARNING: empty after cleaning →", url)
            continue

        all_dfs.append(df)

    master = pd.concat(all_dfs, ignore_index=True)
    master = master.sort_values("Date")

    out = BASE / "epl_master.csv"
    master.to_csv(out, index=False)
    print("\nSUCCESS → saved:", out)


if __name__ == "__main__":
    build_master()
