import sys
from pathlib import Path
from datetime import date

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # .../soccer_agent_local
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from db import init_db, insert_results  # noqa: E402

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
EPL_MASTER = ROOT / "data" / "raw" / "epl_master.csv"
FBREF_XG = ROOT / "data" / "raw" / "fbref_epl_xg.csv"

# ---------------------------------------------------------------------
# Team normalization (FBref / other → your DB naming)
# ---------------------------------------------------------------------
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
    # add any others you've used in fixtures/results
}


def normalize_team(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_epl_master() -> pd.DataFrame:
    if not EPL_MASTER.exists():
        print(f"[WARN] epl_master.csv not found at {EPL_MASTER}")
        return pd.DataFrame()

    df = pd.read_csv(EPL_MASTER)

    # Expect these columns; adjust if your file differs
    required = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"epl_master.csv missing columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG"]).copy()

    # Normalize team names
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team)

    return df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]


def load_fbref_results() -> pd.DataFrame:
    if not FBREF_XG.exists():
        print(f"[WARN] fbref_epl_xg.csv not found at {FBREF_XG}")
        return pd.DataFrame()

    df = pd.read_csv(FBREF_XG)

    # Dynamically detect columns (since fbref naming can vary)
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

    # Completed matches only → drop rows with NaN goals
    df = df.dropna(subset=["FTHG", "FTAG"]).copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    # Only keep matches up to today (avoid weird future pre-filled rows)
    today = pd.Timestamp(date.today())
    df = df[df["Date"] <= today].copy()

    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team)

    return df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    print("\n===========================================")
    print("   Ingest ALL EPL results into DB.results")
    print("===========================================\n")

    init_db()

    master_df = load_epl_master()
    fbref_df = load_fbref_results()

    print(f"Loaded {len(master_df)} rows from epl_master.csv")
    print(f"Loaded {len(fbref_df)} rows from fbref_epl_xg.csv")

    if master_df.empty and fbref_df.empty:
        print("No results loaded from either source. Nothing to ingest.")
        return

    all_df = pd.concat([master_df, fbref_df], ignore_index=True)

    # Drop duplicates by (Date, HomeTeam, AwayTeam)
    all_df.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam"],
        keep="last",
        inplace=True,
    )

    print(f"Total unique matches to upsert into DB: {len(all_df)}")

    # Let db.insert_results handle Result column and upsert logic
    insert_results(all_df)

    print("✓ Ingestion complete. DB.results is now backfilled.\n")


if __name__ == "__main__":
    main()
