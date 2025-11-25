import pandas as pd
from pathlib import Path
import sqlite3

# Paths
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "soccer_agent.db"
FBREF_CSV = ROOT / "data" / "raw" / "fbref_epl_xg.csv"

# -----------------------------------------------------------------------------
# Team normalization (FBref format → Your DB format)
# -----------------------------------------------------------------------------
TEAM_MAP = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Brighton and Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Chelsea": "Chelsea",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Fulham": "Fulham",
    "Everton": "Everton",
    "Burnley": "Burnley",
    "Leeds United": "Leeds",
    "Aston Villa": "Aston Villa",
    "Crystal Palace": "Crystal Palace",
    "Southampton": "Southampton",
    "Sheffield United": "Sheffield Utd",
    # Add more if needed
}

def normalize_team(name):
    name = str(name).strip()
    return TEAM_MAP.get(name, name)


# -----------------------------------------------------------------------------
# Load and clean FBref CSV
# -----------------------------------------------------------------------------
def load_fbref_results():
    df = pd.read_csv(FBREF_CSV)

    # Identify column names dynamically
    possible_date = ["Date", "date"]
    possible_home = ["Home", "HomeTeam", "home_team"]
    possible_away = ["Away", "AwayTeam", "away_team"]
    possible_hg = ["HG", "FTHG", "home_goals", "HomeGoals"]
    possible_ag = ["AG", "FTAG", "away_goals", "AwayGoals"]

    def find_col(df, options):
        for col in options:
            if col in df.columns:
                return col
        raise ValueError(f"Required column not found in CSV. Tried: {options}")

    col_date = find_col(df, possible_date)
    col_home = find_col(df, possible_home)
    col_away = find_col(df, possible_away)
    col_hg = find_col(df, possible_hg)
    col_ag = find_col(df, possible_ag)

    df = df[[col_date, col_home, col_away, col_hg, col_ag]].copy()
    df.columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]

    # Drop rows without scores
    df = df.dropna(subset=["FTHG", "FTAG"])

    # Convert
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    df["HomeTeam"] = df["HomeTeam"].apply(normalize_team)
    df["AwayTeam"] = df["AwayTeam"].apply(normalize_team)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)

    # Result: H / A / D
    df["Result"] = df.apply(
        lambda r: "H" if r["FTHG"] > r["FTAG"] else ("A" if r["FTAG"] > r["FTHG"] else "D"),
        axis=1,
    )

    return df


# -----------------------------------------------------------------------------
# Insert into SQLite results table
# -----------------------------------------------------------------------------
def insert_results(df):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executemany(
        """
        INSERT INTO results (date, home_team, away_team, FTHG, FTAG, Result)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, home_team, away_team) DO UPDATE SET
            FTHG=excluded.FTHG,
            FTAG=excluded.FTAG,
            Result=excluded.Result;
        """,
        df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]].values.tolist(),
    )

    conn.commit()
    conn.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    print("\n===========================================")
    print("   FBref EPL Results → DB Ingestion")
    print("===========================================\n")

    df = load_fbref_results()
    print(f"Loaded {len(df)} matches with results.")

    insert_results(df)
    print(f"✓ Inserted/updated {len(df)} results into DB.")
    print("Done.\n")


if __name__ == "__main__":
    main()
