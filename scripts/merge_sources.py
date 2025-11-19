import pandas as pd
from pathlib import Path

BASE = Path("data/raw")

# Map FROM fbref team names TO master team names
TEAM_MAP = {
    # FBref name: Master name
    "Leeds United": "Leeds",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "Nott'ham Forest": "Nott'm Forest",
    "Brighton and Hove Albion": "Brighton",
    "Brighton & Hove Albion": "Brighton",
    "Wolverhampton Wanderers": "Wolves",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "West Bromwich Albion": "West Brom",
    "West Bromwich": "West Brom",
    "Sheffield Utd": "Sheffield United",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Southampton": "Southampton",
    "Watford": "Watford",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
}

def normalize(s):
    """Remove weird unicode, trim, collapse whitespace."""
    return (
        s.astype(str)
         .str.replace(r"\xa0", " ", regex=True)
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )

def combine():
    print("Loading files…")
    master = pd.read_csv(BASE/"epl_master.csv")
    fbref = pd.read_csv(BASE/"fbref_epl_xg.csv")
    
    print(f"\nMaster file: {len(master)} rows")
    print(f"FBref file: {len(fbref)} rows")

    # Normalize date formats
    master["Date"] = pd.to_datetime(master["Date"], errors="coerce")
    fbref["Date"] = pd.to_datetime(fbref["Date"], errors="coerce")

    # Normalize team names
    for df in (master, fbref):
        df["HomeTeam"] = normalize(df["HomeTeam"])
        df["AwayTeam"] = normalize(df["AwayTeam"])

    # Apply mapping: Map fbref team names to master team names
    fbref["HomeTeam"] = fbref["HomeTeam"].replace(TEAM_MAP)
    fbref["AwayTeam"] = fbref["AwayTeam"].replace(TEAM_MAP)
    
    print("\n" + "="*60)
    print("COLUMN ANALYSIS")
    print("="*60)
    
    print(f"\nMaster columns: {list(master.columns)}")
    print(f"FBref columns: {list(fbref.columns)}")
    
    # Find common columns
    common_cols = set(master.columns).intersection(set(fbref.columns))
    print(f"\n✅ Common columns: {sorted(common_cols)}")
    
    # Columns only in fbref (the xG columns)
    only_fbref = set(fbref.columns) - set(master.columns)
    if only_fbref:
        print(f"\n➕ New columns from fbref: {sorted(only_fbref)}")
        print("   (Master rows will have NaN for these columns)")

    print("\n" + "="*60)
    print("COMBINING DATA")
    print("="*60)
    
    # Concatenate the dataframes (append fbref to master)
    combined = pd.concat([master, fbref], ignore_index=True, sort=False)
    
    # Sort by date
    combined = combined.sort_values("Date").reset_index(drop=True)
    
    print(f"\n✅ Combined total rows: {len(combined)}")
    print(f"   Master rows: {len(master)}")
    print(f"   FBref rows: {len(fbref)}")
    print(f"   Total: {len(master)} + {len(fbref)} = {len(combined)}")
    
    # Show date range
    print(f"\nDate range: {combined['Date'].min()} to {combined['Date'].max()}")
    
    # Show which rows have xG data
    has_xg = combined[["Home_xG", "Away_xG"]].notna().any(axis=1).sum()
    print(f"\nRows with xG data: {has_xg} / {len(combined)} ({has_xg/len(combined)*100:.1f}%)")

    out = BASE/"epl_combined.csv"
    combined.to_csv(out, index=False)
    print(f"\n💾 Saved: {out}")
    
    # Show sample from each source
    print("\n" + "="*60)
    print("SAMPLE DATA")
    print("="*60)
    
    print("\n📊 First 3 rows (from master - no xG data):")
    sample_cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    if "Home_xG" in combined.columns:
        sample_cols.extend(["Home_xG", "Away_xG"])
    print(combined.head(3)[sample_cols].to_string(index=False))
    
    print("\n📊 Last 3 rows (from fbref - includes xG):")
    print(combined.tail(3)[sample_cols].to_string(index=False))


if __name__ == "__main__":
    combine()