import pandas as pd

df = pd.read_csv("data/raw/epl_combined.csv")

print("Columns:", df.columns)

print("\nCheck FTHG NaN count:", df["FTHG"].isna().sum())
print("Check FTAG NaN count:", df["FTAG"].isna().sum())

print("\nHome goals mean:", df["FTHG"].mean())
print("Away goals mean:", df["FTAG"].mean())
