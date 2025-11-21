import pandas as pd
df = pd.read_csv("models/history/epl_results.csv", parse_dates=["Date"])
print(len(df["Date"].unique()))