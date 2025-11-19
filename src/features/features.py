import pandas as pd
from .utils import standardize_team_names

def build_basic_features(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df = standardize_team_names(df)
    df["HomePts"] = (df["Result"] == "H").astype(int)*3 + (df["Result"] == "D").astype(int)*1
    df["AwayPts"] = (df["Result"] == "A").astype(int)*3 + (df["Result"] == "D").astype(int)*1
    return df
