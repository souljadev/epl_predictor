import numpy as np
import pandas as pd

def set_seed(seed: int = 1337):
    import random
    random.seed(seed)
    np.random.seed(seed)

def standardize_team_names(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["HomeTeam", "AwayTeam"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df

def train_test_split_by_season(df: pd.DataFrame, test_start_season: int):
    train = df[df["Season"] < test_start_season].copy()
    test = df[df["Season"] >= test_start_season].copy()
    return train, test
