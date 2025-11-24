# src/predictor.py

from pathlib import Path
from typing import Tuple

import pandas as pd

from models.dixon_coles import DixonColesModel
from models.elo import EloModel
from models.ensemble import ensemble_win_probs


ROOT = Path(__file__).resolve().parents[1]  # project root: soccer_agent_local/


def train_models(
    results_df: pd.DataFrame,
    dc_cfg: dict,
    elo_cfg: dict,
) -> Tuple[DixonColesModel, EloModel]:
    """
    Train Dixon–Coles and Elo models on historical results_df.
    results_df must contain at least: Date, HomeTeam, AwayTeam, FTHG, FTAG.
    """

    # Basic cleaning
    df = results_df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["FTHG", "FTAG"])

    # Train Dixon–Coles
    dc = DixonColesModel(
        rho_init=dc_cfg.get("rho_init", 0.0),
        home_adv_init=dc_cfg.get("home_adv_init", 0.15),
        lr=dc_cfg.get("lr", 0.05),
    )
    use_xg = "Home_xG" in df.columns and "Away_xG" in df.columns
    dc.fit(df, use_xg=use_xg)

    # Train Elo
    elo = EloModel(
        k_factor=elo_cfg.get("k_factor", 18.0),
        home_advantage=elo_cfg.get("home_advantage", 55.0),
        base_rating=elo_cfg.get("base_rating", 1500.0),
        draw_base=elo_cfg.get("draw_base", 0.25),
        draw_max_extra=elo_cfg.get("draw_max_extra", 0.10),
        draw_scale=elo_cfg.get("draw_scale", 400.0),
    )
    elo.fit(df)

    return dc, elo


def predict_fixtures(
    fixtures_df: pd.DataFrame,
    dc_model: DixonColesModel,
    elo_model: EloModel,
    w_dc: float = 0.6,
    w_elo: float = 0.4,
) -> pd.DataFrame:
    """
    Predict probabilities and expected goals for a set of fixtures
    using the provided Dixon–Coles and Elo models.
    fixtures_df must contain: Date, HomeTeam, AwayTeam.
    """

    rows = []
    for _, row in fixtures_df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        try:
            lamH, lamA = dc_model.predict_expected_goals(home, away)
            pH_dc, pD_dc, pA_dc = dc_model.predict(home, away)
        except Exception:
            # Skip if DC cannot handle this fixture
            continue

        try:
            pH_elo, pD_elo, pA_elo = elo_model.predict(home, away)
        except Exception:
            # Skip if Elo cannot handle this fixture
            continue

        pH, pD, pA = ensemble_win_probs(
            (pH_dc, pD_dc, pA_dc),
            (pH_elo, pD_elo, pA_elo),
            w_dc=w_dc,
            w_elo=w_elo,
        )

        rows.append(
            {
                "Date": row["Date"],
                "HomeTeam": home,
                "AwayTeam": away,
                "pH": pH,
                "pD": pD,
                "pA": pA,
                "ExpHomeGoals": lamH,
                "ExpAwayGoals": lamA,
                "ExpTotalGoals": lamH + lamA,
            }
        )

    return pd.DataFrame(rows)
