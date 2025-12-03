# src/predictor.py

from pathlib import Path
from typing import Tuple

import pandas as pd

from models.dixon_coles import DixonColesModel
from models.elo import EloModel
from models.ensemble import ensemble_win_probs


ROOT = Path(__file__).resolve().parents[1]  # project root: soccer_agent_local/


# =====================================================================
# TRAIN MODELS
# =====================================================================

def train_models(
    results_df: pd.DataFrame,
    dc_params: dict | None = None,
    elo_params: dict | None = None,
) -> Tuple[DixonColesModel, EloModel]:
    """
    Train Dixon-Coles and Elo models on a historical results DataFrame.
    """
    dc_params = dc_params or {}
    elo_params = elo_params or {}

    # ------------------------------------------------------------------
    # DC model: keep only params that DixonColesModel actually supports
    # ------------------------------------------------------------------
    valid_dc_keys = {"rho_init", "home_adv_init"}  # filter to supported args
    filtered_dc = {k: v for k, v in dc_params.items() if k in valid_dc_keys}

    dc_model = DixonColesModel(**filtered_dc)
    dc_model.fit(results_df)

    # ------------------------------------------------------------------
    # Elo model: usually only supports k_factor
    # ------------------------------------------------------------------
    valid_elo_keys = {"k_factor"}
    filtered_elo = {k: v for k, v in elo_params.items() if k in valid_elo_keys}

    elo_model = EloModel(**filtered_elo)
    elo_model.fit(results_df)

    return dc_model, elo_model


# =====================================================================
# PREDICT FIXTURES (UPDATED WITH CONTEXT-AWARE DRAW CALIBRATION)
# =====================================================================

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

    fixtures_df must contain:
        Date, HomeTeam, AwayTeam
    """

    rows = []

    for _, row in fixtures_df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # ------------------------------------------------------------
        # 1. DC expected goals + probabilities
        # ------------------------------------------------------------
        try:
            lamH, lamA = dc_model.predict_expected_goals(home, away)
            pH_dc, pD_dc, pA_dc = dc_model.predict(home, away)
        except Exception:
            continue

        # ------------------------------------------------------------
        # 2. Elo probabilities
        # ------------------------------------------------------------
        try:
            pH_elo, pD_elo, pA_elo = elo_model.predict(home, away)
        except Exception:
            continue

        # ------------------------------------------------------------
        # 3. Raw Ensemble (same logic you already use)
        # ------------------------------------------------------------
        pH, pD, pA = ensemble_win_probs(
            (pH_dc, pD_dc, pA_dc),
            (pH_elo, pD_elo, pA_elo),
            w_dc=w_dc,
            w_elo=w_elo,
        )

        # =====================================================================
        # 4. CONTEXT-AWARE DRAW CALIBRATION
        # =====================================================================

        # STEP 1 — Mild global correction
        pD *= 1.20  # +20% baseline boost

        # Compute context features
        goal_diff = abs(lamH - lamA)
        total_goals = lamH + lamA

        # ------------------------------------------------------------
        # NEW STEP — reduce overconfidence in close matchups
        # ------------------------------------------------------------
        if goal_diff < 0.40:
            pH *= 0.93
            pA *= 0.93

        if total_goals < 2.40:
            pH *= 0.95
            pA *= 0.95


        # STEP 2 — Close match: more draws
        if goal_diff < 0.40:
            pD *= 1.28  # was +35%

        # STEP 3 — Low scoring match: more draws
        if total_goals < 2.40:
            pD *= 1.20   # was +15%

        # STEP 4 — Ultra-close AND ultra-low
        if goal_diff < 0.25 and total_goals < 2.20:
            pD *= 1.15   # +15%

        # STEP 5 — Blend with historical EPL draw rate
        epl_draw_rate = 0.24
        blend_weight = 0.10
        pD = (1 - blend_weight) * pD + blend_weight * epl_draw_rate

        # STEP 6 — Renormalize to sum to 1
        Z = pH + pD + pA
        if Z > 0:
            pH /= Z
            pD /= Z
            pA /= Z
        else:
            pH = pD = pA = 1 / 3

        # ------------------------------------------------------------
        # 5. Store results
        # ------------------------------------------------------------
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
