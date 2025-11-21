import logging
from pathlib import Path
import math
from collections import defaultdict  # kept in case of future use

import pandas as pd
import yaml

from ..models.dixon_coles import DixonColesModel
from ..models.elo import EloModel
from ..models.ensemble import ensemble_win_probs


class Orchestrator:
    """
    Orchestrator for training Dixon–Coles + Elo models and generating
    full betting markets + exact score predictions.

    This version is aligned with the backtest pipelines:
      - Uses xG-enhanced Dixon–Coles for expected goals and 1X2
      - Uses the EloModel class for 1X2 probabilities
      - Ensembles them with the same weights as backtests
    """

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Data paths
        data_cfg = self.config.get("data", {})
        # Default to epl_combined.csv if not overridden
        self.results_path = Path(
            data_cfg.get("results_csv", "data/raw/epl_combined.csv")
        )

        model_cfg = self.config.get("model", {})
        self.dc_cfg = model_cfg.get("dc", {})
        self.elo_cfg = model_cfg.get("elo", {})
        self.ensemble_cfg = model_cfg.get("ensemble", {"w_dc": 0.6, "w_elo": 0.4})

        orch_cfg = self.config.get("orchestrator", {})
        self.output_dir = Path(orch_cfg.get("output_dir", "models/predictions"))
        # Train start date: default to xG era (2018-01-01) if not provided
        self.train_start_date_str = orch_cfg.get("train_start_date", "2018-01-01")

        # Model state
        self.results_df: pd.DataFrame | None = None
        self.dc_model: DixonColesModel | None = None
        self.elo_model: EloModel | None = None

        logging.info(f"Loaded config from {self.config_path}")

        # Initial training
        self._load_results()
        self._fit_poisson_model()  # now fits Dixon–Coles
        self._fit_elo_model()      # fits EloModel

    # =====================================================================
    # CONFIG + DATA LOADING
    # =====================================================================
    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_results(self) -> None:
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {self.results_path}")

        logging.info(f"Loading historical results from {self.results_path}")
        df = pd.read_csv(self.results_path)

        # Standardize column names if needed
        rename_map = {
            "home_goals": "FTHG",
            "away_goals": "FTAG",
            "Home": "HomeTeam",
            "Away": "AwayTeam",
            "HG": "FTHG",
            "AG": "FTAG",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        required_cols = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Results CSV missing columns: {missing}")

        # Drop rows with missing goal data
        before = len(df)
        df = df.dropna(subset=["FTHG", "FTAG"])
        df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].apply(
            pd.to_numeric, errors="coerce"
        )
        df = df.dropna(subset=["FTHG", "FTAG"])
        after = len(df)
        if before != after:
            logging.warning(f"Dropped {before - after} matches missing goal data.")

        # Sort by date if available
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
                df = df.sort_values("Date")
            except Exception:
                pass

        self.results_df = df.reset_index(drop=True)
        logging.info(f"Loaded {len(self.results_df)} historical matches.")

    # =====================================================================
    # DIXON–COLES MODEL (re-using legacy name _fit_poisson_model)
    # =====================================================================
    def _fit_poisson_model(self) -> None:
        """
        Fit the xG-based Dixon–Coles model on historical data.

        We keep the legacy name `_fit_poisson_model` so that existing
        calls from `train()` and __init__ still work unchanged.
        """
        if self.results_df is None:
            raise RuntimeError("Results dataframe not loaded.")

        df = self.results_df

        # Optional training window aligned with backtests
        if "Date" in df.columns and self.train_start_date_str:
            try:
                cutoff = pd.to_datetime(self.train_start_date_str)
                df = df[df["Date"] >= cutoff].copy()
                logging.info(
                    f"Training Dixon–Coles on matches from {cutoff.date()} onward "
                    f"({len(df)} matches)."
                )
            except Exception as e:
                logging.warning(f"Could not parse train_start_date: {e}")

        # Instantiate and fit Dixon–Coles
        rho_init = float(self.dc_cfg.get("rho_init", 0.0))
        home_adv_init = float(self.dc_cfg.get("home_adv_init", 0.15))
        lr = float(self.dc_cfg.get("lr", 0.05))

        dc = DixonColesModel(
            rho_init=rho_init,
            home_adv_init=home_adv_init,
            lr=lr,
        )

        # Use xG if available, otherwise fall back to goals
        dc.fit(df, use_xg=True, home_xg_col="Home_xG", away_xg_col="Away_xG")

        self.dc_model = dc
        logging.info("Fitted xG-based Dixon–Coles model.")

    # =====================================================================
    # ELO MODEL
    # =====================================================================
    def _fit_elo_model(self) -> None:
        if self.results_df is None:
            raise RuntimeError("Results dataframe not loaded.")

        df = self.results_df

        # Same training window as DC
        if "Date" in df.columns and self.train_start_date_str:
            try:
                cutoff = pd.to_datetime(self.train_start_date_str)
                df = df[df["Date"] >= cutoff].copy()
                logging.info(
                    f"Training Elo on matches from {cutoff.date()} onward "
                    f"({len(df)} matches)."
                )
            except Exception as e:
                logging.warning(f"Could not parse train_start_date for Elo: {e}")

        k_factor = float(self.elo_cfg.get("k_factor", 18.0))
        home_adv = float(self.elo_cfg.get("home_advantage", 55.0))

        elo = EloModel(k_factor=k_factor, home_advantage=home_adv)
        elo.fit(df)

        self.elo_model = elo
        logging.info("Fitted Elo ratings via EloModel.")

    # =====================================================================
    # POISSON UTILITIES (for exact score matrix)
    # =====================================================================
    @staticmethod
    def _poisson(k: int, lam: float) -> float:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)

    def _expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """
        Return expected goals (lambda_home, lambda_away) from the
        xG-trained Dixon–Coles model.
        """
        if self.dc_model is None:
            raise RuntimeError("Dixon–Coles model not fitted.")
        λH, λA = self.dc_model.predict_expected_goals(home, away)
        return float(λH), float(λA)

    def _exact_score_matrix(
        self, home: str, away: str, max_goals: int = 6
    ) -> dict[tuple[int, int], float]:
        """
        Build exact score probability matrix using Poisson with
        λH, λA from Dixon–Coles.
        """
        λH, λA = self._expected_goals(home, away)
        matrix: dict[tuple[int, int], float] = {}

        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p = self._poisson(hg, λH) * self._poisson(ag, λA)
                matrix[(hg, ag)] = p

        total = sum(matrix.values())
        if total > 0:
            for k in matrix:
                matrix[k] /= total
        else:
            # Degenerate fallback: give 0-0 full mass
            matrix = {(0, 0): 1.0}
        return matrix

    def _poisson_match_probs(
        self, home: str, away: str, max_goals: int = 6
    ) -> tuple[float, float, float]:
        """
        Return 1X2 probabilities from the Dixon–Coles model.

        We keep the legacy name `_poisson_match_probs` so that the
        surrounding code doesn't need to change.
        """
        if self.dc_model is None:
            raise RuntimeError("Dixon–Coles model not fitted.")
        pH, pD, pA = self.dc_model.predict(home, away, max_goals=max_goals)
        return float(pH), float(pD), float(pA)

    # =====================================================================
    # ELO PROBS
    # =====================================================================
    def _elo_probs(self, home: str, away: str) -> tuple[float, float, float]:
        if self.elo_model is None:
            raise RuntimeError("Elo model not fitted.")
        pH, pD, pA = self.elo_model.predict(home, away)
        return float(pH), float(pD), float(pA)

    # =====================================================================
    # PUBLIC API: TRAIN + PREDICT
    # =====================================================================
    def train(self) -> None:
        """
        Re-train models from historical data.
        """
        logging.info("Re-training Dixon–Coles + Elo models...")
        self._load_results()
        self._fit_poisson_model()
        self._fit_elo_model()
        logging.info("Training complete.")

    def run_predictions(self, fixtures_path: str) -> None:
        """
        Generate predictions and betting markets for a fixtures CSV.
        Writes:
          - predictions_full.csv
          - predictions_exact_scores.csv
        """
        fixtures_path = Path(fixtures_path)
        if not fixtures_path.exists():
            raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

        df = pd.read_csv(fixtures_path)

        required = {"HomeTeam", "AwayTeam"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Fixtures file missing columns: {missing}")

        w_dc = float(self.ensemble_cfg.get("w_dc", 0.6))
        w_elo = float(self.ensemble_cfg.get("w_elo", 0.4))

        full_rows: list[dict] = []
        exact_rows: list[dict] = []

        for _, r in df.iterrows():
            home = r["HomeTeam"]
            away = r["AwayTeam"]
            date = r.get("Date", "")

            # Dixon–Coles 1X2 probabilities
            pH_dc, pD_dc, pA_dc = self._poisson_match_probs(home, away)

            # Elo 1X2 probabilities
            pH_elo, pD_elo, pA_elo = self._elo_probs(home, away)

            # Ensemble (using shared helper)
            pH, pD, pA = ensemble_win_probs(
                (pH_dc, pD_dc, pA_dc),
                (pH_elo, pD_elo, pA_elo),
                w_dc=w_dc,
                w_elo=w_elo,
            )

            # Exact score matrix from Poisson with DC lambdas
            matrix = self._exact_score_matrix(home, away, max_goals=6)

            # Most likely score
            best_score = max(matrix, key=matrix.get)
            best_prob = matrix[best_score]

            # BTTS
            btts_yes = sum(
                prob for (hg, ag), prob in matrix.items() if hg > 0 and ag > 0
            )
            btts_no = 1.0 - btts_yes

            # Over/Under 2.5
            over_2_5 = sum(
                prob for (hg, ag), prob in matrix.items() if (hg + ag) > 2
            )
            under_2_5 = 1.0 - over_2_5

            # Double Chance
            dc_1x = pH + pD  # home or draw
            dc_x2 = pD + pA  # draw or away
            dc_12 = pH + pA  # either side wins

            # Expected goals
            λH, λA = self._expected_goals(home, away)
            exp_total_goals = λH + λA

            full_rows.append(
                {
                    "Date": date,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "home_win_prob": pH,
                    "draw_prob": pD,
                    "away_win_prob": pA,
                    "most_likely_score": f"{best_score[0]}-{best_score[1]}",
                    "most_likely_score_prob": best_prob,
                    "btts_yes": btts_yes,
                    "btts_no": btts_no,
                    "over_2_5": over_2_5,
                    "under_2_5": under_2_5,
                    "double_chance_1x": dc_1x,
                    "double_chance_x2": dc_x2,
                    "double_chance_12": dc_12,
                    "exp_home_goals": λH,
                    "exp_away_goals": λA,
                    "exp_total_goals": exp_total_goals,
                }
            )

            # Exact scores rows
            for (hg, ag), prob in matrix.items():
                exact_rows.append(
                    {
                        "Date": date,
                        "HomeTeam": home,
                        "AwayTeam": away,
                        "score": f"{hg}-{ag}",
                        "prob": prob,
                    }
                )

        # Output paths
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        full_path = out_dir / "predictions_full.csv"
        exact_path = out_dir / "predictions_exact_scores.csv"

        pd.DataFrame(full_rows).to_csv(full_path, index=False)
        pd.DataFrame(exact_rows).to_csv(exact_path, index=False)

        logging.info(f"Wrote full predictions to: {full_path}")
        logging.info(f"Wrote exact-score predictions to: {exact_path}")
