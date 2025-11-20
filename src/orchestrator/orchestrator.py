import logging
from pathlib import Path
import math
from collections import defaultdict

import pandas as pd
import yaml


class Orchestrator:
    """
    Orchestrator for training Poisson + Elo models and generating
    full betting markets + exact score predictions.
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
        self.elo_cfg = model_cfg.get("elo", {})
        self.ensemble_cfg = model_cfg.get("ensemble", {"w_dc": 0.6, "w_elo": 0.4})

        # Model state
        self.results_df: pd.DataFrame | None = None
        self.league_home_goals: float | None = None
        self.league_away_goals: float | None = None
        self.draw_rate: float | None = None
        self.attack_home: dict[str, float] = {}
        self.defence_home: dict[str, float] = {}
        self.attack_away: dict[str, float] = {}
        self.defence_away: dict[str, float] = {}
        self.elo_ratings: dict[str, float] = {}

        logging.info(f"Loaded config from {self.config_path}")

        # Initial training
        self._load_results()
        self._fit_poisson_model()
        self._fit_elo_model()

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
        logging.info(f"Loaded {len(self.results_df)} cleaned matches.")

    # =====================================================================
    # POISSON / DIXON-COLES STYLE MODEL
    # =====================================================================
    def _fit_poisson_model(self) -> None:
        df = self.results_df
        assert df is not None

        # League averages
        self.league_home_goals = float(df["FTHG"].mean())
        self.league_away_goals = float(df["FTAG"].mean())
        self.draw_rate = float((df["FTHG"] == df["FTAG"]).mean())

        teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())

        home_scored = defaultdict(float)
        home_conceded = defaultdict(float)
        home_games = defaultdict(int)

        away_scored = defaultdict(float)
        away_conceded = defaultdict(float)
        away_games = defaultdict(int)

        for _, r in df.iterrows():
            h = r["HomeTeam"]
            a = r["AwayTeam"]
            hg = float(r["FTHG"])
            ag = float(r["FTAG"])

            home_scored[h] += hg
            home_conceded[h] += ag
            home_games[h] += 1

            away_scored[a] += ag
            away_conceded[a] += hg
            away_games[a] += 1

        # Compute attack/defence strengths
        for t in teams:
            if home_games[t] > 0:
                self.attack_home[t] = (
                    home_scored[t] / home_games[t]
                ) / self.league_home_goals
                self.defence_home[t] = (
                    home_conceded[t] / home_games[t]
                ) / self.league_away_goals
            else:
                self.attack_home[t] = 1.0
                self.defence_home[t] = 1.0

            if away_games[t] > 0:
                self.attack_away[t] = (
                    away_scored[t] / away_games[t]
                ) / self.league_away_goals
                self.defence_away[t] = (
                    away_conceded[t] / away_games[t]
                ) / self.league_home_goals
            else:
                self.attack_away[t] = 1.0
                self.defence_away[t] = 1.0

        # Clean up NaNs / infs
        for d in [
            self.attack_home,
            self.defence_home,
            self.attack_away,
            self.defence_away,
        ]:
            for tm in d:
                if pd.isna(d[tm]) or math.isinf(d[tm]):
                    d[tm] = 1.0

        logging.info("Fitted Poisson attack/defense strengths.")

    @staticmethod
    def _poisson(k: int, lam: float) -> float:
        return math.exp(-lam) * (lam**k) / math.factorial(k)

    def _expected_goals(self, home: str, away: str) -> tuple[float, float]:
        """Return expected goals (lambda_home, lambda_away) from Poisson model."""
        λH = (
            self.league_home_goals
            * self.attack_home.get(home, 1.0)
            * self.defence_away.get(away, 1.0)
        )
        λA = (
            self.league_away_goals
            * self.attack_away.get(away, 1.0)
            * self.defence_home.get(home, 1.0)
        )
        return float(λH), float(λA)

    def _exact_score_matrix(
        self, home: str, away: str, max_goals: int = 6
    ) -> dict[tuple[int, int], float]:
        """
        Return dict[(home_goals, away_goals)] -> probability,
        normalized to sum to 1.
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
            # Fallback: uniform tiny probabilities if everything broke
            n = (max_goals + 1) ** 2
            matrix = {(hg, ag): 1.0 / n for hg in range(max_goals + 1) for ag in range(max_goals + 1)}

        return matrix

    def _poisson_match_probs(
        self, home: str, away: str, max_goals: int = 6
    ) -> tuple[float, float, float]:
        """
        Aggregate Poisson exact-score matrix into 1X2 probabilities.
        """
        λH, λA = self._expected_goals(home, away)

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p = self._poisson(hg, λH) * self._poisson(ag, λA)
                if hg > ag:
                    p_home += p
                elif hg == ag:
                    p_draw += p
                else:
                    p_away += p

        total = p_home + p_draw + p_away
        if total > 0:
            return p_home / total, p_draw / total, p_away / total
        # Fallback equal split if something degenerate happens
        return 1.0 / 3, 1.0 / 3, 1.0 / 3

    # =====================================================================
    # ELO MODEL
    # =====================================================================
    def _fit_elo_model(self) -> None:
        df = self.results_df
        assert df is not None

        k_factor = float(self.elo_cfg.get("k_factor", 18.0))
        home_adv = float(self.elo_cfg.get("home_advantage", 55.0))

        teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())
        ratings = {t: 1500.0 for t in teams}

        for _, r in df.iterrows():
            h = r["HomeTeam"]
            a = r["AwayTeam"]
            hg = float(r["FTHG"])
            ag = float(r["FTAG"])

            rh = ratings[h]
            ra = ratings[a]

            diff = (rh + home_adv) - ra
            exp_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

            if hg > ag:
                s_home = 1.0
            elif hg == ag:
                s_home = 0.5
            else:
                s_home = 0.0

            ratings[h] = rh + k_factor * (s_home - exp_home)
            ratings[a] = ra + k_factor * ((1.0 - s_home) - (1.0 - exp_home))

        self.elo_ratings = ratings
        logging.info("Fitted Elo ratings.")

    def _elo_probs(self, home: str, away: str) -> tuple[float, float, float]:
        rh = self.elo_ratings.get(home, 1500.0)
        ra = self.elo_ratings.get(away, 1500.0)
        home_adv = float(self.elo_cfg.get("home_advantage", 55.0))

        diff = (rh + home_adv) - ra
        p_home_raw = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        p_away_raw = 1.0 - p_home_raw

        draw_rate = self.draw_rate if self.draw_rate is not None else 0.25
        scale = 1.0 - draw_rate

        p_home = p_home_raw * scale
        p_away = p_away_raw * scale
        p_draw = draw_rate

        total = p_home + p_draw + p_away
        if total > 0:
            return p_home / total, p_draw / total, p_away / total
        return 1.0 / 3, 1.0 / 3, 1.0 / 3

    # =====================================================================
    # PUBLIC API: TRAIN + PREDICT
    # =====================================================================
    def train(self) -> None:
        """
        Re-train models from historical data.
        """
        logging.info("Re-training Poisson + Elo models...")
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

            # Poisson 1X2 probabilities
            pH_dc, pD_dc, pA_dc = self._poisson_match_probs(home, away)

            # Elo 1X2 probabilities
            pH_elo, pD_elo, pA_elo = self._elo_probs(home, away)

            # Ensemble
            pH = w_dc * pH_dc + w_elo * pH_elo
            pD = w_dc * pD_dc + w_elo * pD_elo
            pA = w_dc * pA_dc + w_elo * pA_elo

            total = pH + pD + pA
            if total > 0:
                pH /= total
                pD /= total
                pA /= total
            else:
                pH = pD = pA = 1.0 / 3

            # Exact score matrix (from Poisson)
            matrix = self._exact_score_matrix(home, away, max_goals=6)

            # Most likely exact score
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

            # Expected goals from Poisson
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

            # All exact scores
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
        out_dir = Path(
            self.config.get("orchestrator", {}).get(
                "output_dir", "models/predictions"
            )
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        full_df = pd.DataFrame(full_rows)
        exact_df = pd.DataFrame(exact_rows)

        full_path = out_dir / "predictions_full.csv"
        exact_path = out_dir / "predictions_exact_scores.csv"

        full_df.to_csv(full_path, index=False)
        exact_df.to_csv(exact_path, index=False)

        logging.info(f"Saved full markets to {full_path}")
        logging.info(f"Saved exact score matrix to {exact_path}")
