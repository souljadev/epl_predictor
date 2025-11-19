import logging
from pathlib import Path
import math
from collections import defaultdict

import pandas as pd
import yaml


class Orchestrator:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Always use combined EPL dataset unless overridden in config
        data_cfg = self.config.get("data", {})
        self.results_path = Path(data_cfg.get("results_csv", "data/raw/epl_combined.csv"))

        model_cfg = self.config.get("model", {})
        self.elo_cfg = model_cfg.get("elo", {})
        self.ensemble_cfg = model_cfg.get("ensemble", {"w_dc": 0.6, "w_elo": 0.4})

        # Internal state
        self.results_df = None
        self.league_home_goals = None
        self.league_away_goals = None
        self.draw_rate = None
        self.attack_home = {}
        self.defence_home = {}
        self.attack_away = {}
        self.defence_away = {}
        self.elo_ratings = {}

        logging.info(f"Loaded config from {self.config_path}")

        # Train models
        self._load_results()
        self._fit_poisson_model()
        self._fit_elo_model()

    # ======================================================================
    # CONFIG + DATA LOADING
    # ======================================================================
    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_results(self):
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results CSV not found: {self.results_path}")

        logging.info(f"Loading historical results from {self.results_path}")
        df = pd.read_csv(self.results_path)

        # Standardize column names
        rename_map = {
            "home_goals": "FTHG",
            "away_goals": "FTAG",
            "Home": "HomeTeam",
            "Away": "AwayTeam",
            "HG": "FTHG",
            "AG": "FTAG",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Validate essential columns
        required = {"HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in results CSV: {missing}")

        # --------------------------------------------------------------
        # 🔥 CORE FIX: remove rows with missing goal data
        # --------------------------------------------------------------
        before = len(df)
        df = df.dropna(subset=["FTHG", "FTAG"])
        after = len(df)
        if before != after:
            logging.warning(f"Dropped {before - after} matches missing goal data.")

        # Ensure numeric types
        df["FTHG"] = pd.to_numeric(df["FTHG"], errors="coerce")
        df["FTAG"] = pd.to_numeric(df["FTAG"], errors="coerce")
        df = df.dropna(subset=["FTHG", "FTAG"])

        # Sort by date if available
        if "Date" in df.columns:
            try:
                df["Date"] = pd.to_datetime(df["Date"])
            except:
                pass
            df = df.sort_values("Date")

        self.results_df = df.reset_index(drop=True)
        logging.info(f"Loaded {len(self.results_df)} cleaned matches.")

    # ======================================================================
    # POISSON / DIXON-COLES STYLE MODEL
    # ======================================================================
    def _fit_poisson_model(self):
        df = self.results_df

        # League averages
        self.league_home_goals = df["FTHG"].mean()
        self.league_away_goals = df["FTAG"].mean()
        self.draw_rate = (df["FTHG"] == df["FTAG"]).mean()

        teams = pd.unique(df[["HomeTeam", "AwayTeam"]].values.ravel())

        home_scored = defaultdict(float)
        home_conceded = defaultdict(float)
        home_games = defaultdict(int)

        away_scored = defaultdict(float)
        away_conceded = defaultdict(float)
        away_games = defaultdict(int)

        for _, r in df.iterrows():
            h, a, hg, ag = r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"]

            home_scored[h] += hg
            home_conceded[h] += ag
            home_games[h] += 1

            away_scored[a] += ag
            away_conceded[a] += hg
            away_games[a] += 1

        # Compute attack & defense ratings
        for t in teams:
            if home_games[t] > 0:
                self.attack_home[t] = (home_scored[t] / home_games[t]) / self.league_home_goals
                self.defence_home[t] = (home_conceded[t] / home_games[t]) / self.league_away_goals
            else:
                self.attack_home[t] = 1.0
                self.defence_home[t] = 1.0

            if away_games[t] > 0:
                self.attack_away[t] = (away_scored[t] / away_games[t]) / self.league_away_goals
                self.defence_away[t] = (away_conceded[t] / away_games[t]) / self.league_home_goals
            else:
                self.attack_away[t] = 1.0
                self.defence_away[t] = 1.0

        # --------------------------------------------------------------
        # NaN cleanup (critical)
        # --------------------------------------------------------------
        for d in [self.attack_home, self.defence_home, self.attack_away, self.defence_away]:
            for tm in d:
                if pd.isna(d[tm]) or math.isinf(d[tm]):
                    d[tm] = 1.0

        logging.info("Fitted Poisson attack/defense strengths.")

    @staticmethod
    def _poisson(k, lam):
        return math.exp(-lam) * lam**k / math.factorial(k)

    def _poisson_match_probs(self, home, away, max_goals=6):
        λ_home = self.league_home_goals * self.attack_home.get(home, 1) * self.defence_away.get(away, 1)
        λ_away = self.league_away_goals * self.attack_away.get(away, 1) * self.defence_home.get(home, 1)

        pH = pD = pA = 0

        for hg in range(max_goals + 1):
            for ag in range(max_goals + 1):
                p = self._poisson(hg, λ_home) * self._poisson(ag, λ_away)
                if hg > ag:
                    pH += p
                elif hg == ag:
                    pD += p
                else:
                    pA += p

        total = pH + pD + pA
        return pH / total, pD / total, pA / total

    # ======================================================================
    # ELO MODEL
    # ======================================================================
    def _fit_elo_model(self):
        df = self.results_df
        k = float(self.elo_cfg.get("k_factor", 18.0))
        home_adv = float(self.elo_cfg.get("home_advantage", 55.0))

        teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
        ratings = {t: 1500 for t in teams}

        for _, r in df.iterrows():
            h, a, hg, ag = r["HomeTeam"], r["AwayTeam"], r["FTHG"], r["FTAG"]
            rh, ra = ratings[h], ratings[a]

            diff = (rh + home_adv) - ra
            exp_home = 1 / (1 + 10 ** (-diff / 400))

            if hg > ag:
                s_home = 1
            elif hg == ag:
                s_home = 0.5
            else:
                s_home = 0

            ratings[h] = rh + k * (s_home - exp_home)
            ratings[a] = ra + k * ((1 - s_home) - (1 - exp_home))

        self.elo_ratings = ratings
        logging.info("Fitted Elo ratings.")

    def _elo_probs(self, home, away):
        rh = self.elo_ratings.get(home, 1500)
        ra = self.elo_ratings.get(away, 1500)
        home_adv = float(self.elo_cfg.get("home_advantage", 55.0))

        diff = (rh + home_adv) - ra
        p_home_raw = 1 / (1 + 10 ** (-diff / 400))
        p_away_raw = 1 - p_home_raw

        p_draw = self.draw_rate if self.draw_rate else 0.25

        scale = 1 - p_draw
        p_home = p_home_raw * scale
        p_away = p_away_raw * scale

        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    # ======================================================================
    # TRAIN
    # ======================================================================
    def train(self):
        logging.info("Re-training models...")
        self._load_results()
        self._fit_poisson_model()
        self._fit_elo_model()
        logging.info("Training complete.")

    # ======================================================================
    # PREDICT FIXTURES
    # ======================================================================
    def run_predictions(self, fixtures_path):
        fixtures_path = Path(fixtures_path)

        if not fixtures_path.exists():
            raise FileNotFoundError(f"Fixtures file not found: {fixtures_path}")

        df = pd.read_csv(fixtures_path)

        missing = {"Date", "HomeTeam", "AwayTeam"} - set(df.columns)
        if missing:
            raise ValueError(f"Fixtures file missing columns: {missing}")

        w_dc = float(self.ensemble_cfg.get("w_dc", 0.6))
        w_elo = float(self.ensemble_cfg.get("w_elo", 0.4))

        home_probs = []
        draw_probs = []
        away_probs = []

        for _, r in df.iterrows():
            home, away = r["HomeTeam"], r["AwayTeam"]

            pH_dc, pD_dc, pA_dc = self._poisson_match_probs(home, away)
            pH_elo, pD_elo, pA_elo = self._elo_probs(home, away)

            pH = w_dc * pH_dc + w_elo * pH_elo
            pD = w_dc * pD_dc + w_elo * pD_elo
            pA = w_dc * pA_dc + w_elo * pA_elo

            total = pH + pD + pA
            home_probs.append(pH / total)
            draw_probs.append(pD / total)
            away_probs.append(pA / total)

        df["home_win_prob"] = home_probs
        df["draw_prob"] = draw_probs
        df["away_win_prob"] = away_probs

        output_dir = Path(self.config.get("orchestrator", {}).get("output_dir", "models/predictions"))
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "predictions.csv"

        df.to_csv(out_path, index=False)
        logging.info(f"Predictions saved to {out_path}")
