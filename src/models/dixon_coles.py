import numpy as np
from math import exp


class DixonColesModel:
    """
    Incremental Dixon–Coles style model.

    Maintains:
      - team attack strengths
      - team defense strengths
      - home advantage
      - rho (low-scoring correlation factor)

    Supports:
      - update(home, away, FTHG, FTAG)
      - predict_expected_goals(home, away) -> (lam_home, lam_away)
      - predict(home, away) -> (pH, pD, pA)
      - fit(df)  # for expanding backtest
      - match_probs(home, away) -> {"lambdas": (lamH, lamA), "win_probs": {"H","D","A"}}
    """

    def __init__(self, rho_init=0.0, home_adv_init=0.15, lr=0.05):
        self.attack = {}
        self.defense = {}
        self.rho = rho_init
        self.home_adv = home_adv_init
        self.lr = lr

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _init_team(self, team):
        if team not in self.attack:
            self.attack[team] = 0.0
        if team not in self.defense:
            self.defense[team] = 0.0

    @staticmethod
    def _rho_adjust(home_goals, away_goals, lam_h, lam_a, rho):
        """
        Standard Dixon–Coles rho adjustment for low scores.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - (lam_h * lam_a * rho)
        if home_goals == 0 and away_goals == 1:
            return 1 + (lam_h * rho)
        if home_goals == 1 and away_goals == 0:
            return 1 + (lam_a * rho)
        if home_goals == 1 and away_goals == 1:
            return 1 - rho
        return 1.0

    @staticmethod
    def _poisson_p(k, lam):
        if lam <= 0:
            return 1.0 if k == 0 else 0.0
        return exp(-lam) * lam**k / np.math.factorial(k)

    # ------------------------------
    # Core incremental update
    # ------------------------------
    def update(self, home, away, home_goals, away_goals):
        """
        Single-match incremental update using a simple gradient-like step.
        """
        self._init_team(home)
        self._init_team(away)

        lam_h = exp(self.attack[home] - self.defense[away] + self.home_adv)
        lam_a = exp(self.attack[away] - self.defense[home])

        # Gradients (log-likelihood style)
        grad_attack_home = home_goals - lam_h
        grad_defense_away = lam_h - home_goals

        grad_attack_away = away_goals - lam_a
        grad_defense_home = lam_a - away_goals

        # Update parameters
        self.attack[home] += self.lr * grad_attack_home
        self.defense[away] += self.lr * grad_defense_away

        self.attack[away] += self.lr * grad_attack_away
        self.defense[home] += self.lr * grad_defense_home

        # Light rho update based on low-scoring correlation
        if home_goals <= 1 and away_goals <= 1:
            target_rho = 1.0 if home_goals == away_goals else -1.0
            self.rho += 0.01 * (target_rho - self.rho)

    # ------------------------------
    # Fit over a dataframe (expanding backtest)
    # ------------------------------
    def fit(self, df):
        """
        Fit the model over a historical dataframe.

        Expects columns: HomeTeam, AwayTeam, FTHG, FTAG
        """
        # Sort by date if Date exists
        if "Date" in df.columns:
            df = df.sort_values("Date")

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            FTHG = int(row["FTHG"])
            FTAG = int(row["FTAG"])
            self.update(home, away, FTHG, FTAG)

        return self

    # ------------------------------
    # Prediction utilities
    # ------------------------------
    def predict_expected_goals(self, home, away):
        """Return (λ_home, λ_away)."""
        self._init_team(home)
        self._init_team(away)

        lam_home = exp(self.attack[home] - self.defense[away] + self.home_adv)
        lam_away = exp(self.attack[away] - self.defense[home])

        return lam_home, lam_away

    def predict(self, home, away, max_goals=6):
        """
        Return (pH, pD, pA) using Poisson + DC rho adjustment.
        """
        lam_h, lam_a = self.predict_expected_goals(home, away)

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p_ij = (
                    self._poisson_p(i, lam_h)
                    * self._poisson_p(j, lam_a)
                    * self._rho_adjust(i, j, lam_h, lam_a, self.rho)
                )

                if i > j:
                    p_home += p_ij
                elif i == j:
                    p_draw += p_ij
                else:
                    p_away += p_ij

        Z = p_home + p_draw + p_away
        if Z == 0:
            return 0.33, 0.34, 0.33

        return p_home / Z, p_draw / Z, p_away / Z

    # ------------------------------
    # Expanding backtest helper
    # ------------------------------
    def match_probs(self, home, away, max_goals=6):
        """
        Convenience wrapper for expanding backtest:

        Returns:
        {
          "lambdas": (lamH, lamA),
          "win_probs": {"H": pH, "D": pD, "A": pA}
        }
        """
        lamH, lamA = self.predict_expected_goals(home, away)
        pH, pD, pA = self.predict(home, away, max_goals=max_goals)

        return {
            "lambdas": (lamH, lamA),
            "win_probs": {
                "H": pH,
                "D": pD,
                "A": pA,
            },
        }
