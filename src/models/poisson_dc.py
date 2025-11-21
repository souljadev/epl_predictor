import numpy as np
import pandas as pd
from math import lgamma
from typing import Dict, Tuple


class DixonColes:
    """
    Dixon–Coles style Poisson model with ridge regularization
    on attack/defense strengths.

    API:
    - .fit(df) where df has columns: HomeTeam, AwayTeam, FTHG, FTAG
    - .match_probs(home, away) → {
          "win_probs": {"H": pH, "D": pD, "A": pA},
          "lambdas": (lam_home, lam_away)
      }
    """

    def __init__(
        self,
        rho_init: float = 0.0,
        max_iter: int = 300,
        tol: float = 1e-6,
        lambda_reg: float = 0.1,
        max_goals: int = 10,
    ):
        """
        Parameters
        ----------
        rho_init : float
            Initial Dixon–Coles correlation parameter.
        max_iter : int
            Max iterations for the optimizer (if used).
        tol : float
            Tolerance for convergence (not used in this simple GD version, but kept for compatibility).
        lambda_reg : float
            Ridge strength for attack/defense parameters.
        max_goals : int
            Truncation for Poisson goal sums when computing probabilities.
        """
        self.rho = rho_init
        self.max_iter = max_iter
        self.tol = tol
        self.lambda_reg = lambda_reg
        self.max_goals = max_goals

        self.teams: list[str] = []
        self.team_index: Dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_adv: float = 0.0

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _init_params(self, n_teams: int):
        # Small random init near 0
        self.attack = np.zeros(n_teams)
        self.defense = np.zeros(n_teams)
        self.home_adv = 0.1
        # rho already set in __init__

    @staticmethod
    def _poisson_log_pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """
        Log Poisson pmf using lgamma for numerical stability:
        log P(k | lambda) = k*log(lambda) - lambda - log(k!)
        """
        return k * np.log(lam) - lam - lgamma(k + 1.0)

    @staticmethod
    def _dc_tau(x: int, y: int, rho: float) -> float:
        """
        Dixon-Coles adjustment factor for low scores.

        tau(x, y) = 1 + rho for (0,0)
                  = 1 - rho for (0,1) or (1,0)
                  = 1 + rho for (1,1)
                  = 1 otherwise
        """
        if x == 0 and y == 0:
            return 1.0 + rho
        elif x == 0 and y == 1:
            return 1.0 - rho
        elif x == 1 and y == 0:
            return 1.0 - rho
        elif x == 1 and y == 1:
            return 1.0 + rho
        else:
            return 1.0

    # ------------------------------------------------------------------ #
    # Likelihood
    # ------------------------------------------------------------------ #
    def _match_loglik(self, home_idx, away_idx, FTHG, FTAG) -> float:
        """
        Log-likelihood for a single match with DC adjustment.
        """
        a = self.attack
        d = self.defense

        lam_home = np.exp(self.home_adv + a[home_idx] - d[away_idx])
        lam_away = np.exp(a[away_idx] - d[home_idx])

        # Poisson log pmfs
        log_p_home = self._poisson_log_pmf(FTHG, lam_home)
        log_p_away = self._poisson_log_pmf(FTAG, lam_away)

        tau = self._dc_tau(int(FTHG), int(FTAG), self.rho)
        return float(log_p_home + log_p_away + np.log(tau))

    def _total_loglik(self, df: pd.DataFrame) -> float:
        """
        Total log-likelihood with ridge penalty.
        """
        ll = 0.0
        for _, row in df.iterrows():
            hi = self.team_index[row["HomeTeam"]]
            ai = self.team_index[row["AwayTeam"]]
            FTHG = row["FTHG"]
            FTAG = row["FTAG"]
            ll += self._match_loglik(hi, ai, FTHG, FTAG)

        # Ridge penalty on attack/defense
        penalty = 0.5 * self.lambda_reg * (
            float(np.sum(self.attack ** 2)) + float(np.sum(self.defense ** 2))
        )

        return ll - penalty

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame):
        """
        Fit DC model via simple gradient-free iterative improvement.
        This is not as fast as a full optimizer but avoids extra deps.
        """
        teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
        self.teams = teams
        self.team_index = {t: i for i, t in enumerate(teams)}
        n_teams = len(teams)

        self._init_params(n_teams)

        # Basic coordinate-descent style updates (coarse but robust)
        current_ll = self._total_loglik(df)

        for it in range(self.max_iter):
            improved = False

            # Small random perturbations to parameters
            for arr_name in ["attack", "defense"]:
                arr = getattr(self, arr_name)
                for i in range(n_teams):
                    old_val = arr[i]

                    for delta in (+0.01, -0.01):
                        arr[i] = old_val + delta
                        new_ll = self._total_loglik(df)
                        if new_ll > current_ll + self.tol:
                            current_ll = new_ll
                            improved = True
                            old_val = arr[i]
                        else:
                            arr[i] = old_val

            # Home advantage tweak
            for delta in (+0.01, -0.01):
                old_val = self.home_adv
                self.home_adv = old_val + delta
                new_ll = self._total_loglik(df)
                if new_ll > current_ll + self.tol:
                    current_ll = new_ll
                    improved = True
                    old_val = self.home_adv
                else:
                    self.home_adv = old_val

            # Rho tweak
            for delta in (+0.01, -0.01):
                old_val = self.rho
                self.rho = old_val + delta
                new_ll = self._total_loglik(df)
                if new_ll > current_ll + self.tol:
                    current_ll = new_ll
                    improved = True
                    old_val = self.rho
                else:
                    self.rho = old_val

            if not improved:
                break

        # Center attack/defense around zero (identifiability)
        self.attack -= np.mean(self.attack)
        self.defense -= np.mean(self.defense)

        return self

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    def _expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        hi = self.team_index[home]
        ai = self.team_index[away]
        lam_home = float(np.exp(self.home_adv + self.attack[hi] - self.defense[ai]))
        lam_away = float(np.exp(self.attack[ai] - self.defense[hi]))
        return lam_home, lam_away

    def _score_matrix(self, lam_home: float, lam_away: float):
        """
        Build score probability matrix with DC correlation, truncated at max_goals.
        Returns a (max_goals+1, max_goals+1) matrix.
        """
        max_g = self.max_goals
        mat = np.zeros((max_g + 1, max_g + 1), dtype=float)

        # Precompute Poisson log pmfs
        k = np.arange(max_g + 1)
        log_p_home = self._poisson_log_pmf(k[:, None], lam_home)
        log_p_away = self._poisson_log_pmf(k[None, :], lam_away)

        for i in range(max_g + 1):
            for j in range(max_g + 1):
                tau = self._dc_tau(i, j, self.rho)
                mat[i, j] = np.exp(log_p_home[i, 0] + log_p_away[0, j]) * tau

        # Normalize just in case
        mat /= mat.sum()
        return mat

    def match_probs(self, home: str, away: str) -> dict:
        """
        Return:
        {
          "win_probs": {"H": pH, "D": pD, "A": pA},
          "lambdas": (lam_home, lam_away)
        }
        """
        lam_home, lam_away = self._expected_goals(home, away)
        score_mat = self._score_matrix(lam_home, lam_away)

        # H/D/A probabilities
        pH = float(np.tril(score_mat, -1).sum())  # home goals > away goals
        pA = float(np.triu(score_mat, 1).sum())   # away goals > home goals
        pD = float(np.trace(score_mat))           # diagonal

        Z = pH + pD + pA
        if Z <= 0:
            pH = pD = pA = 1.0 / 3.0
        else:
            pH, pD, pA = pH / Z, pD / Z, pA / Z

        return {
            "win_probs": {"H": pH, "D": pD, "A": pA},
            "lambdas": (lam_home, lam_away),
        }
