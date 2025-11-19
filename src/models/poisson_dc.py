import numpy as np
import pandas as pd
from scipy.optimize import minimize
from math import log

class DixonColes:
    def __init__(self, rho_init=0.0, max_iter=300, tol=1e-6, reg=0.1):
        self.rho = rho_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg  # L2 regularization strength
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.1
        self.teams_ = []

    def _init_params(self, teams):
        self.teams_ = teams
        self.attack = {t: 0.01 for t in teams}
        self.defense = {t: -0.01 for t in teams}
        self.home_adv = 0.1
        self.rho = 0.0

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - (lam * mu * rho)
        elif x == 0 and y == 1:
            return 1 + (lam * rho)
        elif x == 1 and y == 0:
            return 1 + (mu * rho)
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _log_likelihood(self, params, matches):
        n = len(self.teams_)

        atk = dict(zip(self.teams_, params[:n]))
        dfn = dict(zip(self.teams_, params[n:2*n]))
        home_adv = params[-2]
        rho = params[-1]

        atk_mean = np.mean(list(atk.values()))
        for k in atk:
            atk[k] -= atk_mean

        ll = 0.0
        reg_loss = 0.0

        from math import factorial, exp

        # L2 Regularization on attack/defense ratings
        for t in atk:
            reg_loss += self.reg * (atk[t] ** 2 + dfn[t] ** 2)

        for _, row in matches.iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            x, y = int(row["FTHG"]), int(row["FTAG"])

            lam = np.exp(atk[h] - dfn[a] + home_adv)
            mu = np.exp(atk[a] - dfn[h])

            # prevent exploding Poisson values
            lam = np.clip(lam, 1e-6, 6.0)
            mu = np.clip(mu, 1e-6, 6.0)

            px = -lam + x*np.log(lam) - log(factorial(x))
            py = -mu + y*np.log(mu) - log(factorial(y))

            tau = self._tau(x, y, lam, mu, rho)
            if tau <= 0:
                tau = 1e-6

            ll += (px + py + np.log(tau))

        return -(ll - reg_loss)

    def fit(self, results_df: pd.DataFrame):
        teams = sorted(set(results_df["HomeTeam"]).union(results_df["AwayTeam"]))
        self._init_params(teams)

        n = len(teams)
        x0 = np.zeros(2*n + 2)
        x0[:n] = 0.01
        x0[n:2*n] = -0.01
        x0[-2] = 0.1
        x0[-1] = self.rho

        res = minimize(
            self._log_likelihood,
            x0,
            args=(results_df,),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter}
        )

        params = res.x
        self.attack = dict(zip(self.teams_, params[:n]))
        self.defense = dict(zip(self.teams_, params[n:2*n]))
        self.home_adv = params[-2]
        self.rho = params[-1]

        return self

    def predict_score_mean(self, home, away):
        lam = np.exp(self.attack[home] - self.defense[away] + self.home_adv)
        mu = np.exp(self.attack[away] - self.defense[home])

        # Clip exploding values
        return np.clip(lam, 0.01, 5.0), np.clip(mu, 0.01, 5.0)

    def match_probs(self, home, away, max_goals=6):
        lam, mu = self.predict_score_mean(home, away)
        from math import factorial, exp
        probs = {}
        pH = pD = pA = 0.0

        for x in range(0, max_goals+1):
            for y in range(0, max_goals+1):
                px = exp(-lam) * (lam**x) / factorial(x)
                py = exp(-mu) * (mu**y) / factorial(y)

                tau = self._tau(x, y, lam, mu, self.rho)
                tau = max(tau, 1e-6)

                p = px * py * tau
                probs[(x, y)] = p

                if x > y: pH += p
                elif x == y: pD += p
                else: pA += p

        Z = sum(probs.values())
        pH, pD, pA = pH/Z, pD/Z, pA/Z

        return {
            "score_probs": probs,
            "win_probs": (pH, pD, pA),
            "lambdas": (lam, mu),
        }
