import pandas as pd
import numpy as np

class EloModel:
    def __init__(self, k_factor=18.0, home_advantage=55.0):
        self.k = k_factor
        self.home_adv = home_advantage
        self.ratings = {}

    def _get(self, team):
        return self.ratings.get(team, 1500.0)

    def fit(self, results_df: pd.DataFrame):
        teams = sorted(set(results_df["HomeTeam"]).union(results_df["AwayTeam"]))
        for t in teams:
            self.ratings.setdefault(t, 1500.0)
        for _, row in results_df.sort_values("Date").iterrows():
            h, a = row["HomeTeam"], row["AwayTeam"]
            gh, ga = row["FTHG"], row["FTAG"]
            ra_h = self._get(h) + self.home_adv
            ra_a = self._get(a)
            exp_h = 1/(1+10**((ra_a - ra_h)/400))
            if gh>ga: s_h = 1.0
            elif gh==ga: s_h = 0.5
            else: s_h = 0.0
            self.ratings[h] = self._get(h) + self.k*(s_h - exp_h)
            self.ratings[a] = self._get(a) + self.k*((1-s_h) - (1-exp_h))
        return self

    def predict_win_probs(self, home, away):
        ra_h = self._get(home) + self.home_adv
        ra_a = self._get(away)
        p_home = 1/(1+10**((ra_a - ra_h)/400))
        p_draw = 0.24
        p_away = max(0.0, 1 - p_home - p_draw)
        p_home = np.clip(p_home, 0, 1)
        Z = p_home + p_draw + p_away
        return p_home/Z, p_draw/Z, p_away/Z
