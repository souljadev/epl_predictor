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

    def __init__(self, k_factor=20, home_advantage=55):
        self.k = k_factor              # Update rate
        self.home_adv = home_advantage # Elo home advantage in points

    # ----------------------------------------------------
    # REQUIRED BY backtest_fast()
    # ----------------------------------------------------
    def init_ratings(self, teams, initial_rating=1500):
        """
        Initialize ratings for all teams once at the start.
        """
        return {team: initial_rating for team in teams}

    # ----------------------------------------------------
    def expected_result(self, Ra, Rb, is_home):
        """
        Expected result using Elo logistic curve.
        """
        ha = self.home_adv if is_home else 0   # Home advantage in Elo points
        return 1 / (1 + 10 ** (-(Ra + ha - Rb) / 400))

    # ----------------------------------------------------
    def predict_win_probs_raw(self, ratings, home, away):
        """
        Returns win/draw/loss probabilities using Elo goal expectation scaling.
        """
        Ra = ratings[home]
        Rb = ratings[away]

        p_home = self.expected_result(Ra, Rb, is_home=True)
        p_away = self.expected_result(Rb, Ra, is_home=False)

        # Draw probability heuristic
        p_draw = max(0, 1 - (p_home + p_away))
        p_draw = min(p_draw, 0.35)  # usually ~20–30%

        return {"H": p_home, "D": p_draw, "A": p_away}

    # ----------------------------------------------------
    # REQUIRED BY backtest_fast()
    # ----------------------------------------------------
    def update_ratings_match(self, ratings, home, away, FTHG, FTAG):
        """
        Update Elo ratings after seeing the actual match result.
        """

        Ra = ratings[home]
        Rb = ratings[away]

        # Actual result
        if FTHG > FTAG:
            Sa, Sb = 1, 0
        elif FTHG < FTAG:
            Sa, Sb = 0, 1
        else:
            Sa, Sb = 0.5, 0.5

        # Expected results
        Ea = self.expected_result(Ra, Rb, is_home=True)
        Eb = 1 - Ea

        # Elo updates
        ratings[home] = Ra + self.k * (Sa - Ea)
        ratings[away] = Rb + self.k * (Sb - Eb)

        return ratings