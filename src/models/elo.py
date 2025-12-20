import numpy as np


class EloModel:
    """
    Enhanced Elo model for football (soccer).

    Features:
    - Adaptive K: larger updates for surprising results.
    - Margin sensitivity: bigger rating changes for multi-goal wins.
    - Dynamic draw probability: increases for evenly matched teams.
    - Fully compatible with fast incremental training (backtest_fast.py).
    """

    def __init__(
        self,
        k_factor: float = 28.0,
        home_advantage: float = 65.0,
        base_rating: float = 1500.0,
        draw_base: float = 0.25,
        draw_max_extra: float = 0.10,
        draw_scale: float = 400.0,
    ):
        """
        Parameters
        ----------
        k_factor : float
            Base K for Elo updates.
        home_advantage : float
            Home advantage added to the home team's rating.
        base_rating : float
            Default Elo for new teams.
        draw_base : float
            Baseline draw probability.
        draw_max_extra : float
            Maximum extra draw probability when evenly matched.
        draw_scale : float
            Rating-difference scaling factor for draw bump.
        """
        self.k = k_factor
        self.home_adv = home_advantage
        self.base_rating = base_rating

        # Draw model
        self.draw_base = draw_base
        self.draw_max_extra = draw_max_extra
        self.draw_scale = draw_scale

        # Ratings dict populated during updates
        self.ratings: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Rating Utilities
    # ------------------------------------------------------------------ #

    def _get_rating(self, team):
        return self.ratings.get(team, self.base_rating)

    def _ensure_team(self, team):
        if team not in self.ratings:
            self.ratings[team] = float(self.base_rating)

    def expected_home_prob(self, R_home: float, R_away: float) -> float:
        """Expected home win prob in a 2-outcome setup (no draw)."""
        diff = (R_home + self.home_adv) - R_away
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def margin_multiplier(self, goal_diff: int) -> float:
        """Margin-of-victory multiplier."""
        if goal_diff <= 1:
            return 1.0
        mult = 1.0 + (goal_diff - 1) * 0.075
        return min(mult, 1.75)

    # ------------------------------------------------------------------ #
    # Core Rating Update Logic
    # ------------------------------------------------------------------ #

    def update_ratings_match(self, ratings, home, away, FTHG, FTAG):
        """Internal Elo update for a single match."""
        R_home = ratings[home]
        R_away = ratings[away]

        # Expected home win probability
        exp_home = self.expected_home_prob(R_home, R_away)
        exp_away = 1.0 - exp_home

        # Actual match result
        if FTHG > FTAG:
            S_home, S_away = 1.0, 0.0
        elif FTHG < FTAG:
            S_home, S_away = 0.0, 1.0
        else:
            S_home, S_away = 0.5, 0.5

        # Surprise factor
        surprise = abs(S_home - exp_home)

        # Margin sensitivity
        goal_diff = abs(FTHG - FTAG)
        margin_mult = self.margin_multiplier(goal_diff)

        # Effective K
        K_eff = self.k * (1.0 + surprise) * margin_mult

        # New ratings
        ratings[home] = R_home + K_eff * (S_home - exp_home)
        ratings[away] = R_away + K_eff * (S_away - exp_away)

        return ratings

    # ------------------------------------------------------------------ #
    # Fit Entire Dataset (optional)
    # ------------------------------------------------------------------ #

    def fit(self, df):
        """Fit Elo on full historical dataset."""
        teams = sorted(set(df["HomeTeam"]).union(df["AwayTeam"]))
        self.ratings = {t: float(self.base_rating) for t in teams}

        for _, row in df.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]
            FTHG = row["FTHG"]
            FTAG = row["FTAG"]
            self.ratings = self.update_ratings_match(self.ratings, home, away, FTHG, FTAG)

        return self

    # ------------------------------------------------------------------ #
    # Probability Model (3-way)
    # ------------------------------------------------------------------ #

    def _dynamic_draw_prob(self, R_home: float, R_away: float) -> float:
        """Increasing draw probability for evenly matched teams."""
        diff = abs((R_home + self.home_adv) - R_away)
        bump = self.draw_max_extra * np.exp(-(diff / self.draw_scale) ** 2)
        pD = self.draw_base + bump
        return float(max(0.0, min(0.5, pD)))

    def _three_way_probs(self, R_home: float, R_away: float) -> dict:
        """Compute final 3-way probabilities."""
        # Two-outcome update
        pH_two = self.expected_home_prob(R_home, R_away)
        pA_two = 1.0 - pH_two

        # Draw share
        pD = self._dynamic_draw_prob(R_home, R_away)

        # Remaining mass
        remaining = max(1e-12, 1.0 - pD)
        pH = remaining * pH_two
        pA = remaining * pA_two

        # Normalize
        Z = pH + pD + pA
        return {"H": pH / Z, "D": pD / Z, "A": pA / Z}

    # ------------------------------------------------------------------ #
    # COMPATIBILITY LAYER for BACKTEST
    # ------------------------------------------------------------------ #

    def update(self, home, away, FTHG, FTAG):
        """
        Backtest-compatible incremental update:
        elo.update(home, away, fthg, ftag)
        """
        # Ensure teams exist
        self._ensure_team(home)
        self._ensure_team(away)

        self.ratings = self.update_ratings_match(
            self.ratings, home, away, FTHG, FTAG
        )

    def predict(self, home, away):
        """
        Backtest-compatible probability prediction:
        elo.predict(home, away) -> (pH, pD, pA)
        """
        self._ensure_team(home)
        self._ensure_team(away)

        R_home = self.ratings[home]
        R_away = self.ratings[away]

        probs = self._three_way_probs(R_home, R_away)
        return probs["H"], probs["D"], probs["A"]
