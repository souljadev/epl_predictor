import json
from src.models.elo import EloModel


def train_elo_model(df):
    """
    Wrapper that trains the Elo model on historical match data
    and returns a JSON-serializable parameters dict.
    """

    model = EloModel()
    model.fit(df)

    # Convert numpy floats to Python floats
    ratings = {team: float(r) for team, r in model.ratings.items()}

    params = {
        "ratings": ratings,
        "k_factor": float(model.k),
        "home_advantage": float(model.home_adv),
        "base_rating": float(model.base_rating),
        "draw_base": float(model.draw_base),
        "draw_max_extra": float(model.draw_max_extra),
        "draw_scale": float(model.draw_scale),
    }

    return params
