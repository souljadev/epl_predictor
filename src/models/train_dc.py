import json
from src.models.dixon_coles import DixonColesModel


def train_dc_model(df):
    """
    Wrapper to train the incremental Dixon–Coles model
    and return parameters in a serializable dict.
    """

    model = DixonColesModel()
    model.fit(df)

    # Convert numpy floats to Python floats
    attack = {team: float(val) for team, val in model.attack.items()}
    defense = {team: float(val) for team, val in model.defense.items()}

    params = {
        "attack": attack,
        "defense": defense,
        "rho": float(model.rho),
        "home_adv": float(model.home_adv),
    }

    return params
