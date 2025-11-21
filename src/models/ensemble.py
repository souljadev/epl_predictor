from typing import Dict, Tuple, Union


Probs = Union[Dict[str, float], Tuple[float, float, float], list]


def _to_tuple(probs: Probs) -> Tuple[float, float, float]:
    """
    Convert different probability formats to (pH, pD, pA).
    Supports:
    - dict with keys "H","D","A"
    - tuple/list in order (pH, pD, pA)
    """
    if isinstance(probs, dict):
        return float(probs["H"]), float(probs["D"]), float(probs["A"])
    else:
        pH, pD, pA = probs
        return float(pH), float(pD), float(pA)


def ensemble_win_probs(
    dc_probs: Probs,
    elo_probs: Probs,
    w_dc: float = 0.6,
    w_elo: float = 0.4,
) -> Tuple[float, float, float]:
    """
    Blend Dixon–Coles and Elo probabilities.

    Parameters
    ----------
    dc_probs : dict or tuple
        DC probabilities for (H,D,A). Can be:
          - {"H":..., "D":..., "A":...}
          - (pH, pD, pA)
    elo_probs : dict or tuple
        Elo probabilities in the same format.
    w_dc : float
        Weight for the DC model.
    w_elo : float
        Weight for the Elo model.

    Returns
    -------
    (pH, pD, pA) : tuple of floats
        Normalized ensemble probabilities.
    """
    h1, d1, a1 = _to_tuple(dc_probs)
    h2, d2, a2 = _to_tuple(elo_probs)

    pH = w_dc * h1 + w_elo * h2
    pD = w_dc * d1 + w_elo * d2
    pA = w_dc * a1 + w_elo * a2

    Z = pH + pD + pA
    if Z <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0

    return pH / Z, pD / Z, pA / Z
