def ensemble_win_probs(dc_probs, elo_probs, w_dc=0.6, w_elo=0.4):
    # Extract floats explicitly
    h1 = float(dc_probs["H"])
    d1 = float(dc_probs["D"])
    a1 = float(dc_probs["A"])

    h2 = float(elo_probs["H"])
    d2 = float(elo_probs["D"])
    a2 = float(elo_probs["A"])

    # Weighted blend
    h = w_dc*h1 + w_elo*h2
    d = w_dc*d1 + w_elo*d2
    a = w_dc*a1 + w_elo*a2

    # Normalize to ensure sum = 1
    Z = h + d + a
    return h/Z, d/Z, a/Z
