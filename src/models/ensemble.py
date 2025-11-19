def ensemble_win_probs(dc_probs, elo_probs, w_dc=0.6, w_elo=0.4):
    h1, d1, a1 = dc_probs
    h2, d2, a2 = elo_probs
    h = w_dc*h1 + w_elo*h2
    d = w_dc*d1 + w_elo*d2
    a = w_dc*a1 + w_elo*a2
    Z = h + d + a
    return h/Z, d/Z, a/Z
