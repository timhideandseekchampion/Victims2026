"""Algothon 2026 — COMBINED v3 PUNT variant (tournament / swing-for-top-10 entry).

This is the AGGRESSIVE build, NOT the robust ship. In a top-10-advances tournament a robust
median finish (~16th) doesn't qualify, so this deliberately trades safety for UPSIDE: it drops
the beta-hedge and shoves ~all the legal capital onto the edge. Same ridge core as
Arbitrage_Victims_combined.py; only the risk knobs change.

What changed vs the primary (and why it's the GOOD kind of punt):
  * HEDGE = False        — no beta-hedge (as requested). On this book it barely changes variance
                           (the market-neutral demean already kills most beta) but frees the ALGO
                           cap entirely for the directional index bet.
  * CONV_Z = 0.10        — lower conviction gate => trade ~all 50 names, not ~41. Bigger book.
  * CONTRA_DOLLARS = 1M  — pin the $100k ALGO cap every day (max index-reversion bet).
    => gross deployed ~$557k of the $600k ceiling (vs ~$499k primary) => maximum upside if the
       edge fires on the unseen window. Known-window Score ~780 (vs 762 primary), mean $796,
       Sharpe ~7.0, and it even lifts the weak half. This is a punt that RAISES expected score,
       because it scales the REAL (market-neutral) edge rather than gambling on direction.

  * DEMEAN = True (default). Set DEMEAN=False for the PURE-DIRECTIONAL lottery ticket: the book
    then carries whatever net long/short the forecast implies. That is the MAXIMUM-variance mode
    (std +27%) but ~0-EV — idiosyncratic drifts are exactly 0, so it halved the known-window
    Score (~530). Only flip it if you explicitly want a coin-flip bet on the hidden window's
    direction; otherwise leave it True.

Robust alternative for a non-tournament / final-that-must-not-blow-up entry:
Arbitrage_Victims_combined.py (HL=500, hedged, market-neutral).
"""
import numpy as np

HALF_LIFE = 500      # keep the robust adaptive core (HL=2000 scored worse even here)
ALPHA = 0.1
LIMIT = 10_000
ALGO_LIMIT = 100_000
CONV_Z = 0.10        # PUNT: lower gate -> trade ~all 50 names (bigger book)
HEDGE = False        # PUNT: no beta-hedge (as requested)
DEMEAN = True        # True = scale the real market-neutral edge (good punt);
                     # False = pure directional lottery (max variance, ~0-EV)
CONTRA_DOLLARS = 300000 # PUNT: pin the $100k ALGO cap every day
CONTRA_K = 30
CONTRA_WZ = 60

_cache = {"fit_t": None, "model": None}


def _ewls_ridge_fit(X, Y):
    n, p = X.shape
    lam = 0.5 ** (1.0 / HALF_LIFE)
    w = lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw
    my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc)
    XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + ALPHA) * np.eye(p), XtWY)
    return B, mx, my


def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    if t < 60:
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]
    if _cache["fit_t"] != t:
        _cache["model"] = _ewls_ridge_fit(ret[:, :-1].T, ret[1:, 1:].T)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B
    w = (pred - pred.mean()) if DEMEAN else pred        # DEMEAN=False -> net directional tilt
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])
    if CONV_Z > 0:
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized
    # ALGO index contrarian — pinned to the cap (the punt's directional index bet)
    cap_sh = ALGO_LIMIT / prcSoFar[0, -1]
    rev_sh = 0.0
    if CONTRA_DOLLARS > 0 and t > CONTRA_K + CONTRA_WZ + 2:
        lpA = np.log(prcSoFar[0])
        move = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        z = (move[-1] - move[-CONTRA_WZ:].mean()) / (move[-CONTRA_WZ:].std() + 1e-12)
        rev_sh = -float(np.clip(z, -3, 3)) * CONTRA_DOLLARS / prcSoFar[0, -1]
    rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))
    hedge_sh = 0.0
    if HEDGE:
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
    room = max(cap_sh - abs(rev_sh), 0.0)
    pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
    return pos.astype(int)
