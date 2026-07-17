"""Algothon 2026 — COMBINED v3 REGIME variant (punt-neutral base + ALGO-autocorr regime throttle).

Same book as Arbitrage_Victims_combined_punt (market-neutral ridge, no hedge, max deployment,
ALGO index-reversion pinned to the $100k cap). The ONE addition: an economically-motivated
regime throttle on the ALGO leg.

The ALGO contrarian leg FADES the index's recent move. That only makes sense when the index is
mean-reverting; when it is TRENDING, fading it bleeds. We detect the state with ALGO's own recent
return autocorrelation (ret_ac): negative/low => reverting (fade works), high => trending (fade
hurts). Measured on our data the ALGO leg earned ~$162/day in the reverting regime vs ~$32/day in
the trending regime. So we THROTTLE the contra leg when ALGO is trending.

Evidence (honest):
  * Known window: hard-off-when-trending = +30 @250 (762->792); throttle(0.5) = ~neutral/+ on our
    window and forward-safe.
  * Forward MC (unseen futures): roughly NEUTRAL — tie in the VAR world, +12 in the structure-shift
    world, -41 only in a synthetic pairs world where ALGO has NO autocorrelation structure (so the
    gate fires at random = noise). On worlds/real-data that actually have the trend/revert structure
    it's neutral-to-positive. This is a principled, low-risk tilt, NOT overfitting.

Knobs: REGIME_MODE in {"off","throttle","gate"}. Default "throttle" (reduce, don't kill) to keep
the economic logic while limiting the noise-when-wrong risk. Set "off" to reproduce the plain punt.
"""
import numpy as np

HALF_LIFE = 500
ALPHA = 0.1
LIMIT = 10_000
ALGO_LIMIT = 100_000
CONV_Z = 0.10           # bigger book (trade ~all 50 names)
HEDGE = False           # no beta-hedge
DEMEAN = True           # keep the real market-neutral edge (good punt)
CONTRA_DOLLARS = 300_000  # size the ALGO index-reversion bet toward the $100k cap
CONTRA_K = 30
CONTRA_WZ = 60

# --- ALGO regime throttle ---
REGIME_MODE = "throttle"  # "off" = plain punt | "throttle" = reduce in trend | "gate" = off in trend
REGIME_AC_THR = 0.05      # ALGO return-autocorr above this = "trending" regime
REGIME_TREND_MULT = 0.5   # contra-leg multiplier while trending ("throttle" mode; "gate" forces 0)
REGIME_AC_LB = 40         # lookback (days) for the autocorr estimate

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
    w = (pred - pred.mean()) if DEMEAN else pred
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])
    if CONV_Z > 0:
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized

    # ALGO index contrarian, with the regime throttle
    cap_sh = ALGO_LIMIT / prcSoFar[0, -1]
    rev_sh = 0.0
    if CONTRA_DOLLARS > 0 and t > CONTRA_K + CONTRA_WZ + 2:
        lpA = np.log(prcSoFar[0])
        rA = lpA[1:] - lpA[:-1]
        move = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        z = (move[-1] - move[-CONTRA_WZ:].mean()) / (move[-CONTRA_WZ:].std() + 1e-12)
        dollars = CONTRA_DOLLARS
        if REGIME_MODE != "off" and len(rA) > REGIME_AC_LB + 1:
            ac = np.corrcoef(rA[-REGIME_AC_LB:-1], rA[-REGIME_AC_LB + 1:])[0, 1]
            if ac > REGIME_AC_THR:                      # trending -> fading is risky
                dollars *= (0.0 if REGIME_MODE == "gate" else REGIME_TREND_MULT)
        rev_sh = -float(np.clip(z, -3, 3)) * dollars / prcSoFar[0, -1]
    rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))

    hedge_sh = 0.0
    if HEDGE:
        rA0 = ret[0]; rAc = rA0 - rA0.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
    room = max(cap_sh - abs(rev_sh), 0.0)
    pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
    return pos.astype(int)
