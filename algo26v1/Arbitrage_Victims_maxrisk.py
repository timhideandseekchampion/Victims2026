"""Algothon 2026 — MAX-RISK experimental variant (signal-directed).

Deliberately NOT the robust ship artifact. This exists to see what a maximally
aggressive version does on the LIVE board, on the explicit instruction "put on max
risk, don't trust the backtest." It keeps the validated peer-lead-lag book edge but
removes every risk-DAMPENER and cranks every sizing knob to the exchange caps:

  1. RAW forecast tilt (w = pred, NOT demeaned) -> the book carries whatever NET
     directional exposure the model's aggregate view implies, instead of being forced
     market-neutral. This is the main new source of variance.
  2. CONV_Z = 0.1 -> conviction filter lightened from 0.2; nearly all 50 names trade
     at the full $10k cap (close to $500k gross deployed every day).
  3. HEDGE = False -> residual market beta is left ON, not hedged out with ALGO.
  4. CONTRA_DOLLARS = 1_000_000 -> the ALGO reversion notional is so large it PINS the
     $100k (10x) ALGO cap every day, in the direction of the trailing-move reversion
     signal. Max directional index bet, signal-directed (no hardcoded long/short call).

EXPECTED PROFILE: much higher variance, LOWER expected value than v4 (the removed
demean/hedge/conviction were each validated as +EV or variance-reducing on days 1-500;
idiosyncratic drifts are exactly 0 so a net tilt is ~0-EV + high variance). Prints big
in a trending window, bleeds in an adverse one. Do NOT treat as the general-round
artifact — v4 remains that. See module history / FINDINGS.md.
"""
import numpy as np

HALF_LIFE = 2000     # v4 base: stationary DGP => use ~all available data
ALPHA = 0.1          # light ridge shrinkage on the 51x50 fit
LIMIT = 10_000       # per-asset dollar position limit
ALGO_LIMIT = 100_000 # ALGO (index) dollar position limit — special 10x cap
HEDGE = False        # MAX RISK: beta hedge OFF (let residual market exposure ride)
CONTRA_DOLLARS = 1_000_000  # MAX RISK: huge -> pins the $100k ALGO cap every day
CONTRA_K = 30        # lookback (days) for the ALGO move we fade
CONTRA_WZ = 60       # window to z-score that move
CONV_Z = 0.1         # MAX RISK: conviction filter LIGHTENED (0.2 -> 0.1) -> nearly all names trade

_cache = {"model": None, "fit_t": None}


def _ewls_ridge_fit(X, Y):
    """Exponentially-weighted ridge, weighted-demean form. Returns (B, mx, my)."""
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
    if _cache["fit_t"] != t:                       # refit on any change in history length
        X = ret[:, :-1].T
        Y = ret[1:, 1:].T
        _cache["model"] = _ewls_ridge_fit(X, Y)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B              # next-day forecast (50,)
    w = pred                                       # MAX RISK: RAW forecast (net tilt floats with signal)
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])
    if CONV_Z > 0:
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized
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
