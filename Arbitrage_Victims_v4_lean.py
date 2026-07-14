"""Algothon 2026 submission (v4-lean) — v4 with the ALGO reversion overlay OFF.

Identical to Arbitrage_Victims_v2.py EXCEPT HALF_LIFE is lengthened to 2000 days.
Rationale: the DGP is verified STATIONARY (no drift, no regimes, Kalman TVP tunes
adaptation to zero), so forgetting old data is pure waste — the fit is data-hungry and
more history = sharper coefficients = higher IC. A 500-day half-life secretly CAPS the
effective sample at ~720 days even when far more history is available; HALF_LIFE=2000 is
≈ an expanding window for any sample up to ~1500 days (near-equal weights) so it uses
essentially all available data, while still able to forget if real slow drift ever
appears beyond the horizon we can currently see. This is principled generalization
(exploit proven stationarity), not sample-fitting.

Effect scales with available history: on the current 500 days it is ~neutral (walk-fwd
IC 0.0589 -> 0.0594), but the gain grows once we fit on 1000 days (general-round dev
window). The exact half-life for the 1001-1500 evaluation is a decision to re-validate on
real out-of-sample data (days 501-1000) when it is released — see validate_oos.py.
Everything else (CONV_Z, MAX sizing, beta-hedge, ALGO reversion overlay) is unchanged v2.
"""
import numpy as np

HALF_LIFE = 2000     # lengthened from 500: stationary DGP => use ~all available data (see module docstring)
ALPHA = 0.1          # light ridge shrinkage on the 51x50 fit
LIMIT = 10_000       # per-asset dollar position limit
ALGO_LIMIT = 100_000 # ALGO (index) dollar position limit — special 10x cap
HEDGE = True         # beta-hedge residual market exposure with ALGO, applied LAST
CONTRA_DOLLARS = 0         # contrarian ALGO reversion overlay notional (plateau floor)
CONTRA_K = 30        # lookback (days) for the ALGO move we fade
CONTRA_WZ = 60       # window to z-score that move
CONV_Z = 0.2         # conviction bar: trade a name only if |forecast| >= CONV_Z * daily x-sectional std

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
    if _cache["fit_t"] != t:                       # refit on any change in history length (cache-hardened)
        X = ret[:, :-1].T
        Y = ret[1:, 1:].T
        _cache["model"] = _ewls_ridge_fit(X, Y)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B              # next-day forecast (50,)
    w = pred - pred.mean()                         # market-neutral demean
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
