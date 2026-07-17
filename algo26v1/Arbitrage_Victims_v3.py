"""Algothon 2026 submission (v3) — v2 with the per-asset intercept dropped.

Identical to Arbitrage_Victims_v2.py EXCEPT the forecast omits the fitted per-asset
mean `my`. Rationale: the organizers set every stock's idiosyncratic drift to exactly
0 (verified: chi2(49) p=0.61, 0/50 names with |t|>2), so `my` estimates a quantity
that is truly zero — it is pure estimation noise. Dropping it imposes that known prior
and monotonically improves walk-forward forecast IC (0.0589 -> 0.0637, t 6.9->7.5,
better in BOTH halves). This is model SIMPLIFICATION toward ground truth (one fewer
term), not a fitted signal, so it cannot overfit.

HONEST EXPECTATION: this is a forecast-quality improvement, NOT a score improvement.
Because sizing is sign-based (MAX $10k), the calibration gain barely reaches PnL:
paired vs v2 it is +6/d full-window (t=0.19) and -23/d on the last 250 (a wash within
noise). Ship it only as a cleaner/more-principled model; expect score ~= v2 (~502
public). v2 remains the safe fallback. Do NOT partially keep `my` (a tuned blend k in
(0,1) looks better on this sample but that gain is a warm-up artifact and is overfitting).
"""
import numpy as np

HALF_LIFE = 500      # EWLS forgetting half-life (days)
ALPHA = 0.1          # light ridge shrinkage on the 51x50 fit
LIMIT = 10_000       # per-asset dollar position limit
ALGO_LIMIT = 100_000 # ALGO (index) dollar position limit — special 10x cap
HEDGE = True         # beta-hedge residual market exposure with ALGO, applied LAST
CONTRA_DOLLARS = 200_000   # contrarian ALGO reversion overlay notional (plateau floor)
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
    if _cache["fit_t"] != t:
        X = ret[:, :-1].T
        Y = ret[1:, 1:].T
        _cache["model"] = _ewls_ridge_fit(X, Y)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]
    pred = (ret[:, -1] - mx) @ B                  # v3: intercept `my` DROPPED (true idio drift is exactly 0)
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
