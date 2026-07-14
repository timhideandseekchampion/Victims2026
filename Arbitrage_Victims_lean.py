"""Algothon 2026 — LEAN variant (book-only, no ALGO reversion overlay).

Rationale: the full strategy's ALGO reversion overlay inflated the backtest
(+141 on last-250) but delivered only +17 live (8x decay, permutation p~0.43) —
it's overfit to the visible sample. This variant keeps ONLY the pieces that are
statistically validated on the full 500 days and should GENERALIZE:
  * EWLS forgetting-ridge peer-lead-lag forecast (half-life 500, light L2 0.1)
    — cross-sectional IC 0.058, t=6.88 on full data, both halves, perm p<1e-4;
      the best of 21 estimators, significantly beats plain OLS (paired p=0.02).
  * conviction bar (trade only names whose |forecast| clears a significance
    threshold) — dropped names are coin-flips (t=0.0), kept names t=5.1, p=5e-4.
  * MAX sizing ($10k/name) — Score-optimal for this objective.
  * beta-hedge the residual net market exposure with the ALGO index (mechanical
    risk reduction, not a fitted signal).
No ALGO reversion, no directional bet, no tuned overlays. The hypothesis: this
scores LOWER on backtest (~585-620 last-250) but does NOT decay out-of-sample,
so it may live-score at/above the full strategy. Self-contained (numpy only).
"""
import numpy as np

HALF_LIFE = 500      # EWLS forgetting half-life. Score-swept 2026-07-13: there's an INTERIOR optimum at
                     # HL~500-750 (Score@440 468/484/504/503/496/464 for HL 250/375/500/750/1000/inf;
                     # @250 peaks at HL=750). Infinite memory (expanding window) is the WORST option -- more
                     # data marginally improves cross-sectional IC (.0593->.0599) but that does NOT survive
                     # into Score; recency-weighting sizes/times positions better. 500-750 is a flat plateau;
                     # kept at 500 (validated, best@440). Do NOT set None/inf. 750 is an equally-valid pick.
ALPHA = 0.1          # light ridge shrinkage on the 51x50 fit
LIMIT = 10_000       # per-asset dollar position limit
CONV_Z = 0.1         # conviction bar: trade a name only if |forecast| >= CONV_Z * daily cross-sectional std
HEDGE = True         # beta-hedge residual net market exposure with ALGO (mechanical, generalizes)
VOL_TARGET = True    # scale the book down when recent idiosyncratic vol spikes above its long-run level, so
VOL_SHORT = 20       # PnL StdDev stays stable if the unseen window has a higher-vol regime (protects the
                     # Sharpe factor in Score). Caps at 1.0 (never exceed limits), floors at 0.5. On the
                     # current sample recent~=long-run so scale~=1 (near no-op); it's forward insurance.

_cache = {"model": None, "last_fit_t": -10}


def _ewls_ridge_fit(X, Y):
    """Exponentially-weighted ridge, weighted-demean form. Returns (B, mx, my)."""
    n, p = X.shape
    if HALF_LIFE is None:                        # expanding window: weight all history equally
        w = np.ones(n)
    else:
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
    if t < 60:                                   # need a warm-up before fitting
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]                 # daily log returns (nInst, t-1)
    if _cache["model"] is None or t - _cache["last_fit_t"] >= 1:
        X = ret[:, :-1].T                        # today's cross-section (all 51)
        Y = ret[1:, 1:].T                        # next-day return of the 50 tradeable assets
        _cache["model"] = _ewls_ridge_fit(X, Y)
        _cache["last_fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B            # next-day forecast (50,)
    w = pred - pred.mean()                        # market-neutral demean
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])    # MAX sizing on the 50 assets
    if CONV_Z > 0:                                # trade only names whose conviction clears the bar
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    if VOL_TARGET and ret.shape[1] > VOL_SHORT:   # de-risk if recent vol > long-run vol (regime insurance)
        rn = ret[1:]                              # constituent log returns (50, t-1)
        vs = np.std(rn[:, -VOL_SHORT:], axis=1).mean()   # recent idiosyncratic vol
        vl = np.std(rn, axis=1).mean()                   # expanding long-run baseline
        if vs > 1e-12:
            sized = sized * float(np.clip(vl / vs, 0.5, 1.0))
    pos[1:] = sized
    if HEDGE:                                     # cancel residual net beta with ALGO (index) — risk reduction only
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        pos[0] = -net_beta / prcSoFar[0, -1]
    return pos.astype(int)
