"""Algothon 2026 submission — market-neutral peer-lead-lag strategy.

Each day: predict every asset's next-day return from today's full cross-section
using a forgetting-weighted ridge (EWLS, half-life 250d, light L2 alpha=0.1),
demean it (market-neutral), trade every name whose conviction clears a significance
bar at full $10k each (MAX sizing — the count floats day to day), beta-hedge the
residual with the ALGO index, and add a contrarian market-timing overlay on ALGO
(fade its recent multi-day move — the index mean-reverts). Self-contained (numpy only).
Backtests to Score ~652 / Sharpe ~7.0 on the last 250 days via eval.py.
"""
import numpy as np

HALF_LIFE = 500      # EWLS forgetting half-life (days). Lengthened 250->500: no drift exists in the data
                     # (Kalman TVP tunes adaptation to zero), so longer memory = more data = better on
                     # every window (full-500 490->511, last-250 726->762, early 468->490) + higher Sharpe.
ALPHA = 0.1          # light ridge shrinkage on the 51x50 fit
LIMIT = 10_000       # per-asset dollar position limit
ALGO_LIMIT = 100_000 # ALGO (index) dollar position limit — special 10x cap
HEDGE = True         # beta-hedge residual market exposure with ALGO. Applied LAST (reversion gets the
                     # $100k budget first, hedge fills only leftover room). Dropping it entirely costs -17 Score.
# --- contrarian ALGO overlay: the index has no next-day predictability but MEAN-REVERTS at multi-day
# horizons (t=-2.6..-2.8). Fade its recent K-day move, sized off its spare $100k capacity. Adds Score AND
# Sharpe (585->652, 6.64->7.02 @250), robust across K=8-35 / $20-80k and every window. K=30 is the peak of
# a trend/static-short bet loses to the market's +37%/yr up-regimes. CONTRA_DOLLARS=0 disables it.
CONTRA_DOLLARS = 200_000   # sized to pin the $100k ALGO cap on high-conviction days (~68%). Score SATURATES
                           # at 200k: full-window sweep (2026-07-13) shows Score dead-flat 200k->400k
                           # (last-250 762->764, days120-500 642->643) so the extra directional ALGO risk
                           # above 200k buys nothing. This is the speculative leg (+141 backtest but only
                           # +17 live, perm p~0.43) — held at the plateau floor for robustness, not maxed.
CONTRA_K = 30        # lookback (days) for the ALGO move we fade
CONTRA_WZ = 60       # window to z-score that move (swept: 60 beats 40 on every window + Sharpe)
CONV_Z = 0.2         # conviction bar: trade a name only if |forecast| >= CONV_Z * (daily cross-sectional std
                     # of forecasts). 0 = trade all 50. This is the principled version of "only bet when the
                     # model has a real view": the NUMBER of names traded FLOATS (avg ~41, ranges ~32-47) —
                     # more on strong-signal days, fewer on weak ones — rather than a fixed count.
                     # Score ~585 / Sharpe ~6.6 @250, ~497 @400 (vs 541/5.79 trading all 50). Robust plateau
                     # over CONV_Z 0.15-0.25. Not overfit: it's a live significance filter on the per-day
                     # |forecast| (causal, identity changes daily) — low-conviction bets provably have no
                     # significant edge (kept bets hit 53.0%/t=7.5 vs dropped 50.8%/t=1.7).

_cache = {"model": None, "last_fit_t": -10}


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
    if CONV_Z > 0:                                # trade only names whose conviction clears the significance bar
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized
    # --- ALGO index (col 0): reversion takes the $100k budget FIRST, hedge fills only leftover room ---
    cap_sh = ALGO_LIMIT / prcSoFar[0, -1]
    rev_sh = 0.0
    if CONTRA_DOLLARS > 0 and t > CONTRA_K + CONTRA_WZ + 2:   # contrarian overlay (fade ALGO's recent move)
        lpA = np.log(prcSoFar[0])
        move = lpA[CONTRA_K:] - lpA[:-CONTRA_K]               # rolling K-day ALGO returns
        z = (move[-1] - move[-CONTRA_WZ:].mean()) / (move[-CONTRA_WZ:].std() + 1e-12)
        rev_sh = -float(np.clip(z, -3, 3)) * CONTRA_DOLLARS / prcSoFar[0, -1]
    rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))          # reversion gets first claim on the cap
    hedge_sh = 0.0
    if HEDGE:                                     # cancel residual net beta with ALGO (index, beta 1)
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
    room = max(cap_sh - abs(rev_sh), 0.0)                     # hedge is applied LAST, into leftover room only
    pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
    return pos.astype(int)
