"""Adaptive OLS + MAX sizing — the submission candidate.

Same directed-peer-lead-lag forecast and same MAX sizing as ols_max.py, but the
coefficient matrix B is fit with a FORGETTING scheme (exponentially-weighted least
squares, half-life 120 days by default) so it adapts to a drifting relationship.
Flip CONFIG to compare schemes; copy this file to teamName.py at submission time.
"""
import numpy as np
import adaptive_estimator as ae

# scheme: "expanding" (never forget, == ols_max), "rolling"+"window", or "ewls"+"half_life".
# Default half_life=250: mild recency weighting that is near-free on the current stable
# data (harness: -1.6% Score vs expanding) while still fully re-learning a drifted
# relationship within ~1yr. Shorter half-lives adapt faster but cost real Score here
# (h=120: -12.7%, h=60: -44%) — only lower it if live data actually shows drift.
# hedge="beta": use the ALGO index to cancel the book's residual net beta (equal $ long/short
# is NOT beta-neutral since betas span 0.5-1.68). Free +0.33 Sharpe / +10 Score in backtest.
# alpha=0.1: light L2 shrinkage on the 51x50 fit. Stabilises the noisy coefficients ->
# sharper sign accuracy -> big Score/Sharpe gain (442->541, 4.62->5.79). Robust across
# alpha 0.03-0.3; heavy ridge (>=1) over-shrinks and kills it. Ensembling other models did
# NOT help (they're 0.83-correlated or noise) — the win is this single shrinkage term.
# conv_z=0.2: only trade names whose |forecast| clears 0.2x the daily cross-sectional std of forecasts
# (a live significance filter; the number of names traded floats ~32-47). Low-conviction bets have no
# significant edge (kept 53%/t=7.5 vs dropped 51%/t=1.7). Score 541->585, Sharpe 5.79->6.6.
# contra_dollars: contrarian ALGO overlay — fade the index's recent K-day move (it mean-reverts, t=-2.8;
# 89% of days, even through up-regimes). Adds Score AND Sharpe (585->637, 6.64->6.87), robust across every
# window and param. Not a drift bet (a static short loses on up-regimes) — this is symmetric mean-reversion.
CONFIG = {"scheme": "ewls", "half_life": 250, "refit_every": 1, "hedge": "beta", "alpha": 0.1, "conv_z": 0.2,
          "contra_dollars": 40_000, "contra_k": 30, "contra_wz": 40}
LIMIT = 10_000
_cache = {"model": None, "last_fit_t": -10}


def reset():
    _cache.update({"model": None, "last_fit_t": -10})


def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    if t < 60:
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]
    if _cache["model"] is None or t - _cache["last_fit_t"] >= CONFIG.get("refit_every", 1):
        X = ret[:, :-1].T
        Y = ret[1:, 1:].T
        _cache["model"] = ae.fit_rows(X, Y, CONFIG)
        _cache["last_fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B          # next-day forecast (50,)
    w = pred - pred.mean()                       # market-neutral demean
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])    # MAX sizing on the 50 assets
    cz = CONFIG.get("conv_z", 0.0)
    if cz > 0:                                    # only bet where conviction clears the significance bar
        sized = np.where(np.abs(w) >= cz * (np.std(w) + 1e-12), sized, 0.0)
    pos[1:] = sized
    if CONFIG.get("hedge") == "beta":            # cancel residual net beta with the ALGO index
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom   # each asset's beta to ALGO
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas                       # $ beta exposure of the book
        pos[0] = -net_beta / prcSoFar[0, -1]                                  # offset with ALGO (index, β≈1)
    cd = CONFIG.get("contra_dollars", 0.0); K = CONFIG.get("contra_k", 20); Wz = CONFIG.get("contra_wz", 40)
    if cd > 0 and t > K + Wz + 2:                 # contrarian ALGO overlay: fade its recent K-day move
        lpA = np.log(prcSoFar[0]); move = lpA[K:] - lpA[:-K]
        z = (move[-1] - move[-Wz:].mean()) / (move[-Wz:].std() + 1e-12)
        pos[0] += -float(np.clip(z, -3, 3)) * cd / prcSoFar[0, -1]
    return pos.astype(int)
