"""Backtest-only strategy: market-neutral OLS cross-sectional forecast.

getMyPosition(prcSoFar) predicts each of the 50 tradeable assets' next-day return
from today's cross-section of all 51 returns (expanding-window OLS, refit every 5
days), goes long the top / short the bottom dollar-neutral, sized to the eval.py
position limits. ALGO (instrument 0, the index) is left flat — the demeaned book is
already market-neutral. This is a validation artifact, NOT a tuned submission.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

_cache = {"B": None, "last_fit_t": -10}

def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    if t < 60:                       # need warm-up to fit
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]     # (51, t-1)
    # refit every 5 days on the expanding window
    if t - _cache["last_fit_t"] >= 5 or _cache["B"] is None:
        Xtr = ret[:, :-1].T          # days 0..t-3  -> features = that day's 51 returns
        Ytr = ret[1:, 1:].T          # next-day returns of the 50 tradeable assets
        _cache["B"] = LinearRegression().fit(Xtr, Ytr)
        _cache["last_fit_t"] = t
    pred = _cache["B"].predict(ret[:, -1].reshape(1, -1))[0]   # (50,) next-day forecast
    w = pred - pred.mean()           # cross-sectional demean -> dollar-neutral
    mx = np.abs(w).max()
    if mx < 1e-12:
        return pos
    dollars = (w / mx) * 10_000      # strongest conviction sits at the $10k limit
    pos[1:] = (dollars / prcSoFar[1:, -1]).astype(int)
    return pos
