"""Max-sizing OLS strategy — every position at the full dollar limit, no risk scaling.

Same directed-peer-lead-lag forecast as ols_strategy.py, but instead of sizing
proportionally to conviction it takes the SIGN of each asset's (market-neutral)
predicted return and goes full $10k long or short. ALGO (index) is left flat.
This maximises gross book -> maximises the size-scaled eval.py Score, at some cost
to Sharpe vs the conviction-weighted book.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

_cache = {"B": None, "last_fit_t": -10}
LIMIT = 10_000

def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    if t < 60:
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]
    if t - _cache["last_fit_t"] >= 5 or _cache["B"] is None:
        Xtr = ret[:, :-1].T
        Ytr = ret[1:, 1:].T
        _cache["B"] = LinearRegression().fit(Xtr, Ytr)
        _cache["last_fit_t"] = t
    pred = _cache["B"].predict(ret[:, -1].reshape(1, -1))[0]
    w = pred - pred.mean()                      # market-neutral demean
    # MAX sizing: full limit in the predicted direction, every asset
    pos[1:] = np.sign(w) * (LIMIT / prcSoFar[1:, -1])
    return pos.astype(int)
