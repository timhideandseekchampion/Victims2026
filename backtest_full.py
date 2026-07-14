"""Faithful copy of eval.py's scoring loop, but with a configurable test window and
a pluggable strategy module. Does NOT modify teamName.py or eval.py. Importable
without side effects (the demo run is guarded by __main__).
"""
import numpy as np, pandas as pd
import ols_strategy

def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.values.T, df.shape  # (nInst, nt)

prcAll, (nt_all, nInst) = loadPrices("./prices.txt")

commRate = np.full(nInst, 0.0001); commRate[0] = 0.00002
dlrPosLimit = np.full(nInst, 10_000); dlrPosLimit[0] = 100_000

def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10: return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + param**2)

def _reset(strat):
    """Reset a strategy module's fit cache between runs, whatever keys it uses."""
    if hasattr(strat, "reset"):
        strat.reset()
    elif hasattr(strat, "_cache"):
        for k in list(strat._cache.keys()):
            strat._cache[k] = -10 if "last" in k else None

def calcPL(prcHist, numTestDays, strat=ols_strategy):
    cash=0; curPos=np.zeros(nInst); totDVolume=0; value=0; comm=0
    todayPLL=[]; _, nt = prcHist.shape
    startDay = nt - numTestDays
    _reset(strat)
    for t in range(startDay, nt+1):
        prcHistSoFar = prcHist[:, :t]; curPrices = prcHistSoFar[:, -1]
        if t < nt:
            newPosOrig = strat.getMyPosition(prcHistSoFar)
            posLimits = (dlrPosLimit/curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)
        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices*np.abs(deltaPos); totDVolume += np.sum(dvolumes)
        comm = np.sum(dvolumes*commRate)
        curPos = np.array(newPos); posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value; value = cash + posValue
        if t > startDay: todayPLL.append(todayPL)
    pll = np.array(todayPLL); mu, sd = pll.mean(), pll.std()
    sharpe = np.sqrt(250)*mu/sd if sd>0 else 0.0
    return mu, sd, sharpe, totDVolume

if __name__ == "__main__":
    for ntd in [250, 400]:
        mu, sd, sh, dvol = calcPL(prcAll, ntd)
        print(f"== numTestDays={ntd} (days {nt_all-ntd}..{nt_all}) ==")
        print(f"   mean(PL)={mu:.2f}  StdDev={sd:.2f}  annSharpe={sh:.2f}  totDvol={dvol:.0f}  Score={score(mu,sd):.2f}")
