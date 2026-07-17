#!/usr/bin/env python
"""Ground-truth scorer: run any module's getMyPosition through the exact eval.py PnL loop.

Usage: score_all.py [numTestDays ...]  (defaults 250 440)
Prints Score / mean / std / Sharpe for each Arbitrage_Victims* module.
"""
import sys, importlib
import numpy as np
import pandas as pd

prcAll = pd.read_csv("./prices.txt", sep=r"\s+", header=0, index_col=None).values.T
nInst, nt = prcAll.shape

defaultCommRate, inst0CommRate = 0.0001, 0.00002
defaultDlr, inst0Dlr = 10_000, 100_000
commRate = np.full(nInst, defaultCommRate); commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlr); dlrPosLimit[0] = inst0Dlr


def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + param**2)


def calcPL(getPosition, numTestDays):
    cash = 0; curPos = np.zeros(nInst); totDVolume = 0; value = 0; comm = 0
    todayPLL = []
    startDay = nt - numTestDays
    for t in range(startDay, nt + 1):
        prcHistSoFar = prcAll[:, :t]
        curPrices = prcHistSoFar[:, -1]
        if t < nt:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)
        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        totDVolume += np.sum(dvolumes)
        comm = np.sum(dvolumes * commRate)
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue
        if t > startDay:
            todayPLL.append(todayPL)
    pll = np.array(todayPLL)
    mu, sd = np.mean(pll), np.std(pll)
    sh = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return mu, sd, sh, score(mu, sd)


def load(modname):
    m = importlib.import_module(modname)
    importlib.reload(m)
    # reset any module-level cache
    if hasattr(m, "_cache"):
        for k in list(m._cache):
            m._cache[k] = None if k != "last_fit_t" else -10
    return m.getMyPosition


if __name__ == "__main__":
    windows = [int(x) for x in sys.argv[1:]] or [250, 440]
    mods = [
        "Arbitrage_Victims", "Arbitrage_Victims_v2", "Arbitrage_Victims_v4",
        "Arbitrage_Victims_v5", "Arbitrage_Victims_maxrisk",
    ]
    print(f"{'module':32s} " + " ".join(f"{'S@'+str(w):>10s} {'Sh@'+str(w):>7s}" for w in windows))
    for mod in mods:
        row = f"{mod:32s} "
        for w in windows:
            gp = load(mod)
            mu, sd, sh, sc = calcPL(gp, w)
            row += f"{sc:10.1f} {sh:7.2f} "
        print(row)
