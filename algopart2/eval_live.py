#!/usr/bin/env python
"""Official-engine backtest of SAFE_rotate.py (identical harness to eval_safe.py)."""
import numpy as np
import pandas as pd
from SAFE_live import getMyPosition as getPosition

pricesFile = "./prices.txt"
numTestDays = 250
defaultCommRate = 0.0001
inst0CommRate = 0.00002
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000

def loadPrices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df.shape[1], df.shape[0], (df.values).T

nInst, nt, prcAll = loadPrices(pricesFile)
print(f"Loaded {nInst} instruments for {nt} days")
commRate = np.full(nInst, defaultCommRate); commRate[0] = inst0CommRate
dlrPosLimit = np.full(nInst, defaultDlrPosLimit); dlrPosLimit[0] = inst0DlrPosLimit

def score(mu, sigma):
    if mu <= 0 or sigma < 1e-10: return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + 1.0)

def calcPL(prcHist, numTestDays):
    cash=0; curPos=np.zeros(nInst); totDVolume=0; value=0; comm=0; todayPLL=[]
    _, nt = prcHist.shape; startDay = nt - numTestDays
    for t in range(startDay, nt + 1):
        prcHistSoFar = prcHist[:, :t]; curPrices = prcHistSoFar[:, -1]
        if t < nt:
            newPosOrig = getPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)
        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos); totDVolume += np.sum(dvolumes)
        comm = np.sum(dvolumes * commRate)
        curPos = np.array(newPos); posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value; value = cash + posValue
        if t > startDay: todayPLL.append(todayPL)
    pll = np.array(todayPLL); plmu, plstd = pll.mean(), pll.std()
    annSharpe = np.sqrt(250) * plmu / plstd if plstd > 0 else 0.0
    return plmu, plstd, annSharpe, totDVolume

meanpl, plstd, sharpe, dvol = calcPL(prcAll, numTestDays)
print("=====")
print(f"mean(PL): {meanpl:.1f}")
print(f"StdDev(PL): {plstd:.2f}")
print(f"annSharpe(PL): {sharpe:.2f}")
print(f"Score: {score(meanpl, plstd):.2f}")
