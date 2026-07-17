#!/usr/bin/env python
"""Algothon 2026 eval for COMBINED v3. Scores the last-250 (leaderboard) window by
default; pass a number to score a different window, e.g. `python eval.py 440`."""
import sys
import numpy as np
import pandas as pd
from Arbitrage_Victims_combined import getMyPosition as getPosition

numTestDays = int(sys.argv[1]) if len(sys.argv) > 1 else 250
prcAll = pd.read_csv("./prices.txt", sep=r"\s+", header=0, index_col=None).values.T
nInst, nt = prcAll.shape
print(f"Loaded {nInst} instruments for {nt} days; scoring last {numTestDays} days")

commRate = np.full(nInst, 0.0001); commRate[0] = 0.00002
dlrPosLimit = np.full(nInst, 10_000); dlrPosLimit[0] = 100_000


def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + param**2)


def calcPL(numTestDays):
    cash = 0; curPos = np.zeros(nInst); totDV = 0; value = 0; comm = 0; pll = []
    startDay = nt - numTestDays
    for t in range(startDay, nt + 1):
        prc = prcAll[:, :t]; cur = prc[:, -1]
        if t < nt:
            npos = np.clip(getPosition(prc), -(dlrPosLimit / cur).astype(int),
                           (dlrPosLimit / cur).astype(int)).astype(int)
        else:
            npos = np.array(curPos)
        d = npos - curPos
        cash -= cur.dot(d) + comm
        dv = cur * np.abs(d); totDV += dv.sum(); comm = (dv * commRate).sum()
        curPos = np.array(npos)
        pl = cash + curPos.dot(cur) - value
        value = cash + curPos.dot(cur)
        if t > startDay:
            pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sh = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return mu, sd, sh, totDV


mu, sd, sh, dv = calcPL(numTestDays)
print("=====")
print(f"mean(PL): {mu:.1f}")
print(f"StdDev(PL): {sd:.2f}")
print(f"annSharpe(PL): {sh:.2f}")
print(f"totDvolume: {dv:.0f}")
print(f"Score: {score(mu, sd):.2f}")
