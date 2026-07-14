#!/usr/bin/env python
"""Generate showcase_data.json — full instrumented backtests of every production
strategy in algo26v1, run through eval.py-faithful accounting, plus the shared
research/edge story pulled from the existing analysis JSONs.

Strategies showcased (all market-neutral cross-sectional peer-lead-lag forecasts):
  * conviction  — ols_strategy.py : conviction-weighted, expanding OLS, ALGO flat
  * max         — ols_max.py      : MAX (full-limit sign) sizing, expanding OLS, ALGO flat
  * adaptive    — ols_adaptive.py : EWLS(h=250) + ridge a=0.1 + MAX + ALGO beta-hedge  (SUBMISSION)

Output feeds showcase_dashboard.html (a single self-contained page).
"""
import json, importlib
import numpy as np, pandas as pd

PRICES = "./prices.txt"
NUM_TEST_DAYS = 250

df = pd.read_csv(PRICES, sep=r"\s+", header=0)
NAMES = list(df.columns)
prcAll = df.values.T                      # (51, 500)
nInst, nt_all = prcAll.shape

commRate = np.full(nInst, 0.0001); commRate[0] = 0.00002
dlrPosLimit = np.full(nInst, 10_000); dlrPosLimit[0] = 100_000


def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return float(mu)
    sr = np.sqrt(250) * mu / sigma
    return float(mu * sr**2 / (sr**2 + param**2))


def reset(strat):
    if hasattr(strat, "reset"):
        strat.reset()
    elif hasattr(strat, "_cache"):
        for k in list(strat._cache.keys()):
            strat._cache[k] = -10 if "last" in k else None


def run(strat, numTestDays=NUM_TEST_DAYS):
    """Faithful eval.py accounting, fully instrumented. Records per-day PnL, the
    full dollar-position matrix, per-asset PnL attribution and commission."""
    reset(strat)
    cash = 0.0; curPos = np.zeros(nInst); value = 0.0; comm = 0.0
    totDVolume = 0.0
    startDay = nt_all - numTestDays

    dayPL = []                 # scored daily PnL
    dayComm = []               # commission paid that day
    dayTurn = []               # $ traded that day
    dayGross = []              # gross book ($) held that day
    dayNet = []                # net $ (long-short) book
    dayTrades = []             # # of names whose position changed (size or side)
    dayFlips = []              # # of names whose SIGN changed (directional turnover)
    posDollarMat = []          # (days, nInst) dollar positions held
    assetPnL = np.zeros(nInst) # cumulative per-asset gross PnL attribution
    prevPos = np.zeros(nInst)
    prevSign = np.zeros(nInst)
    prevPrices = prcAll[:, startDay - 1]

    for t in range(startDay, nt_all + 1):
        prcHistSoFar = prcAll[:, :t]
        curPrices = prcHistSoFar[:, -1]
        if t < nt_all:
            newPosOrig = strat.getMyPosition(prcHistSoFar)
            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            newPos = np.array(curPos)
        deltaPos = newPos - curPos
        cash -= curPrices.dot(deltaPos) + comm
        dvolumes = curPrices * np.abs(deltaPos)
        dvol = float(np.sum(dvolumes)); totDVolume += dvol
        comm = float(np.sum(dvolumes * commRate))
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        value = cash + posValue

        if t > startDay:
            # attribute PnL: position held yesterday * price move today
            perAsset = prevPos * (curPrices - prevPrices)
            assetPnL += perAsset
            dayPL.append(float(todayPL))
            dayComm.append(comm)
            dayTurn.append(dvol)
            posD = curPos * curPrices
            dayGross.append(float(np.abs(posD).sum()))
            dayNet.append(float(posD.sum()))
            dayTrades.append(int((deltaPos != 0).sum()))
            curSign = np.sign(curPos)
            dayFlips.append(int(((curSign != prevSign) & (curSign != 0)).sum()))
            posDollarMat.append([int(round(x)) for x in posD])
        prevPos = curPos.copy()
        prevSign = np.sign(curPos)
        prevPrices = curPrices.copy()

    pll = np.array(dayPL)
    mu, sd = float(pll.mean()), float(pll.std())
    sharpe = float(np.sqrt(250) * mu / sd) if sd > 0 else 0.0
    cum = np.cumsum(pll)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    ddmin_i = int(np.argmin(dd))
    total_trades = int(np.sum(dayTrades))       # total position updates (size or side)
    total_flips = int(np.sum(dayFlips))         # total directional flips (sign changes)
    # avg directional holding period = avg names held / avg names flipping side per day
    avg_names_held = float(np.mean([(np.array(p) != 0).sum() for p in posDollarMat]))
    avg_flips = float(np.mean(dayFlips))
    avg_hold = round(avg_names_held / avg_flips, 1) if avg_flips else 0.0

    posD_last = np.array(posDollarMat[-1])
    return {
        "meta": {
            "score": round(score(mu, sd), 1),
            "sharpe": round(sharpe, 2),
            "meanPL": round(mu, 1),
            "stdPL": round(sd, 1),
            "totalPnL": round(float(cum[-1]), 0),
            "maxDD": round(float(dd[ddmin_i]), 0),
            "maxDDat": ddmin_i,
            "maxDDpct": round(float(dd[ddmin_i]) / max(peak[ddmin_i], 1e-9) * 100, 1) if peak[ddmin_i] > 0 else 0.0,
            "bestDay": round(float(pll.max()), 0),
            "worstDay": round(float(pll.min()), 0),
            "winDays": round(float((pll > 0).mean() * 100), 1),
            "days": len(pll),
            "totalTrades": total_trades,
            "avgTrades": round(float(np.mean(dayTrades)), 1),
            "totalFlips": total_flips,
            "avgHold": avg_hold,
            "grossBook": round(float(np.mean(dayGross)), 0),
            "netBookStd": round(float(np.std(dayNet)), 0),
            "totDvol": round(totDVolume, 0),
            "commTotal": round(float(np.sum(dayComm)), 0),
            "commPctGross": round(float(np.sum(dayComm)) / max(float(np.sum(np.abs(pll)) + np.sum(dayComm)), 1e-9) * 100, 1),
        },
        "cumPL": [round(float(x), 1) for x in cum],
        "dayPL": [round(float(x), 1) for x in pll],
        "drawdown": [round(float(x), 1) for x in dd],
        "turnover": [round(float(x), 0) for x in dayTurn],
        "grossBook": [round(float(x), 0) for x in dayGross],
        "netBook": [round(float(x), 0) for x in dayNet],
        "trades": dayTrades,
        # per-asset attribution (tradeable 50 only; skip ALGO index at 0 unless it traded)
        "assetPnL": {NAMES[i]: round(float(assetPnL[i]), 0) for i in range(nInst)},
        "lastPos": {NAMES[i]: int(posD_last[i]) for i in range(nInst)},
        "posMatrix": posDollarMat,      # (days, nInst) dollar positions
    }


STRATS = [
    ("conviction", "Conviction OLS", "ols_strategy",
     "Conviction-weighted OLS forecast, strongest signal at the $10k limit. Expanding window, refit every 5 days. ALGO index left flat.",
     {"sizing": "conviction", "window": "expanding", "refit": 5, "hedge": "none", "ridge": "—"}),
    ("max", "MAX OLS", "ols_max",
     "Full-limit SIGN sizing: every name at $10k long/short in the predicted direction. Expanding window, refit every 5 days. ALGO flat.",
     {"sizing": "MAX (sign)", "window": "expanding", "refit": 5, "hedge": "none", "ridge": "—"}),
    ("adaptive", "Adaptive — submission", "ols_adaptive",
     "The submission. EWLS forgetting fit (half-life 250) + light ridge α=0.1 stabilises the 51×50 coefficients; MAX sizing but only on names whose conviction clears a significance bar (the count floats ~32-47/day, skipping no-edge coin-flips); residual net beta hedged with the cheap ALGO index; plus a contrarian ALGO overlay that fades the index's recent move (it mean-reverts). Refit daily.",
     {"sizing": "MAX + conv-bar", "window": "EWLS h=250", "refit": 1, "hedge": "β via ALGO", "ridge": "0.1"}),
]

strategies = {}
for key, label, mod, desc, cfg in STRATS:
    m = importlib.import_module(mod)
    importlib.reload(m)
    print(f"running {label} ...", flush=True)
    r = run(m)
    r["label"] = label
    r["module"] = mod + ".py"
    r["desc"] = desc
    r["config"] = cfg
    strategies[key] = r
    print(f"   Score {r['meta']['score']}  Sharpe {r['meta']['sharpe']}  PnL {r['meta']['totalPnL']:.0f}", flush=True)

# ----- shared research / edge context from the existing verified analyses -----
def load(fn, default=None):
    try:
        with open(fn) as f:
            return json.load(f)
    except Exception:
        return default

research = {
    "sweep": load("panel_data.json", {}).get("sweep"),      # model bake-off (IC by regularisation)
    "warm": load("full500.json", {}).get("warm"),           # IC vs warm-up start (OOS stability)
    "robust": load("robust.json"),                          # permutation / bootstrap / net-sharpe
    "hedge": load("test_data.json"),                        # hedge variant comparison
    "leadlag": None,
}
dd = load("dash_data.json", {})
if dd:
    research["leadlag"] = {
        "meta": dd.get("meta"),
        "nodes": dd.get("nodes"),
        "edges": dd.get("edges"),
    }

# scheme comparison (expanding vs ewls half-lives) — recompute quickly via IC is heavy;
# reuse harness numbers if present, else derive from strategies we ran.
payload = {
    "generated": "2026-07-11",
    "universe": {"nAssets": len(NAMES) - 1, "nDays": nt_all, "testDays": NUM_TEST_DAYS,
                 "names": NAMES, "indexName": NAMES[0]},
    "strategies": strategies,
    "order": [s[0] for s in STRATS],
    "research": research,
    "prices": {NAMES[i]: [round(float(v), 2) for v in prcAll[i, -NUM_TEST_DAYS:]] for i in range(nInst)},
}

with open("showcase_data.json", "w") as f:
    json.dump(payload, f, separators=(",", ":"))
print("\nwrote showcase_data.json  (%d strategies, %d assets, %d test days)" %
      (len(strategies), len(NAMES) - 1, NUM_TEST_DAYS))
