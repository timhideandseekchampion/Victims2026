"""test_llalgo.py — score the lead-lag-driven ALGO leg vs the shipped reversion leg.
Same accounting as eval_core.py (commissions carry day-to-day, integer clip, score = mu*SR^2/(SR^2+1)),
run over several 250-day windows. Baseline = SAFE.getMyPosition (reversion ALGO leg);
lead-lag variants = SAFE_llalgo with ALGO_LL_W in {1.0, 0.7, 0.5}."""
import numpy as np, pandas as pd
import SAFE, SAFE_llalgo

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd
    return mu * sr**2 / (sr**2 + 1.0)

def run(getpos, startDay, endDay):
    """PnL over test days (startDay, endDay], mirroring eval_core.calcPL."""
    cash = 0.0; curPos = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(startDay, endDay + 1):
        cur = prc[:, t-1]
        if t < endDay:
            newPos = np.asarray(getpos(prc[:, :t]))
            lim = (dlr / cur).astype(int)
            newPos = np.clip(newPos, -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        dP = newPos - curPos
        cash -= cur.dot(dP) + comm
        comm = np.sum(cur * np.abs(dP) * commRate)
        curPos = np.array(newPos)
        pl = cash + curPos.dot(cur) - value
        value = cash + curPos.dot(cur)
        if t > startDay: pll.append(pl)
    pll = np.array(pll)
    return pll.mean(), pll.std(), score(pll.mean(), pll.std())

WINDOWS = {"500-750 (GRADED)": (501, 750), "400-650": (401, 650), "250-500": (251, 500)}

def show(name, getpos):
    print(f"\n{name}")
    for wl, (S, E) in WINDOWS.items():
        mu, sd, sc = run(getpos, S, E)
        print(f"  {wl:<18} mu={mu:7.1f}  std={sd:8.1f}  score={sc:7.1f}")

show("BASELINE  (SAFE.py, reversion ALGO leg)", SAFE.getMyPosition)

# gated: reversion by default, lead-lag ONLY when |frac| >= gate
SAFE_llalgo.ALGO_LL_W = 1.0
for gate in (0.06, 0.10, 0.12, 0.14):
    SAFE_llalgo.ALGO_LL_GATE = gate
    show(f"GATED     |frac|>={gate}  (reversion default, pure lead-lag on high-skew days)",
         SAFE_llalgo.getMyPosition)
