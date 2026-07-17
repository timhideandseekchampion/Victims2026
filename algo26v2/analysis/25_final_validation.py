"""Robustness check for the final book: sub-window stability + leaner variants.

We re-import the strategy's components and test weight variants across the full
250-day walk-forward AND two 125-day sub-windows, to see which config is robust
(not just best on one window) and whether trimming high-turnover overlays helps.
"""
import warnings, importlib
warnings.filterwarnings("ignore")
import numpy as np
import final_strategy as fs
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section)

P, df, tickers = prices_array()
N, T = P.shape
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0


def backtest(weights, start, end):
    fs.W_PAIRS, fs.W_ALGO, fs.W_CORR, fs.W_LEAD, fs.W_XS = weights
    fs._pairs = {"day": -10**9, "list": []}; fs._pair_state = {}
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    for t in range(start, end + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < end:
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(fs.getMyPosition(hist), -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > start: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sr**2 / (sr**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return sr, score, totDVol

FULL, H1, H2 = (T-250, T), (T-250, T-125), (T-125, T)
variants = {
    "full stack":            (1.5, 2.0, 0.5, 1.0, 0.5),
    "no lead-lag/xs (lean)": (1.5, 2.0, 0.5, 0.0, 0.0),
    "pairs+ALGO only":       (1.5, 2.0, 0.0, 0.0, 0.0),
    "pairs+ALGO+corr":       (1.5, 2.0, 1.0, 0.0, 0.0),
    "equal weights":         (1.0, 1.0, 1.0, 1.0, 1.0),
    "pairs-heavy":           (2.0, 1.5, 0.5, 0.5, 0.3),
}
section("25. FINAL-BOOK ROBUSTNESS: weight variants x windows (Sharpe / Score)")
print(f"{'variant':<24}{'FULL 250d':>18}{'first 125d':>18}{'last 125d':>18}{'$vol(M)':>10}")
for name, w in variants.items():
    sF, scF, dv = backtest(w, *FULL)
    s1, sc1, _ = backtest(w, *H1)
    s2, sc2, _ = backtest(w, *H2)
    print(f"{name:<24}{sF:>7.2f}/{scF:>8.1f}{s1:>7.2f}/{sc1:>8.1f}"
          f"{s2:>7.2f}/{sc2:>8.1f}{dv/1e6:>10.1f}")

section("VERDICT")
print("Pick the variant with the best WORST-window score (robust), not the best")
print("full-sample score. Lower $volume = less commission drag = more durable.")
