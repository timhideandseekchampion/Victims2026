"""Maximize score by deploying to the position limits.

Key fact: scaling every position by k leaves Sharpe unchanged but multiplies
mean PnL (hence score) by k -- until positions clip at the $ limits. So we (a)
add more cointegrated pairs to cover more names, (b) scale $ sizing until names
saturate their $10k / $100k caps. Report score + gross exposure utilisation.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from strat_engine import Engine, cfg
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
MAXGROSS = dlrPosLimit.sum()  # 50*10k + 100k = 600k

cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv").sort_values("coint_p")
def pairs_below(p):
    s = cd[cd.coint_p < p]
    return [(IDX[a], IDX[b]) for a, b in zip(s.a, s.b)]


def backtest(c, start, end):
    eng = Engine(c)
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    gross_util = []
    for t in range(start, end + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < end:
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(eng.position(hist), -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        gross_util.append(np.abs(curPos * cur).sum() / MAXGROSS)
        if t > start: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sr**2 / (sr**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return sr, score, mu, np.mean(gross_util) * 100


FULL = (T - 250, T); LAST = (T - 125, T)
section("28A. PAIR COUNT x SIZING (scale to saturate limits)")
print(f"{'config':<34}{'Sharpe':>7}{'Score':>7}{'mean$':>7}{'gross%':>8}{'last125':>9}")
grid = [
    ("p<.01 (43p) $10k",        cfg(w_pairs=1, fixed_pairs=pairs_below(0.01), pair_dollars=10000)),
    ("p<.01 (43p) $25k",        cfg(w_pairs=1, fixed_pairs=pairs_below(0.01), pair_dollars=25000)),
    ("p<.02 (~70p) $25k",       cfg(w_pairs=1, fixed_pairs=pairs_below(0.02), pair_dollars=25000)),
    ("p<.05 (~190p) $25k",      cfg(w_pairs=1, fixed_pairs=pairs_below(0.05), pair_dollars=25000)),
    ("p<.02 $25k +ALGO",        cfg(w_pairs=1, w_algo=1, fixed_pairs=pairs_below(0.02),
                                    pair_dollars=25000, algo_dollars=100000)),
    ("p<.02 $40k +ALGO",        cfg(w_pairs=1, w_algo=1, fixed_pairs=pairs_below(0.02),
                                    pair_dollars=40000, algo_dollars=100000)),
    ("p<.05 $40k +ALGO",        cfg(w_pairs=1, w_algo=1, fixed_pairs=pairs_below(0.05),
                                    pair_dollars=40000, algo_dollars=100000)),
    ("p<.02 $40k +ALGO+corr",   cfg(w_pairs=1, w_algo=1, w_corr=1, fixed_pairs=pairs_below(0.02),
                                    pair_dollars=40000, algo_dollars=100000, corr_dollars=10000)),
]
best = (None, -1)
for name, c in grid:
    sr, sc, mu, gu = backtest(c, *FULL)
    _, scL, _, _ = backtest(c, *LAST)
    if sc > best[1]: best = (name, sc, sr)
    print(f"{name:<34}{sr:>7.2f}{sc:>7.0f}{mu:>7.0f}{gu:>7.0f}%{scL:>9.0f}")
print(f"\nBest full-sample: {best[0]} -> score {best[1]:.0f} (Sharpe {best[2]:.2f})")
print(f"Max gross allowed: ${MAXGROSS:,.0f}")

section("28B. ENTRY/EXIT REFINEMENT on the best pair set (p<.02, $30k, +ALGO)")
print(f"{'entry':>6}{'exit':>6}{'Sharpe':>8}{'Score':>8}{'mean$':>7}")
for e in (0.5, 0.75, 1.0, 1.25):
    for x in (0.3, 0.5, 0.7):
        c = cfg(w_pairs=1, w_algo=1, fixed_pairs=pairs_below(0.02), pair_dollars=30000,
                algo_dollars=100000, pair_entry=e, pair_exit=x)
        sr, sc, mu, gu = backtest(c, *FULL)
        print(f"{e:>6.2f}{x:>6.2f}{sr:>8.2f}{sc:>8.0f}{mu:>7.0f}")
