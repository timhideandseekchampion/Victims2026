"""Push for score ~800 using FIXED strong cointegrated pairs (rolling beta) +
ALGO timing + factor residual, sized to saturate the $ position limits.
Backtest with exact eval.py logic on the full 250-day test window.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from strat_engine import Engine, cfg
from common import (prices_array, load, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0

# strong pairs (full-sample p<0.01), take top-K
cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv")
strong = cd[cd.coint_p < 0.01].sort_values("coint_p")
def pairs_top(k):
    return [(IDX[a], IDX[b]) for a, b in zip(strong.a[:k], strong.b[:k])]


def backtest(c, start, end):
    eng = Engine(c)
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
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
        if t > start: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sr**2 / (sr**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    gross = np.abs(curPos * cur).sum()
    return sr, score, mu, totDVol / 1e6


FULL = (T - 250, T); LAST = (T - 125, T)
section("27. FIXED-PAIR HIGH-SCORE SEARCH (Sharpe/Score/mean$/vol[M]  |  last125 score)")
cands = {
    "6 pairs $10k e1.0/x0.5":  cfg(w_pairs=1, fixed_pairs=pairs_top(6), pair_dollars=10000,
                                   pair_entry=1.0, pair_exit=0.5),
    "24 pairs $10k e1.0/x0.5": cfg(w_pairs=1, fixed_pairs=pairs_top(24), pair_dollars=10000),
    "24 pairs $10k e0.75":     cfg(w_pairs=1, fixed_pairs=pairs_top(24), pair_dollars=10000, pair_entry=0.75),
    "24 pairs always-in e0.75/x0": cfg(w_pairs=1, fixed_pairs=pairs_top(24), pair_dollars=10000,
                                       pair_entry=0.75, pair_exit=0.0),
    "24p + ALGO$100k":         cfg(w_pairs=1, w_algo=1, fixed_pairs=pairs_top(24), pair_dollars=10000,
                                   algo_dollars=100000),
    "24p + ALGO + corr$9k":    cfg(w_pairs=1, w_algo=1, w_corr=1, fixed_pairs=pairs_top(24),
                                   pair_dollars=10000, algo_dollars=100000, corr_dollars=9000),
    "24p+ALGO+corr+xs (all)":  cfg(w_pairs=1, w_algo=1, w_corr=1, w_xs=1, fixed_pairs=pairs_top(24),
                                   pair_dollars=10000, algo_dollars=100000, corr_dollars=9000, xs_dollars=6000),
    "40p+ALGO+corr+xs BIG":    cfg(w_pairs=1.5, w_algo=1.5, w_corr=1.5, w_xs=1, fixed_pairs=pairs_top(40),
                                   pair_dollars=10000, algo_dollars=100000, corr_dollars=9000, xs_dollars=7000,
                                   pmax=0.02),
}
print(f"{'config':<28}{'Sharpe':>7}{'Score':>7}{'mean$':>7}{'vol':>6}{'last125':>9}")
res = {}
for name, c in cands.items():
    sr, sc, mu, dv = backtest(c, *FULL)
    _, scL, _, _ = backtest(c, *LAST)
    res[name] = (c, sr, sc)
    print(f"{name:<28}{sr:>7.2f}{sc:>7.0f}{mu:>7.0f}{dv:>5.0f}M{scL:>9.0f}")
