"""Search for high-scoring configs (targeting score ~500+) and report each.

Main lever: dollar sizing toward the $10k/name limit + more pairs. We backtest
with exact eval.py logic on the last 250 days and also a strict last-125 window.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from strat_engine import Engine, cfg
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section)

P, T = prices_array()[0], None
P = prices_array()[0]
N, T = P.shape
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0


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
    return sr, score, mu, totDVol / 1e6


FULL = (T - 250, T); LAST = (T - 125, T)

section("26. HIGH-SCORE CONFIG SEARCH (Sharpe / Score / mean$ / $vol[M])")
candidates = {
    # scale pair sizing up, more pairs
    "P: 16pairs $8k":      cfg(w_pairs=1, max_pairs=16, pair_dollars=8000),
    "P: 30pairs $10k":     cfg(w_pairs=1, max_pairs=30, pair_dollars=10000, pmax=0.05),
    "P: 40pairs $10k p.10":cfg(w_pairs=1, max_pairs=40, pair_dollars=10000, pmax=0.10),
    "P: 30pairs $10k x2":  cfg(w_pairs=2, max_pairs=30, pair_dollars=10000),
    # pairs + ALGO timing bigger
    "P30+ALGO$100k":       cfg(w_pairs=1.5, w_algo=1.0, max_pairs=30, pair_dollars=10000, algo_dollars=100000),
    # lean stack scaled
    "lean x2":             cfg(w_pairs=3, w_algo=4, w_corr=1, max_pairs=30, pair_dollars=10000,
                               algo_dollars=100000, corr_dollars=6000),
    # full stack scaled
    "full scaled":         cfg(w_pairs=3, w_algo=4, w_corr=1, w_lead=2, w_xs=1, max_pairs=30,
                               pair_dollars=10000, algo_dollars=100000, corr_dollars=6000,
                               lead_dollars=5000, xs_dollars=5000),
    # multi-factor residual heavy
    "MF+pairs":            cfg(w_pairs=1.5, w_mf=1.0, max_pairs=30, pair_dollars=10000, mf_dollars=8000, mf_k=3),
}
print(f"{'config':<24}{'FULL 250d':>22}{'last 125d':>18}")
for name, c in candidates.items():
    sF, scF, muF, dvF = backtest(c, *FULL)
    sL, scL, muL, dvL = backtest(c, *LAST)
    print(f"{name:<24}{sF:>6.2f}/{scF:>7.0f}/${muF:>5.0f}/{dvF:>4.0f}M{sL:>7.2f}/{scL:>7.0f}")
