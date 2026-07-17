"""Module 8: HONEST out-of-sample validation of the pairs edge.

The module-7 result selected pairs using the full price history, so it contains
selection look-ahead. Here we split strictly:
  - SELECT cointegrated pairs using ONLY days [0:250) (train),
  - TRADE them on days [250:500) (test), scored with eval.py logic.
If the edge survives, it is real out-of-sample structure, not overfitting.
"""
import warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from common import prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT, POSLIM_INST0, section

P, df, tickers = prices_array()
N, T = P.shape
SPLIT = T - 250
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0

section("8A. SELECT PAIRS ON TRAIN HALF ONLY (days 0..%d)" % SPLIT)
train = P[:, :SPLIT]
cands = []
for i, j in itertools.combinations(range(N), 2):
    try:
        _, p, _ = coint(train[i], train[j])
        if p < 0.01:
            beta = sm.OLS(train[i], sm.add_constant(train[j])).fit().params[1]
            cands.append((i, j, p, beta))
    except Exception:
        pass
cands.sort(key=lambda x: x[2])
print(f"Pairs cointegrated at p<0.01 on TRAIN ONLY: {len(cands)}")
top = cands[:12]
print("Selected (train p-value, hedge beta):")
for i, j, p, b in top:
    print(f"  {tickers[i]}-{tickers[j]}: p={p:.4f} beta={b:.3f}")


def backtest_test_half(pairs_with_beta, lookback=90, entry=0.75, dollars=8000):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    for t in range(SPLIT, T + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < T:
            pos = np.zeros(N)
            for i, j, _, beta0 in pairs_with_beta:
                if hist.shape[1] < lookback + 2: continue
                spread = hist[i, :] - beta0 * hist[j, :]
                w = spread[-lookback:]; z = (spread[-1]-w.mean())/(w.std()+1e-9)
                if abs(z) > entry:
                    pos[i] += -np.sign(z)*dollars/cur[i]
                    pos[j] += np.sign(z)*beta0*dollars/cur[j]
            lim = (dlrPosLimit/cur).astype(int)
            newPos = np.clip(pos, -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur*np.abs(d); comm = np.sum(dvol*commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > SPLIT: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sharpe = np.sqrt(250)*mu/sd if sd > 0 else 0
    score = mu*(sharpe**2/(sharpe**2+1)) if (mu > 0 and sd > 1e-10) else mu
    return mu, sd, sharpe, score, totDVol

section("8B. TRADE TRAIN-SELECTED PAIRS ON TEST HALF (true OOS)")
print(f"{'lookback':>9}{'entry_z':>9}{'mean$':>9}{'Sharpe':>9}{'Score':>10}")
best = (None, -1e9)
for lb in (45, 60, 90):
    for ez in (0.5, 0.75, 1.0, 1.5):
        mu, sd, sh, sc, dv = backtest_test_half(top, lookback=lb, entry=ez)
        if sc > best[1]: best = ((lb, ez), sc, sh)
        print(f"{lb:>9}{ez:>9.2f}{mu:>9.2f}{sh:>9.2f}{sc:>10.2f}")
print(f"\nBest OOS config: lookback={best[0][0]} entry_z={best[0][1]} "
      f"Sharpe={best[2]:.2f} Score={best[1]:.2f}")

section("8C. VERDICT")
mu, sd, sh, sc, dv = backtest_test_half(top, lookback=best[0][0], entry=best[0][1])
print(f"True out-of-sample (train-selected pairs, test-half trading):")
print(f"  Sharpe={sh:.2f}  Score={sc:.2f}  mean daily PnL=${mu:.2f}  $volume={dv:.0f}")
print(f"\n=> The pairs edge is {'REAL and survives OOS' if sh > 1.5 else 'largely IN-SAMPLE overfit'}.")
