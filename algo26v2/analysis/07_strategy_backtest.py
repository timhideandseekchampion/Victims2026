"""Module 7: Turn the statistical findings into strategies and score them with
the EXACT eval.py logic (commissions, integer shares, $ position limits).

Signals discovered:
  S1 cointegrated-pair stat-arb (module 3)
  S2 ALGO lead-lag (Granger, module 6) - trade names ALGO leads
  S3 cross-sectional 1-day reversal (module 4)
  S4 tree-ensemble prediction (module 5)
We compare each against the starter momentum strategy on the last 250 days.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from common import prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT, POSLIM_INST0, N_TEST_DAYS, section

P, df, tickers = prices_array()          # P: N x T
N, T = P.shape
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0


def backtest(get_pos, numTestDays=N_TEST_DAYS):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0
    pll = []
    startDay = T - numTestDays
    for t in range(startDay, T + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < T:
            newOrig = get_pos(hist)
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(newOrig, -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos
        cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos)
        pv = curPos.dot(cur); todayPL = cash + pv - value; value = cash + pv
        if t > startDay:
            pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sharpe = np.sqrt(250) * mu / sd if sd > 0 else 0
    if mu <= 0 or sd < 1e-10:
        scoreVal = mu
    else:
        sr = sharpe; scoreVal = mu * (sr**2 / (sr**2 + 1.0))
    return dict(mean=mu, std=sd, sharpe=sharpe, score=scoreVal, dvol=totDVol)


# ---- strategies -------------------------------------------------------------
def starter(hist):
    ni, nt = hist.shape
    if nt < 2: return np.zeros(ni)
    lr = np.log(hist[:, -1] / hist[:, -2]); ln = np.sqrt(lr.dot(lr)); lr /= ln
    if not hasattr(starter, "pos"): starter.pos = np.zeros(ni)
    starter.pos = np.array([int(x) for x in starter.pos + 5000 * lr / hist[:, -1]])
    return starter.pos


PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("SMAH","ILVX"),("HUXZ","ACAC"),
         ("HETT","ULXY"),("CTGI","EELT"),("ACIX","ITPA"),("RTTH","NAYO")]
IDX = {t: i for i, t in enumerate(tickers)}
def pairs_strat(hist, lookback=60, entry=1.0, dollars=8000):
    ni, nt = hist.shape; pos = np.zeros(ni)
    if nt < lookback + 2: return pos
    for a, b in PAIRS:
        ia, ib = IDX[a], IDX[b]
        pa = hist[ia, -lookback:]; pb = hist[ib, -lookback:]
        beta = np.polyfit(pb, pa, 1)[0]
        spread = hist[ia, :] - beta * hist[ib, :]
        w = spread[-lookback:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            sh_a = -np.sign(z) * dollars / hist[ia, -1]
            sh_b = np.sign(z) * beta * dollars / hist[ib, -1]
            pos[ia] += sh_a; pos[ib] += sh_b
    return pos.astype(int)


LEADNAMES = ["CUBO","HRND","ULXY","GARI","HRET","MHRM","LSST","ANSO","ELLT"]
def leadlag(hist, dollars=6000):
    ni, nt = hist.shape; pos = np.zeros(ni)
    if nt < 2: return pos
    algo_ret = np.log(hist[0, -1] / hist[0, -2])
    for nm in LEADNAMES:
        i = IDX[nm]
        pos[i] = np.sign(algo_ret) * dollars / hist[i, -1]   # follow ALGO move
    return pos.astype(int)


def xs_reversal(hist, dollars=3000):
    ni, nt = hist.shape
    if nt < 2: return np.zeros(ni)
    r = np.log(hist[:, -1] / hist[:, -2]); r = r - r.mean()
    w = -r / (np.abs(r).sum() + 1e-9)
    return (w * dollars * ni / hist[:, -1]).astype(int)


section("STRATEGY SCORES ON LAST 250 DAYS (eval.py scoring)")
print(f"{'strategy':<22}{'mean$':>10}{'std$':>10}{'Sharpe':>9}{'Score':>10}{'$volume':>13}")
for name, fn in [("starter(momentum)", starter), ("cointegration-pairs", pairs_strat),
                 ("ALGO-leadlag", leadlag), ("xs-reversal", xs_reversal)]:
    if hasattr(starter, "pos"): del starter.pos
    r = backtest(fn)
    print(f"{name:<22}{r['mean']:>10.2f}{r['std']:>10.2f}{r['sharpe']:>9.2f}"
          f"{r['score']:>10.2f}{r['dvol']:>13.0f}")

section("PAIRS PARAMETER SWEEP (lookback x entry-z)")
print(f"{'lookback':>9}{'entry_z':>9}{'Sharpe':>9}{'Score':>10}")
best = (None, -1e9)
for lb in (30, 45, 60, 90):
    for ez in (0.75, 1.0, 1.5, 2.0):
        r = backtest(lambda h, lb=lb, ez=ez: pairs_strat(h, lookback=lb, entry=ez))
        if r["score"] > best[1]: best = ((lb, ez), r["score"])
        print(f"{lb:>9}{ez:>9.2f}{r['sharpe']:>9.2f}{r['score']:>10.2f}")
print(f"\nBest pairs config: lookback={best[0][0]}, entry_z={best[0][1]}, score={best[1]:.2f}")
