"""Strategies built directly from three test families, backtested with the
EXACT eval.py logic (commissions, integer shares, $ position limits).

  S1  CORRELATION vs ALGO  - rolling OLS of each name's price on ALGO's price;
      trade the residual (idiosyncratic) z-score, hedge the factor leg with ALGO.
      (ALGO is the market factor + has 5x cheaper commission & 10x bigger limit.)
  S2  UNIT-ROOT gated      - same residual, but only trade a name when a LIVE
      ADF test says its residual-vs-ALGO is currently stationary (mean-reverting).
  S3  DIST-FIT reversion   - fit a distribution to each name's recent returns,
      fade extreme standardized daily moves (cross-sectional, market neutral),
      sized by fitted scale.
  S4  ENSEMBLE             - S1+S2+S3 combined.
We compare against the starter and the cointegration-pairs benchmark.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, N_TEST_DAYS, section)

P, df, tickers = prices_array()
N, T = P.shape
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
ALGO = 0


def backtest(get_pos, numTestDays=N_TEST_DAYS):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    start = T - numTestDays
    for t in range(start, T + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < T:
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(get_pos(hist), -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > start: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sharpe = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sharpe**2 / (sharpe**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return dict(mean=mu, std=sd, sharpe=sharpe, score=score, dvol=totDVol)


# ---- S1: correlation-vs-ALGO residual reversion ---------------------------
def s1_corr_algo(hist, lookback=60, entry=1.0, dollars=5000):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lookback + 2: return pos
    la = np.log(hist[ALGO, -lookback:]); cur = hist[:, -1]
    algo_leg = 0.0
    for i in range(1, n):
        li = np.log(hist[i, -lookback:])
        beta = np.polyfit(la, li, 1)[0]
        resid = np.log(hist[i, :]) - beta * np.log(hist[ALGO, :])
        w = resid[-lookback:]; z = (resid[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            sh = -np.sign(z) * dollars / cur[i]
            pos[i] += sh
            algo_leg += -sh * beta * cur[i] / cur[ALGO]  # hedge factor exposure
    pos[ALGO] += algo_leg
    return pos.astype(int)


# ---- S2: unit-root (ADF) gated residual reversion --------------------------
def s2_unitroot(hist, lookback=60, entry=1.0, dollars=5000, adf_p=0.10):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lookback + 2: return pos
    cur = hist[:, -1]; la = np.log(hist[ALGO, -lookback:]); algo_leg = 0.0
    for i in range(1, n):
        resid = np.log(hist[i, -lookback:]) - np.polyfit(la, np.log(hist[i, -lookback:]), 1)[0] * la
        try:
            if adfuller(resid, maxlag=4, autolag=None)[1] > adf_p:
                continue  # not stationary -> skip (this is the unit-root gate)
        except Exception:
            continue
        z = (resid[-1] - resid.mean()) / (resid.std() + 1e-9)
        if abs(z) > entry:
            beta = np.polyfit(la, np.log(hist[i, -lookback:]), 1)[0]
            sh = -np.sign(z) * dollars / cur[i]; pos[i] += sh
            algo_leg += -sh * beta * cur[i] / cur[ALGO]
    pos[ALGO] += algo_leg
    return pos.astype(int)


# ---- S3: dist-fit distributional reversion ---------------------------------
def s3_distfit(hist, lookback=60, entry=1.5, dollars=2500, dist="gauss"):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lookback + 2: return pos
    cur = hist[:, -1]; rlast = np.log(hist[:, -1] / hist[:, -2])
    signals = np.zeros(n)
    for i in range(n):
        r = np.diff(np.log(hist[i, -lookback:]))
        try:
            if dist == "t":
                dof, loc, sc = stats.t.fit(r)
                zq = (rlast[i] - loc) / sc  # standardized by fitted scale
            else:
                loc, sc = r.mean(), r.std()
                zq = (rlast[i] - loc) / (sc + 1e-12)
        except Exception:
            zq = 0.0
        if abs(zq) > entry:
            signals[i] = -np.sign(zq) * (abs(zq) - entry)  # fade extreme move
    if np.abs(signals).sum() > 0:
        signals -= signals.mean()  # market neutral
        w = signals / (np.abs(signals).sum() + 1e-9)
        pos = (w * dollars * n / cur).astype(int)
    return pos.astype(int)


# ---- benchmarks -----------------------------------------------------------
def starter(hist):
    ni, nt = hist.shape
    if nt < 2: return np.zeros(ni)
    lr = np.log(hist[:, -1] / hist[:, -2]); lr /= np.sqrt(lr.dot(lr))
    if not hasattr(starter, "pos"): starter.pos = np.zeros(ni)
    starter.pos = np.array([int(x) for x in starter.pos + 5000 * lr / hist[:, -1]])
    return starter.pos

PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("HETT","ULXY"),("SMAH","ILVX"),
         ("HUXZ","ACAC"),("CTGI","EELT")]
IDX = {t: i for i, t in enumerate(tickers)}
def coint_pairs(hist, lookback=90, entry=0.75, dollars=8000):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lookback + 2: return pos
    for a, b in PAIRS:
        ia, ib = IDX[a], IDX[b]
        beta = np.polyfit(hist[ib, -lookback:], hist[ia, -lookback:], 1)[0]
        spread = hist[ia, :] - beta * hist[ib, :]
        w = spread[-lookback:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            pos[ia] += -np.sign(z) * dollars / hist[ia, -1]
            pos[ib] += np.sign(z) * beta * dollars / hist[ib, -1]
    return pos.astype(int)


def ensemble(hist):
    return (s1_corr_algo(hist) + s2_unitroot(hist) + s3_distfit(hist)
            + coint_pairs(hist)).astype(int)


section("STRATEGIES FROM TEST FAMILIES - scored on last 250 days (eval.py logic)")
print(f"{'strategy':<28}{'family':<16}{'mean$':>9}{'Sharpe':>8}{'Score':>9}{'$vol':>12}")
strategies = [
    ("starter", "momentum(ref)", starter),
    ("S1 corr-vs-ALGO", "correlation", s1_corr_algo),
    ("S2 unit-root gated", "unit-root", s2_unitroot),
    ("S3 dist-fit reversion", "dist-fit", s3_distfit),
    ("coint-pairs", "cointegration(ref)", coint_pairs),
    ("S4 ensemble", "all", ensemble),
]
for name, fam, fn in strategies:
    if hasattr(starter, "pos"): del starter.pos
    r = backtest(fn)
    print(f"{name:<28}{fam:<16}{r['mean']:>9.2f}{r['sharpe']:>8.2f}{r['score']:>9.2f}{r['dvol']:>12.0f}")

section("QUICK SWEEPS (S1 entry_z, S3 entry_z)")
print("S1 correlation-vs-ALGO (lookback x entry_z):")
print(f"{'lb':>5}{'ez':>6}{'Sharpe':>8}{'Score':>8}")
for lb in (40, 60, 90):
    for ez in (0.75, 1.0, 1.5):
        r = backtest(lambda h, lb=lb, ez=ez: s1_corr_algo(h, lookback=lb, entry=ez))
        print(f"{lb:>5}{ez:>6.2f}{r['sharpe']:>8.2f}{r['score']:>8.2f}")

print("\nS3 dist-fit gauss (entry_z):")
print(f"{'ez':>6}{'Sharpe':>8}{'Score':>8}")
for ez in (1.0, 1.5, 2.0):
    r = backtest(lambda h, ez=ez: s3_distfit(h, entry=ez, dist="gauss"))
    print(f"{ez:>6.1f}{r['sharpe']:>8.2f}{r['score']:>8.2f}")
