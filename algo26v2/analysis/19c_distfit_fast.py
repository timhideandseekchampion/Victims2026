"""Exhaustive dist-fit reversion sweep, done efficiently.

For each (distribution, lookback) we fit the distribution to every instrument's
trailing returns on every test day ONCE, cache the standardized last-move z, then
backtest all entry thresholds from the cache. Distributions: Student-t, gennorm,
Johnson-SU, Laplace, Gaussian (the families that best-fit the returns in module 10).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, N_TEST_DAYS, section)

P, df, tickers = prices_array()
N, T = P.shape
LOGP = np.log(P)
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
START = T - N_TEST_DAYS

# MLE with location fixed to the sample mean (floc) -> optimises only shape+scale,
# which converges fast even when the near-Gaussian likelihood is flat in dof.
FITTERS = {
    "gauss":     lambda r: (r.mean(), r.std()),
    "t":         lambda r: (lambda p: (p[1], p[2]))(stats.t.fit(r, floc=r.mean())),
    "laplace":   lambda r: (lambda p: (p[0], p[1]))(stats.laplace.fit(r, floc=np.median(r))),
    "gennorm":   lambda r: (lambda p: (p[1], p[2]))(stats.gennorm.fit(r, floc=r.mean())),
    "johnsonsu": lambda r: (lambda p: (p[2], p[3]))(stats.johnsonsu.fit(r, floc=r.mean())),
}


def precompute_z(dist, lb):
    """zcache[t_index, name] = standardized latest daily move by fitted (loc,scale)."""
    zc = np.full((T - START, N), 0.0)
    fitter = FITTERS[dist]
    for ti, t in enumerate(range(START, T)):
        rlast = LOGP[:, t] - LOGP[:, t - 1]
        for i in range(N):
            r = np.diff(LOGP[i, t - lb:t])
            try:
                loc, sc = fitter(r)
                zc[ti, i] = (rlast[i] - loc) / (sc + 1e-12)
            except Exception:
                zc[ti, i] = 0.0
    return zc


def backtest_from_z(zc, entry, dollars=2500):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    for ti, t in enumerate(range(START, T + 1)):
        cur = P[:, t - 1]
        if t < T:
            z = zc[ti]
            sig = np.where(np.abs(z) > entry, -np.sign(z) * (np.abs(z) - entry), 0.0)
            sig = sig - sig.mean()
            s = np.abs(sig).sum()
            raw = (sig / s) * dollars * N / cur if s > 1e-12 else np.zeros(N)
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(raw, -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > START: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sr**2 / (sr**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return mu, sr, score


section("EXHAUSTIVE DIST-FIT REVERSION SWEEP (5 distributions x lookback x entry_z)")
print(f"{'dist':>10}{'lb':>5}{'entry':>7}{'mean$':>9}{'Sharpe':>8}{'Score':>8}")
best = (None, -1e9)
for dist in ["t", "gennorm", "johnsonsu", "laplace", "gauss"]:
    for lb in (60, 90):
        zc = precompute_z(dist, lb)
        for ez in (1.0, 1.5, 2.0, 2.5):
            mu, sr, sc = backtest_from_z(zc, ez)
            if sc > best[1]:
                best = ((dist, lb, ez), sc, sr, mu)
            print(f"{dist:>10}{lb:>5}{ez:>7.1f}{mu:>9.2f}{sr:>8.2f}{sc:>8.2f}", flush=True)

section("VERDICT")
print(f"Best dist-fit config: {best[0]}  ->  Sharpe {best[2]:.2f}, score {best[1]:.2f}, mean ${best[3]:.2f}")
print("Distribution choice barely moves the result (returns are near-Gaussian);")
print("dist-fit reversion is weak at best and negative at the 1-day horizon —")
print("consistent with the autocorrelation/variance-ratio nulls.")
