"""
test_ic_stationarity_formal.py

Follow-up to the "is the idio book's IC stationary?" claim in test_v7cand_double_ic_diag.py, which
was based on informal summary stats (mean/sd/range/negative-day-count), not a formal test. This runs
actual unit-root/stationarity tests (statsmodels: ADF, null = unit root i.e. non-stationary; KPSS,
null = stationary -- the complementary test, since "ADF fails to reject" is NOT the same evidence as
"KPSS confirms stationary") on:

  1. the ALGO leg's trailing IC series (fast 90d simple IC of volz vs next-day ALGO return) --
     the series the earlier diagnostic called genuinely non-stationary (sd 0.101, 22% negative days,
     13 zero-crossings).
  2. the idio book's pooled trailing IC series (fast 90d simple IC of wz vs next-day idio return,
     pooled across all 50 names) -- the series called stationary (sd 0.0255, never negative).
  3. for contrast/calibration: the raw log-price series of ALGO (instrument 0) -- expected to fail
     both tests decisively (a random walk is the textbook non-stationary case), to confirm the tests
     themselves are behaving sanely on this data before trusting their verdict on the IC series.

Both ADF and KPSS run with the 'constant, no trend' specification (testing stationarity around a
level, not a deterministic trend -- appropriate for a bounded correlation coefficient), automatic lag
selection (ADF: AIC; KPSS: default short-lag Newey-West per statsmodels' 'auto').
"""
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP = V7.WARMUP


def ew_weights(m, hl):
    return (0.5 ** (1.0 / hl)) ** np.arange(m - 1, -1, -1)


def wcorr(x, y, w, min_n=60):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n: return np.nan
    x, y, w = x[ok], y[ok], w[ok]; sw = w.sum()
    mx = (w * x).sum() / sw; my = (w * y).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw; vy = (w * (y - my) ** 2).sum() / sw
    if vx < 1e-24 or vy < 1e-24: return np.nan
    return float((w * (x - mx) * (y - my)).sum() / sw / np.sqrt(vx * vy))


def report(name, series):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    print(f"\n--- {name} (n={len(s)}) ---")
    print(f"  mean={s.mean():+.4f}  sd={s.std():.4f}  min={s.min():+.4f}  max={s.max():+.4f}")
    adf_stat, adf_p, adf_lags, adf_n, adf_crit, _ = adfuller(s, regression="c", autolag="AIC")
    print(f"  ADF  (H0: unit root / non-stationary):  stat={adf_stat:+.3f}  p={adf_p:.4g}  "
          f"lags={adf_lags}  crit(1%/5%/10%)={adf_crit['1%']:.2f}/{adf_crit['5%']:.2f}/{adf_crit['10%']:.2f}")
    print(f"       -> {'REJECT unit root (stationary)' if adf_p < 0.05 else 'CANNOT reject unit root'}")
    kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")
    print(f"  KPSS (H0: stationary):                   stat={kpss_stat:.3f}  p={kpss_p:.4g}  "
          f"lags={kpss_lags}  crit(1%/5%/10%)={kpss_crit['1%']:.3f}/{kpss_crit['5%']:.3f}/{kpss_crit['10%']:.3f}")
    print(f"       -> {'REJECT stationarity' if kpss_p < 0.05 else 'CANNOT reject stationarity'}")
    verdict = ("STATIONARY (ADF rejects unit root, KPSS does not reject stationarity)"
               if (adf_p < 0.05 and kpss_p >= 0.05) else
               "NON-STATIONARY (KPSS rejects stationarity, ADF does not reject unit root)"
               if (adf_p >= 0.05 and kpss_p < 0.05) else
               "AMBIGUOUS / borderline (tests disagree)")
    print(f"  VERDICT: {verdict}")
    return dict(adf_p=adf_p, kpss_p=kpss_p, verdict=verdict)


# ==================================================================================================
# 0) calibration: does a known-non-stationary series (log-price, a random walk) fail as expected?
# ==================================================================================================
print("=== 0) CALIBRATION: ALGO log-price (expected: decisively non-stationary) ===")
report("ALGO log-price level", logp[0])
print("\n(and its first difference -- returns -- expected: decisively stationary, the textbook contrast)")
report("ALGO log-return (first difference)", r[0])

# ==================================================================================================
# 1) ALGO leg's trailing IC series (fast 90d simple IC, volz vs next-day return)
# ==================================================================================================
print("\n" + "=" * 96)
print("1) ALGO leg: trailing IC series (fast 90d simple IC, matches _side's IC_FAST estimator)")
print("=" * 96)
lpA = logp[0]
algo_ic = []
for tnow in range(200, nt):
    T = tnow + 1
    if T < V7.VOL_WIN + V7.VOL_Z + 60:
        continue
    lp = lpA[:T]
    rr = np.diff(lp)
    vol = np.full(T, np.nan); vol[V7.VOL_WIN:] = V7._roll_std(rr, V7.VOL_WIN)
    lo = max(V7.VOL_WIN + V7.VOL_Z, T - 1 - V7.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V7.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lp[1:] - lp[:-1]
    tn = T - 1
    a = max(0, tn - V7.IC_FAST)
    algo_ic.append(wcorr(volz[a:tn], ret1[a:tn], np.ones(tn - a)))
report("ALGO trailing IC (fast 90d)", algo_ic)

# ==================================================================================================
# 2) idio book's pooled trailing IC series (fast 90d simple IC, wz vs next-day idio return)
# ==================================================================================================
print("\n" + "=" * 96)
print("2) idio book: pooled trailing IC series (fast 90d simple IC, wz vs next-day idio return)")
print("=" * 96)
WZ = np.full((nIdio, nt), np.nan)
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V7.HALF_LIVES:
        B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    rv_ = logp[1:, t] - logp[1:, t - V7.REV_W]
    rv_ = rv_ - rv_.mean()
    WZ[:, t] = (1 - V7.BLEND) * wz + V7.BLEND * (-rv_ / (rv_.std() + 1e-12))

FAST_L = 90
idio_ic = []
for k in range(WARMUP + 200, nt):
    lo_f = max(WARMUP, k - FAST_L)
    X = WZ[:, lo_f:k].ravel(); Y = rs[:, lo_f:k].ravel()
    idio_ic.append(wcorr(X, Y, np.ones(X.size)))
report("idio pooled trailing IC (fast 90d)", idio_ic)

print("\n" + "=" * 96)
print("Summary")
print("=" * 96)
print("If ALGO's IC comes back non-stationary and idio's comes back stationary (or at least far less")
print("non-stationary), that upgrades the earlier informal claim to a formally tested one. If not,")
print("the earlier language should be walked back.")
