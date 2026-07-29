"""
precompute_batch100.py

Shared expensive precompute for the batch100 (F73-F80) test scripts. Computes, once, everything
that does NOT depend on the specific idea being tested (the ridge ensemble per half-life, the BLEND
reversion leg, the pairwise boost, the shipped rank-stability signal, and the ALGO leg), and saves it
to batch100_cache.npz. Each test_batch100_<id>.py script loads this cache instead of recomputing the
expensive ridge-ensemble loop from scratch, per the house convention (see test_v19cand_boost_ncandidates.py).

Sanity check target (SAFE_llboost_v10 docstring): OLD=871.0 NEW=912.6 rmean=909.8 rfloor=709.7
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W

days = list(range(WARMUP, nt))
n_hl = len(HALF_LIVES)

print("=== precompute: per-half-life ridge forecasts (FS), REV leg, BOOST, ALGO leg, RS signal ===", flush=True)
t0 = time.time()

FS = np.full((n_hl, nIdio, nt), np.nan)      # per half-life z-scored forecast, BEFORE averaging/BLEND
REV = np.zeros((nIdio, nt))
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    for hi, hl in enumerate(HALF_LIVES):
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        FS[hi, :, t] = fi / (fi.std() + 1e-12)
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)
print(f"  ridge ensemble + REV done ({time.time()-t0:.0f}s)", flush=True)

WZ_RIDGE = np.nanmean(FS, axis=0)                                    # equal-weight avg across half-lives
WZ_PRE = np.zeros((nIdio, nt))
for t in days:
    WZ_PRE[:, t] = (1 - V10.BLEND) * WZ_RIDGE[:, t] + V10.BLEND * REV[:, t]

t0 = time.time()
BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])
print(f"  boost done ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  algo leg done ({time.time()-t0:.0f}s)", flush=True)

# shipped rank-stability raw signal (short=8, long=22), RAW (not yet z-scored/blended)
RS_RAW = np.full((nIdio, nt), np.nan)
for t in days:
    if t < max(RS_SHORT_W, RS_LONG_W) + 5:
        continue
    short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
    long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        continue
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    RS_RAW[:, t] = np.where(disagree, -sz, 0.0)

# full v10 baseline wz (ridge+REV blend, + boost, + RS blend) for sanity check
WZ_V10 = WZ_PRE.copy()
for t in days:
    wz = WZ_V10[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_RAW[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz

np.savez_compressed(
    "batch100_cache.npz",
    FS=FS, REV=REV, WZ_RIDGE=WZ_RIDGE, WZ_PRE=WZ_PRE, BOOST=BOOST, algo_pos=algo_pos,
    RS_RAW=RS_RAW, WZ_V10=WZ_V10, days=np.array(days),
)
print("saved batch100_cache.npz")
