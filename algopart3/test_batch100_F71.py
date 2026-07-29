"""
test_batch100_F71.py

F71: cross-sectional DECILE momentum long/short overlay -- buy top-decile trailing-return idio
names, short bottom-decile -- as an ADDITIVE tilt on top of v10, distinct from the pairwise
lead-lag boost (which trades off a name's *leader's* move, not the name's own trailing return
rank) and distinct from the rank-stability blend (which fades a short-term pullback WITHIN a
medium-term trend, not a plain cross-sectional momentum rank).

CRITICAL SIZING FACT (README): v10's idio positions are `sign(wz) * dlr` -- pure sign-based, full
conviction. A tilt only matters if it's blended INTO wz (and is large enough to flip signs near
wz~0) before the sign is taken -- exactly how the shipped rank-stability blend works. A tilt
added AFTER sign() (e.g. scaling position size) would have zero effect, the same trap the README
flags for the rejected "dispersion" signal. So this is implemented with the identical blend
mechanic as RS_WEIGHT: decile_z is z-scored, then
  wz = (1-WEIGHT)*wz + WEIGHT*decile_z*(mean(|wz|)+eps)
applied AFTER the RS blend (last, i.e. truly "on top of v10"), for a small (lookback, weight)
grid -- one 3x3 sweep, per the 3-5 point budget for a single free-parameter idea (here two
related knobs, kept small).

Signal definition: for each idio name on each day, rank its trailing LB-day return
cross-sectionally; top decile -> +1, bottom decile -> -1, else 0 (a genuine decile long/short
overlay, not a continuous z-score momentum tilt -- distinct construction from both existing
mechanisms).
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
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

print("=== precompute: full v10 idio WZ (ridge ensemble + BLEND + boost + rank-stability), "
      "verbatim -- the F71 decile overlay is applied ON TOP of this, plus ALGO leg unaffected ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_V10 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    if t >= max(RS_SHORT_W, RS_LONG_W) + 5:
        short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
        long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
        sz = short_ret - short_ret.mean(); sstd = sz.std()
        lz = long_ret - long_ret.mean(); lstd = lz.std()
        if sstd > 1e-12 and lstd > 1e-12:
            sz = sz / sstd; lz = lz / lstd
            disagree = np.sign(lz) != np.sign(sz)
            rs_sig = np.where(disagree, -sz, 0.0)
            s_std = rs_sig.std()
            s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos_baseline():
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: no overlay must reproduce SAFE_llboost_v10 exactly ===")
POS_base = build_pos_baseline()
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
if not SANITY_OK:
    raise SystemExit("Sanity check failed -- aborting.")


def decile_signal(t, lb):
    """+1 top decile / -1 bottom decile / 0 else, by trailing lb-day idio return, at day t."""
    if t - lb < 0:
        return None
    tr = logp[1:, t] - logp[1:, t - lb]
    ranks = pd.Series(tr).rank(pct=True).values  # 0..1, higher = better trailing return
    sig = np.zeros(nIdio)
    sig[ranks >= 0.9] = 1.0
    sig[ranks <= 0.1] = -1.0
    return sig


def build_pos_decile(lb, weight):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t].copy()
        dsig = decile_signal(t, lb)
        if dsig is not None and np.abs(dsig).sum() > 0:
            dstd = dsig.std()
            d_z = (dsig - dsig.mean()) / (dstd + 1e-12) if dstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - weight) * wz + weight * d_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, lb, weight, verbose=True):
    Pz = build_pos_decile(lb, weight); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<26}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, lb=lb, weight=weight, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                nworse=nworse, passed=passed)


print("\n=== SWEEP: decile-overlay lookback x weight (applied AFTER RS blend, on top of v10) ===")
LBS = [20, 40, 60]
WEIGHTS = [0.01, 0.02, 0.05]
t0 = time.time()
results = []
for lb in LBS:
    for w in WEIGHTS:
        results.append(evaluate(f"lb={lb},w={w}", lb, w))
print(f"  sweep done ({time.time()-t0:.0f}s)")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for c in passing:
        print(f"  {c['name']:<26} rmean={c['rm']:.1f} n_worse={c['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  {c['name']:<26} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

best = max(results, key=lambda c: c["rm"])
print(f"\nBest by rolling mean: {best['name']} (rmean={best['rm']:.1f} vs v10 rmean={base_scs.mean():.1f})")
