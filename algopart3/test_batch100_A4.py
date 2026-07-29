"""
test_batch100_A4.py

IDEA A4: Resweep REV_W (reversal-leg lookback) against v10, e.g. 5, 7, 10, 12, 15, 20.
Shipped value is 10.

REV_W only affects the REV array (`-zscore(logp[1:,t]-logp[1:,t-REV_W])`), a cheap per-day
computation. The (expensive) raw ridge ensemble mean, pairwise boost, rank-stability signal, and
ALGO leg are all independent of REV_W and precomputed ONCE, reused for every candidate value.
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
RIDGE_A, HALF_LIVES, BLEND = V10.RIDGE_A, V10.HALF_LIVES, V10.BLEND
SHIPPED_REVW = V10.REV_W
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W


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

print("=== precompute: raw ridge ensemble mean, pairwise boost, rank-stability signal, ALGO leg -- "
      "all independent of REV_W, cached once ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))

RAW = np.full((nIdio, nt), np.nan)
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
    RAW[:, t] = np.mean(fs, 0)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

RS_SIG = np.full((nIdio, nt), np.nan)
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
    RS_SIG[:, t] = np.where(disagree, -sz, 0.0)
print(f"  done ({time.time()-t0:.1f}s)", flush=True)


def build_pos(rev_w):
    POS = np.zeros((nInst, nt))
    REV = np.zeros((nIdio, nt))
    for t in days:
        rv_ = logp[1:, t] - logp[1:, t - rev_w]
        rv_ = rv_ - rv_.mean()
        REV[:, t] = -rv_ / (rv_.std() + 1e-12)
    for t in days:
        wz = (1 - BLEND) * RAW[:, t] + BLEND * REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print(f"\n=== sanity check: REV_W={SHIPPED_REVW} (shipped) must reproduce SAFE_llboost_v10 exactly ===")
t0 = time.time()
POS_base = build_pos(SHIPPED_REVW)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  REV_W={SHIPPED_REVW}: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.1f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(rev_w, verbose=True):
    Pz = build_pos(rev_w); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  REV_W={rev_w:<6}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
              f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")
    return dict(w=rev_w, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== SWEEP: REV_W (shipped={SHIPPED_REVW}) ===")
SWEEP = [5, 7, 10, 12, 15, 20]
t0 = time.time()
results = [evaluate(w) for w in SWEEP]
print(f"  sweep done ({time.time()-t0:.1f}s)")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} REV_W values beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"Best by rolling mean: REV_W={best['w']} (rmean={best['rm']:.1f} vs shipped rmean={base_scs.mean():.1f})")
print(f"\nFINAL: sanity_check_passed={SANITY_OK}  shipped(OLD={base_wo:.1f},NEW={base_wn:.1f},"
      f"rmean={base_scs.mean():.1f},rfloor={base_scs.min():.1f})")
