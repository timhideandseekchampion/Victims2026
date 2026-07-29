"""
test_batch100_d64_hier_shrink.py

D64: hierarchical / multilevel ridge -- shrink each name's OWN coefficient vector (one of the 50
columns of B returned by the shared closed-form ridge solve) toward the POPULATION-AVERAGE
coefficient vector (mean across all 50 idio names), rather than only toward zero (which is all the
existing ridge penalty `a` does). This is a simple empirical-Bayes / multilevel-model style shrinkage
applied post-hoc to each half-life's fitted B: B_j <- (1-LAMBDA_H)*B_j + LAMBDA_H*mean_j(B_j).

Distinct from the already-tried "per-half-life differential shrinkage" (test_v11cand_predictor_shrink,
which shrinks the ridge_a strength differently ACROSS half-lives) -- this shrinks ACROSS NAMES within
one half-life's fit, toward a common population coefficient vector, at a fixed half-life.

Everything else (BLEND reversal, pairwise boost, rank-stability blend, ALGO leg) is reused verbatim
from SAFE_llboost_v10, following the test_v20cand_idio_deadband.py house convention.
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

print("=== precompute: REV blend leg, pairwise boost, rank-stability signal, ALGO leg -- all "
      "UNCHANGED / reused verbatim from V10 (independent of the ridge-shrinkage mechanism under test) ===",
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
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def combine_wz(wz_ridge, t):
    wz = (1 - V10.BLEND) * wz_ridge + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_pos(WZ_RIDGE):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = combine_wz(WZ_RIDGE[:, t], t)
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, WZ_RIDGE, base_wo=None, base_wn=None, base_scs=None, verbose=True):
    Pz = build_pos(WZ_RIDGE); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = None
    if base_wo is not None:
        passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = None if base_scs is None else int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ("  <== fail" if passed is False else "")
        extra = f"  n_worse={nworse}/{len(scs)}" if nworse is not None else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}"
              f"{extra}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


def build_wz_ridge(lam_h):
    """lam_h=0.0 must reproduce v10 exactly (no shrinkage toward pop mean). lam_h>0: each half-life's
    B column (per idio name) is pulled toward the mean coefficient vector across all 50 names."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            if lam_h > 0:
                Bbar = B.mean(axis=1, keepdims=True)
                B = (1 - lam_h) * B + lam_h * Bbar
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ


print("\n=== sanity check: LAMBDA_H=0 (mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = build_wz_ridge(0.0)
POS_base = build_pos(WZ_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
    SANITY_OK = False
else:
    print("  OK -- matches v10 to within rounding.")
    SANITY_OK = True


print("\n=== SWEEP: LAMBDA_H (population-mean shrinkage weight on each name's coefficient vector) ===")
results = []
for lam_h in (0.1, 0.2, 0.3, 0.5):
    t0 = time.time()
    WZ_H = build_wz_ridge(lam_h)
    c = evaluate(f"LAMBDA_H={lam_h}", WZ_H, base_wo, base_wn, base_scs)
    results.append(c)

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} LAMBDA_H configs beat v10 on OLD+NEW+rmean jointly.")
if not passing:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
