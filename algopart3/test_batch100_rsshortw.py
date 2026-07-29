"""
test_batch100_rsshortw.py

Resweep A24: does RS_SHORT_W=8 (the rank-stability signal's short-lookback) still sit at its
shipped SAFE_llboost_v10 optimum, or would a different value do better, given everything else
(idio ridge w/ beta-adjusted target, BLEND reversion, pairwise boost, ALGO leg, RS_LONG_W, RS_WEIGHT)
held fixed at v10?

  A24: RS_SHORT_W in {5, 6, 7, 8(shipped), 10, 12}

RS_SHORT_W lives only in V10._rank_stability_signal, and is cheap to recompute (a single
cross-sectional return + z-score per day) -- so the expensive part (ridge ensemble w/ beta-adjusted
target + BLEND reversion + pairwise boost + ALGO leg) is computed ONCE (giving WZ_BOOSTED, the wz
BEFORE the rank-stability blend, exactly matching getMyPosition's order of operations) and reused;
only the rank-stability signal + final blend is recomputed per candidate RS_SHORT_W.
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
RS_SHORT_W0, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT


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

print("=== precompute (shared, independent of RS_SHORT_W): ridge WZ w/ beta-adjusted target + BLEND "
      "reversion + pairwise boost + ALGO leg, verbatim from v10 -- this is wz BEFORE the "
      "rank-stability blend ===", flush=True)
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

WZ_BOOSTED = np.full((nIdio, nt), np.nan)  # ridge ensemble + BLEND + boost, BEFORE rank-stability
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
    WZ_BOOSTED[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(short_w):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_BOOSTED[:, t].copy()
        if t >= max(short_w, RS_LONG_W) + 5:
            short_ret = logp[1:, t] - logp[1:, t - short_w]
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
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print(f"\n=== sanity check: shipped RS_SHORT_W={RS_SHORT_W0} must reproduce v10 exactly ===")
POS_base = build_pos(RS_SHORT_W0)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(short_w):
    Pz = build_pos(short_w); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    tag = "  <== SHIPPED" if short_w == RS_SHORT_W0 else ("  <== PASS" if passed else "")
    print(f"  RS_SHORT_W={short_w:<4}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")
    return dict(short_w=short_w, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== A24: RS_SHORT_W sweep (shipped={RS_SHORT_W0}), RS_LONG_W held at {RS_LONG_W}, "
      f"RS_WEIGHT held at {RS_WEIGHT} ===")
vals = [5, 6, 7, 8, 10, 12]
r24 = [evaluate(v) for v in vals]

passing = [c for c in r24 if c["passed"]]
best = max(r24, key=lambda c: c["rm"])
print(f"\n{len(passing)}/{len(r24)} values pass jointly. Best by rmean: RS_SHORT_W={best['short_w']} "
      f"(rmean={best['rm']:.1f} vs shipped {base_scs.mean():.1f})")
