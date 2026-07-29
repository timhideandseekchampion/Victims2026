"""
test_batch100_A25_A26.py

A25: Resweep RS_LONG_W against v10 (15, 18, 20, 22, 25, 28), RS_SHORT_W fixed at shipped 8.
A26: Finer resweep of RS_WEIGHT against v10 around 0.015 (0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.030),
     RS_SHORT_W/RS_LONG_W fixed at shipped 8/22.

Both diagnostics operate on the rank-stability blend that is the ONLY new mechanism in v10 relative to
v9. Everything upstream of that blend (idio ridge ensemble w/ beta-adjusted target, BLEND reversion,
pairwise boost, ALGO leg) is IDENTICAL for every candidate in this sweep, so it is cached ONCE
("WZ_PRE" = wz right before the rank-stability blend is applied) and reused -- only the cheap
rank-stability signal + final blend is recomputed per candidate, exactly like
test_v19cand_boost_ncandidates.py's caching pattern.
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
SHIPPED_SHORT, SHIPPED_LONG, SHIPPED_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT


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

print("=== precompute: idio ridge ensemble (beta-adjusted target) + BLEND reversion + pairwise boost + "
      "ALGO leg -- IDENTICAL for every rank-stability candidate, cached once (WZ_PRE) ===", flush=True)
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

WZ_PRE = np.full((nIdio, nt), np.nan)
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
    WZ_PRE[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def rs_signal(t, short_w, long_w):
    if t < max(short_w, long_w) + 5:
        return None
    short_ret = logp[1:, t] - logp[1:, t - short_w]
    long_ret = logp[1:, t] - logp[1:, t - long_w]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        return None
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    return np.where(disagree, -sz, 0.0)


def build_pos(short_w, long_w, weight):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        rs_sig = rs_signal(t, short_w, long_w)
        if rs_sig is not None:
            s_std = rs_sig.std()
            s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros(nIdio)
            wz = (1 - weight) * wz + weight * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: shipped (short=8, long=22, weight=0.015) must reproduce SAFE_llboost_v10 "
      "exactly ===")
POS_base = build_pos(SHIPPED_SHORT, SHIPPED_LONG, SHIPPED_WEIGHT)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, short_w, long_w, weight, verbose=True):
    Pz = build_pos(short_w, long_w, weight); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, short_w=short_w, long_w=long_w, weight=weight, wo=wo, wn=wn, rm=scs.mean(),
                rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== A25: SWEEP RS_LONG_W (short=8 fixed, weight=0.015 fixed) ===")
LONG_W_SWEEP = [15, 18, 20, 22, 25, 28]
a25_results = [evaluate(f"long_w={lw}", SHIPPED_SHORT, lw, SHIPPED_WEIGHT) for lw in LONG_W_SWEEP]
a25_passing = [c for c in a25_results if c["passed"]]
print(f"\n{len(a25_passing)}/{len(a25_results)} long_w values beat v10 on OLD+NEW+rmean jointly.")

print("\n=== A26: SWEEP RS_WEIGHT (short=8, long=22 fixed) ===")
WEIGHT_SWEEP = [0.008, 0.010, 0.012, 0.015, 0.018, 0.022, 0.030]
a26_results = [evaluate(f"weight={w}", SHIPPED_SHORT, SHIPPED_LONG, w) for w in WEIGHT_SWEEP]
a26_passing = [c for c in a26_results if c["passed"]]
print(f"\n{len(a26_passing)}/{len(a26_results)} weight values beat v10 on OLD+NEW+rmean jointly.")

print("\n=== SUMMARY ===")
print("A25 (RS_LONG_W):")
for c in a25_results:
    tag = "  <== PASS" if c["passed"] else ""
    print(f"  long_w={c['long_w']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61{tag}")
print("A26 (RS_WEIGHT):")
for c in a26_results:
    tag = "  <== PASS" if c["passed"] else ""
    print(f"  weight={c['weight']:<6} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61{tag}")
