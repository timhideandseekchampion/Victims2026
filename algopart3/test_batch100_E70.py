"""
test_batch100_E70.py

E70: Test a MULTI-HORIZON reversal blend for the idio book's own fade component -- the shipped
unconditional BLEND=0.3 / REV_W=10 leg, which fades each idio name's OWN trailing REV_W-day return
(cross-sectionally z-scored, sign-flipped) -- replacing the single REV_W=10 lookback with an
unweighted AVERAGE of z-scored fade signals across three lookbacks bracketing the shipped value,
"instead of one REV_L" (three lookback sets swept, 5/10/12/15/20, each anchored near REV_W=10).

NOTE on the idea's phrasing: v10 has no per-instrument "ALGO REV_L" parameter -- the only
reversal/fade lookback in the file is REV_W (idio leg, BLEND=0.3), which fades each name's OWN
recent return (distinct from the cross-sectional decile-momentum tilt and from the pairwise
lead-lag boost). That is the mechanism tested here, under the batch author's evident intent of
"replace one lookback with several, blended."

Everything else (ridge ensemble, BLEND=0.3 weight itself, pairwise boost, rank-stability blend,
ALGO leg) is byte-identical to v10 -- only the REV_W-lookback INPUT to the BLEND is replaced by a
multi-lookback average.
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
BLEND = V10.BLEND


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

print("=== precompute: shared pieces unaffected by the REV-blend swap (ridge ensemble target, "
      "boost, ALGO leg) ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))

# ridge ensemble base forecast fi (pre-BLEND), identical across all REV variants -- cache it
FI_BASE = np.full((nIdio, nt), np.nan)
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
    FI_BASE[:, t] = np.mean(fs, 0)
print(f"  ridge ensemble done ({time.time()-t0:.0f}s)", flush=True)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  boost + ALGO leg done ({time.time()-t0:.0f}s total)", flush=True)


def rev_signal(t, L):
    """v10's own-name fade signal at lookback L, day t: cross-sectional z-score of the trailing
    L-day idio return, sign-flipped (fade)."""
    rv_ = logp[1:, t] - logp[1:, t - L]
    rv_ = rv_ - rv_.mean()
    return -rv_ / (rv_.std() + 1e-12)


def build_wz(rev_fn):
    """rev_fn(t) -> the REV-leg array (nIdio,) blended into wz at BLEND weight, for day t."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = FI_BASE[:, t]
        wz = (1 - BLEND) * wz + BLEND * rev_fn(t)
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
        WZ[:, t] = wz
    return WZ


def build_pos(WZ):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: single REV_W=10 (v10 verbatim) must reproduce SAFE_llboost_v10 exactly ===")
WZ_BASE = build_wz(lambda t: rev_signal(t, V10.REV_W))
POS_base = build_pos(WZ_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: does NOT reproduce v10 -- do not trust results below. ***")
if not SANITY_OK:
    raise SystemExit("Sanity check failed -- aborting.")


def rev_multiblend(t, LBS):
    return np.mean([rev_signal(t, L) for L in LBS], axis=0)


print("\n=== E70: multi-horizon reversal blend (average of z-scored fade signals across several "
      "lookbacks) replacing the single REV_W=10 lookback in the BLEND leg ===")
LB_SETS = [(5, 10, 20), (5, 10, 15), (7, 10, 15)]
results = []
for LBS in LB_SETS:
    t0 = time.time()
    WZ = build_wz(lambda t, LBS=LBS: rev_multiblend(t, LBS))
    Pz = build_pos(WZ)
    scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    print(f"  LBS={LBS}  OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}  passed={passed}   "
          f"[{time.time()-t0:.0f}s]")
    results.append(dict(lbs=LBS, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse,
                         passed=passed))

npass = sum(1 for c in results if c["passed"])
print(f"\n{npass}/{len(results)} lookback-set configs beat v10 on OLD+NEW+rmean jointly.")
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}")
