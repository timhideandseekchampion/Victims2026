"""
test_batch100_A5_A6_A7_A8.py

IDEAS A5-A8: Resweep four ALGO-leg-only parameters against v10:
  A5: VOL_WIN (ALGO vol window),        e.g. 10, 15, 20, 25, 30      (shipped 20)
  A6: VOL_Z   (ALGO vol-of-vol window), e.g. 40, 50, 60, 75, 90      (shipped 60)
  A7: IC_FAST (fast IC lookback),       e.g. 60, 75, 90, 105, 120    (shipped 90)
  A8: SWITCH_GAIN (switch-mode gain),   e.g. 1.5, 2.0, 2.5, 3.0, 3.5 (shipped 2.5)

All four parameters live ONLY inside V10._algo_vol_shares (instrument 0 / ALGO). None of them touch
the idio ridge ensemble, BLEND reversal leg, pairwise boost, or rank-stability blend. So the idio side
of the book (positions 1:) is computed ONCE, exactly as in SAFE_llboost_v10, and reused verbatim for
every candidate value in all four sweeps; only the ALGO leg (position 0) is recomputed per candidate,
by monkey-patching the relevant V10 module global and resetting V10's cross-call state
(_PREV_ALGO_SHARES, _PREV_T) before each fresh full-history recompute, then calling
V10._algo_vol_shares exactly as getMyPosition does.
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
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W

SHIPPED = dict(VOL_WIN=V10.VOL_WIN, VOL_Z=V10.VOL_Z, IC_FAST=V10.IC_FAST, SWITCH_GAIN=V10.SWITCH_GAIN)


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

print("=== precompute: full idio side (ridge ensemble + BLEND + boost + rank-stability), IDENTICAL "
      "to v10 -- independent of all four ALGO-leg params tested here, cached once ===", flush=True)
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

IDIO_POS = np.zeros((nInst, nt))
for t in days:
    wz = np.zeros(nIdio)
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
    wz = (1 - BLEND) * wz + BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
    IDIO_POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
print(f"  done ({time.time()-t0:.1f}s)", flush=True)


def algo_leg(overrides):
    """Recompute the full-history ALGO share series with the given V10 globals overridden.
    Monkey-patches V10's module namespace (which _algo_vol_shares reads as bare globals), resets the
    cross-call state, then replays exactly as getMyPosition does."""
    saved = {k: getattr(V10, k) for k in overrides}
    for k, v in overrides.items():
        setattr(V10, k, v)
    V10._PREV_ALGO_SHARES = 0
    V10._PREV_T = -1
    algo_pos = np.zeros(nt)
    try:
        for k in range(130, nt):
            cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
            algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
    finally:
        for k, v in saved.items():
            setattr(V10, k, v)
        V10._PREV_ALGO_SHARES = 0
        V10._PREV_T = -1
    return algo_pos


def build_pos(overrides):
    POS = IDIO_POS.copy()
    POS[0, :] = algo_leg(overrides)
    return POS


print(f"\n=== sanity check: shipped ALGO params {SHIPPED} must reproduce SAFE_llboost_v10 exactly ===")
t0 = time.time()
POS_base = build_pos(SHIPPED)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  shipped: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.1f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, overrides, verbose=True):
    Pz = build_pos(overrides); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<20}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
              f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


def run_sweep(param, values):
    print(f"\n=== SWEEP: {param} (shipped={SHIPPED[param]}) ===")
    t0 = time.time()
    results = [evaluate(f"{param}={v}", {param: v}) for v in values]
    print(f"  sweep done ({time.time()-t0:.1f}s)")
    passing = [c for c in results if c["passed"]]
    print(f"{len(passing)}/{len(results)} {param} values beat v10 on OLD+NEW+rmean jointly.")
    best = max(results, key=lambda c: c["rm"])
    print(f"Best by rolling mean: {best['name']} (rmean={best['rm']:.1f} vs shipped rmean={base_scs.mean():.1f})")
    return results


res_A5 = run_sweep("VOL_WIN", [10, 15, 20, 25, 30])
res_A6 = run_sweep("VOL_Z", [40, 50, 60, 75, 90])
res_A7 = run_sweep("IC_FAST", [60, 75, 90, 105, 120])
res_A8 = run_sweep("SWITCH_GAIN", [1.5, 2.0, 2.5, 3.0, 3.5])

print(f"\nFINAL: sanity_check_passed={SANITY_OK}  shipped(OLD={base_wo:.1f},NEW={base_wn:.1f},"
      f"rmean={base_scs.mean():.1f},rfloor={base_scs.min():.1f})")
