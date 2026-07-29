"""
test_batch100_boostsweeps.py

Batch of resweeps A17-A21: does each pairwise-boost sub-parameter still sit at its shipped
SAFE_llboost_v10 optimum, or would a different value do better, given everything else (idio ridge
w/ beta-adjusted target, BLEND reversion, ALGO leg, rank-stability blend) held fixed at v10?

  A17: BOOST_IC_L        in {190, 220, 250(shipped), 280, 310}
  A18: BOOST_MIN_DAY     in {420, 450, 480(shipped), 510, 540}
  A19: BOOST_P           in {1.5, 1.75, 2.0(shipped), 2.25, 2.5}
  A20: BOOST_SCALE_W     in {700, 850, 1000(shipped), 1150, 1300}
  A21: BOOST_ALPHA       in {0.01, 0.03, 0.05(shipped), 0.08, 0.10}

All five parameters live ONLY inside V10._pairwise_boost / V10._sig_threshold (module-level
globals read at call time), so each candidate value is applied by monkeypatching the corresponding
V10 module attribute immediately before calling V10._pairwise_boost verbatim, then restoring it --
no reimplementation of the boost mechanism. The expensive part (ridge ensemble w/ beta-adjusted
target, BLEND reversion, ALGO leg, rank-stability signal) is independent of all five parameters and
is computed ONCE and reused for every config, exactly like test_v19cand_boost_ncandidates.py's
caching pattern.
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

print("=== precompute (shared, independent of all 5 boost sub-params): ridge WZ w/ beta-adjusted "
      "target + BLEND reversion + ALGO leg + rank-stability signal, verbatim from v10 ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_PRE = np.full((nIdio, nt), np.nan)  # ridge ensemble + BLEND reversion, BEFORE boost / rank-stability
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
    WZ_PRE[:, t] = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]

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


def compute_boost(**overrides):
    """Compute the pairwise boost array over all days, with the given V10 module attrs
    monkeypatched for the duration of the call (restored after) -- reuses V10._pairwise_boost
    verbatim, only the swept constant(s) change."""
    saved = {k: getattr(V10, k) for k in overrides}
    for k, v in overrides.items():
        setattr(V10, k, v)
    try:
        B = np.zeros((nIdio, nt))
        for t in days:
            B[:, t] = V10._pairwise_boost(rs[:, :t])
    finally:
        for k, v in saved.items():
            setattr(V10, k, v)
    return B


def build_pos(boost_arr):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        wz = wz + BOOST_K * boost_arr[:, t]
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: default boost params must reproduce SAFE_llboost_v10 exactly ===")
t0 = time.time()
BOOST_BASE = compute_boost()
POS_base = build_pos(BOOST_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, **overrides):
    B = compute_boost(**overrides)
    Pz = build_pos(B); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    tag = "  <== SHIPPED" if not overrides else ("  <== PASS" if passed else "")
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


results = {}

print(f"\n=== A17: BOOST_IC_L sweep (shipped={V10.BOOST_IC_L}) ===")
vals = [190, 220, 250, 280, 310]
r17 = [evaluate(f"BOOST_IC_L={v}", BOOST_IC_L=v) if v != V10.BOOST_IC_L else evaluate(f"BOOST_IC_L={v}")
       for v in vals]
results["A17"] = r17

print(f"\n=== A18: BOOST_MIN_DAY sweep (shipped={V10.BOOST_MIN_DAY}) ===")
vals = [420, 450, 480, 510, 540]
r18 = [evaluate(f"BOOST_MIN_DAY={v}", BOOST_MIN_DAY=v) if v != V10.BOOST_MIN_DAY else evaluate(f"BOOST_MIN_DAY={v}")
       for v in vals]
results["A18"] = r18

print(f"\n=== A19: BOOST_P sweep (shipped={V10.BOOST_P}) ===")
vals = [1.5, 1.75, 2.0, 2.25, 2.5]
r19 = [evaluate(f"BOOST_P={v}", BOOST_P=v) if v != V10.BOOST_P else evaluate(f"BOOST_P={v}")
       for v in vals]
results["A19"] = r19

print(f"\n=== A20: BOOST_SCALE_W sweep (shipped={V10.BOOST_SCALE_W}) ===")
vals = [700, 850, 1000, 1150, 1300]
r20 = [evaluate(f"BOOST_SCALE_W={v}", BOOST_SCALE_W=v) if v != V10.BOOST_SCALE_W else evaluate(f"BOOST_SCALE_W={v}")
       for v in vals]
results["A20"] = r20

print(f"\n=== A21: BOOST_ALPHA sweep (shipped={V10.BOOST_ALPHA}) ===")
vals = [0.01, 0.03, 0.05, 0.08, 0.10]
r21 = [evaluate(f"BOOST_ALPHA={v}", BOOST_ALPHA=v) if v != V10.BOOST_ALPHA else evaluate(f"BOOST_ALPHA={v}")
       for v in vals]
results["A21"] = r21

print("\n=== SUMMARY ===")
for aid, res in results.items():
    passing = [c for c in res if c["passed"]]
    best = max(res, key=lambda c: c["rm"])
    print(f"  {aid}: {len(passing)}/{len(res)} pass jointly. Best by rmean: {best['name']} "
          f"(rmean={best['rm']:.1f} vs shipped {base_scs.mean():.1f})")
