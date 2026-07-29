"""
test_batch100_betademean.py

Batch resweeps A22-A23: does each beta-adjusted-target sub-parameter still sit at its shipped
SAFE_llboost_v10 optimum, or would a different value do better, given everything else (pairwise
boost, BLEND reversion, ALGO leg, rank-stability blend) held fixed at v10?

  A22: BETA_DEMEAN_LAM  in {0.3, 0.4, 0.5, 0.6(shipped), 0.7}   (BETA_DEMEAN_W held at shipped 500)
  A23: BETA_DEMEAN_W    in {350, 450, 500(shipped), 550, 650}   (BETA_DEMEAN_LAM held at shipped 0.6)

Both parameters live ONLY inside V10._beta_adjusted_target (module-level globals read at call
time), which feeds the ridge ensemble's Y target -- so each candidate value requires monkeypatching
the corresponding V10 module attribute and rerunning the ridge ensemble (the expensive part) for
that config. What does NOT depend on these two parameters -- and is cached once and reused for
every config -- is: the pairwise boost (operates on raw idio returns rs, not the beta-adjusted
target), the BLEND reversion leg, the ALGO leg, and the rank-stability signal.
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
BETA_LAM0, BETA_W0 = V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W


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

print("=== precompute (shared, independent of BETA_DEMEAN_LAM/W): pairwise boost (raw idio returns) "
      "+ BLEND reversion + ALGO leg + rank-stability signal, verbatim from v10 ===", flush=True)
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


def compute_wz_pre(lam, w):
    """Ridge ensemble + BLEND reversion, with BETA_DEMEAN_LAM/W monkeypatched for the duration --
    reuses V10._beta_adjusted_target and V10._ewls_ridge verbatim."""
    saved_lam, saved_w = V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W
    V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W = lam, w
    try:
        WZ = np.full((nIdio, nt), np.nan)
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
            WZ[:, t] = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    finally:
        V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W = saved_lam, saved_w
    return WZ


def build_pos(WZ_PRE):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
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


print(f"\n=== sanity check: default (LAM={BETA_LAM0}, W={BETA_W0}) must reproduce v10 exactly ===")
t0 = time.time()
WZ_base = compute_wz_pre(BETA_LAM0, BETA_W0)
POS_base = build_pos(WZ_base)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, lam, w):
    t0 = time.time()
    WZ = compute_wz_pre(lam, w)
    Pz = build_pos(WZ); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    tag = "  <== SHIPPED" if (lam == BETA_LAM0 and w == BETA_W0) else ("  <== PASS" if passed else "")
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}  [{time.time()-t0:.0f}s]")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== A22: BETA_DEMEAN_LAM sweep (shipped={BETA_LAM0}), BETA_DEMEAN_W held at {BETA_W0} ===")
vals = [0.3, 0.4, 0.5, 0.6, 0.7]
r22 = [evaluate(f"LAM={v}", v, BETA_W0) for v in vals]

print(f"\n=== A23: BETA_DEMEAN_W sweep (shipped={BETA_W0}), BETA_DEMEAN_LAM held at {BETA_LAM0} ===")
vals = [350, 450, 500, 550, 650]
r23 = [evaluate(f"W={v}", BETA_LAM0, v) for v in vals]

print("\n=== SUMMARY ===")
for aid, res in [("A22", r22), ("A23", r23)]:
    passing = [c for c in res if c["passed"]]
    best = max(res, key=lambda c: c["rm"])
    print(f"  {aid}: {len(passing)}/{len(res)} pass jointly. Best by rmean: {best['name']} "
          f"(rmean={best['rm']:.1f} vs shipped {base_scs.mean():.1f})")
