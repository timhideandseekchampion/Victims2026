"""
test_batch100_A23_full.py

FULL production-rigor resweep of A23 (BETA_DEMEAN_W) against SAFE_llboost_v10, following up on the
Stage-1 quick screen in test_batch100_betademean.py, which used a sparse 5-point grid
{350, 450, 500(shipped), 550, 650} and found only W=550 passing jointly (OLD/NEW/rmean all beat
v10), with its immediate neighbours (450, 650) both WORSE than shipped -- i.e. Stage-1 could not
distinguish a genuine local plateau around 550 from an isolated one-point spike.

This script re-runs the same mechanism (BETA_DEMEAN_W lives only inside V10._beta_adjusted_target,
feeding the ridge ensemble's Y target) with a MUCH denser grid:
  - a broad scan, step 25, from 350 to 700
  - a fine scan, step 10, from 500 to 600 (bracketing the Stage-1 winner)
so we can see whether there is a genuine multi-point plateau around any W, or whether 550 was a
lucky isolated point.

Everything else (pairwise boost on raw idio returns, BLEND reversion, ALGO leg, rank-stability
blend) is cached ONCE, verbatim from V10, exactly as in test_batch100_betademean.py -- only the
ridge ensemble (which depends on BETA_DEMEAN_W through the target Y) is recomputed per candidate.
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
BETA_LAM0, BETA_W0 = V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W  # 0.6, 500


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

print("=== precompute (shared, independent of BETA_DEMEAN_W): pairwise boost (raw idio returns) "
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


def compute_wz_pre(w):
    """Ridge ensemble + BLEND reversion, with BETA_DEMEAN_W monkeypatched for the duration --
    reuses V10._beta_adjusted_target and V10._ewls_ridge verbatim (BETA_DEMEAN_LAM held at shipped)."""
    saved_lam, saved_w = V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W
    V10.BETA_DEMEAN_LAM, V10.BETA_DEMEAN_W = BETA_LAM0, w
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
WZ_base = compute_wz_pre(BETA_W0)
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


def evaluate(nm, w):
    t0 = time.time()
    WZ = compute_wz_pre(w)
    Pz = build_pos(WZ); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    tag = "  <== SHIPPED" if w == BETA_W0 else ("  <== PASS" if passed else "")
    print(f"  {nm:<14}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse:3d}/{len(scs)}{tag}  [{time.time()-t0:.0f}s]")
    return dict(name=nm, w=w, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== BROAD scan: BETA_DEMEAN_W in range(350, 701, 25) (shipped={BETA_W0}) ===")
broad_vals = list(range(350, 701, 25))
broad_res = [evaluate(f"W={v}", v) for v in broad_vals]

print(f"\n=== FINE scan: BETA_DEMEAN_W in range(500, 601, 10), bracketing Stage-1 winner (550) ===")
fine_vals = [v for v in range(500, 601, 10) if v not in broad_vals]
fine_res = [evaluate(f"W={v}", v) for v in fine_vals]

all_res = sorted(broad_res + fine_res, key=lambda c: c["w"])

print("\n=== FULL TABLE (sorted by W) ===")
for c in all_res:
    tag = "  <== SHIPPED" if c["w"] == BETA_W0 else ("  <== PASS" if c["passed"] else "")
    print(f"  W={c['w']:<5} OLD={c['wo']:7.1f}  NEW={c['wn']:7.1f}  rmean={c['rm']:7.1f}  "
          f"rfloor={c['rf']:7.1f}  n_worse={c['nworse']:3d}/61{tag}")

passing = [c for c in all_res if c["passed"]]
print(f"\n{len(passing)}/{len(all_res)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for c in sorted(passing, key=lambda c: -c["rm"]):
        print(f"  W={c['w']:<5} rmean={c['rm']:.1f} n_worse={c['nworse']}/61")

best = max(all_res, key=lambda c: c["rm"])
print(f"\nBest by rolling mean overall: W={best['w']} (rmean={best['rm']:.1f} vs shipped {base_scs.mean():.1f}, "
      f"n_worse={best['nworse']}/61, passed={best['passed']})")

# plateau-vs-spike check: are W's immediately adjacent to the best passing candidates also passing
# (or at least beating shipped rmean), or does performance collapse right next to them?
print("\n=== plateau-vs-spike check (is each passing W flanked by neighbors that also beat shipped rmean?) ===")
w_to_res = {c["w"]: c for c in all_res}
sorted_ws = sorted(w_to_res.keys())
for c in passing:
    w = c["w"]
    idx = sorted_ws.index(w)
    left = w_to_res[sorted_ws[idx - 1]] if idx > 0 else None
    right = w_to_res[sorted_ws[idx + 1]] if idx < len(sorted_ws) - 1 else None
    lstr = f"W={left['w']} rmean={left['rm']:.1f}{' beats shipped' if left['rm']>base_scs.mean() else ' BELOW shipped'}" if left else "n/a"
    rstr = f"W={right['w']} rmean={right['rm']:.1f}{' beats shipped' if right['rm']>base_scs.mean() else ' BELOW shipped'}" if right else "n/a"
    print(f"  W={w}: left neighbor [{lstr}], right neighbor [{rstr}]")
