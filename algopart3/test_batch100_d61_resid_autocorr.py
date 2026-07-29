"""
test_batch100_d61_resid_autocorr.py

D61: two-stage ridge. Stage 1 is the shipped ridge, unchanged. Stage 2 models the AUTOCORRELATION of
stage-1's own in-sample fit residuals (a Cochrane-Orcutt-style AR(1) residual correction) and adds a
correction term to the out-of-sample forecast: pred_corrected = pred + PHI*E_last, where E_last is the
most recent in-sample residual (per name) and PHI is a pooled (across names and time, for tractability
-- same "aggregate, stated honestly" simplification test_v12cand_huber.py used for its per-day
robustness weight) lag-1 autocorrelation of the stage-1 residuals.

MECHANISM (per half-life, each call):
  1. Fit B, mx, my via V10._ewls_ridge (unchanged, reused).
  2. Compute in-sample residuals E = (Y-my) - (X-mx)@B  (raw, unweighted -- just for estimating the
     autocorrelation structure, not refitting).
  3. PHI = pooled lag-1 corr of E[:-1,:].ravel() vs E[1:,:].ravel() (one scalar per half-life/day).
  4. pred_corrected = pred + PHI_SCALE * PHI * E[-1, :]  (E[-1,:] = each name's most recent residual).

PHI_SCALE=0 reproduces v10 exactly (pure stage-1). PHI_SCALE=1 applies the full AR(1) correction;
PHI_SCALE=0.5 a half-strength (shrunk) version, in case the pooled/aggregate PHI is too noisy applied
at full strength.

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
      "UNCHANGED / reused verbatim from V10 (independent of the 2nd-stage correction under test) ===",
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


def build_wz_ridge(phi_scale):
    WZ = np.full((nIdio, nt), np.nan)
    phi_track = []
    for t in days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            if phi_scale != 0.0:
                Xc = X - mx; Yc = Y - my
                E = Yc - Xc @ B
                if E.shape[0] >= 30:
                    e0 = E[:-1, :].ravel(); e1 = E[1:, :].ravel()
                    s0, s1 = e0.std(), e1.std()
                    if s0 > 1e-12 and s1 > 1e-12:
                        phi = float(np.corrcoef(e0, e1)[0, 1])
                        pred = pred + phi_scale * phi * E[-1, :]
                        phi_track.append(phi)
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ, phi_track


print("\n=== sanity check: PHI_SCALE=0 (mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE, _ = build_wz_ridge(0.0)
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


print("\n=== DIAGNOSTIC: pooled lag-1 autocorrelation of stage-1 ridge residuals (hl=500, sampled) ===")
sample_phis = []
for t in days[::60]:
    rr_ = r[:, :t]
    Y = V10._beta_adjusted_target(rr_)
    X = rr_[:, :-1].T
    B, mx, my = V10._ewls_ridge(X, Y, 500, RIDGE_A)
    Xc = X - mx; Yc = Y - my
    E = Yc - Xc @ B
    if E.shape[0] >= 30:
        e0 = E[:-1, :].ravel(); e1 = E[1:, :].ravel()
        if e0.std() > 1e-12 and e1.std() > 1e-12:
            sample_phis.append(float(np.corrcoef(e0, e1)[0, 1]))
sample_phis = np.array(sample_phis)
print(f"  pooled lag-1 autocorr of hl=500 in-sample residuals, sampled every 60 days: "
      f"mean={sample_phis.mean():.4f}  std={sample_phis.std():.4f}  "
      f"min={sample_phis.min():.4f}  max={sample_phis.max():.4f}  n={len(sample_phis)}")
print("  interpretation: |phi| this small/noisy => little exploitable AR(1) structure expected in the "
      "correction term regardless of PHI_SCALE." if abs(sample_phis.mean()) < 0.05 else
      "  interpretation: a non-trivial pooled residual autocorrelation exists -- worth the stage-2 test.")


print("\n=== CANDIDATE: PHI_SCALE in {0.5, 1.0} (pooled AR(1) residual correction) ===")
results = []
for phi_scale in (0.5, 1.0):
    t0 = time.time()
    WZ_C, _ = build_wz_ridge(phi_scale)
    c = evaluate(f"PHI_SCALE={phi_scale}", WZ_C, base_wo, base_wn, base_scs)
    results.append(c)
    print(f"  [{time.time()-t0:.0f}s]")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} PHI_SCALE configs beat v10 on OLD+NEW+rmean jointly.")
if not passing:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
