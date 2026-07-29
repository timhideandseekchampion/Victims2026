"""
test_batch100_B31_kalman.py

B31: Re-test Kalman/RLS continuously-adapting ridge coefficients against v10. Previously
(test_q20_items01_04_ridge_variants.py) tested against the ORIGINAL SAFE_llboost baseline (rmean
811.4) -- pre-beta-demean, pre-boost-tuning, pre-rank-stability -- and collapsed decisively (rmean
614-634 at every process-noise setting, floor 353-364). Re-testing against the CURRENT best (v10),
with the CURRENT beta-adjusted target, to check the mechanism isn't secretly fine now that so much
else has changed underneath it.

MECHANISM: identical diagonal-covariance Kalman filter from the original test -- state B (nIdio x
nInst) evolves as a random walk (process noise Q = R_OBS * q_frac), updated every day via the standard
Kalman gain/update using TODAY's return vector as the observation regressor. This is a genuinely
different estimator from the fixed 4-half-life EW-ridge ensemble: coefficients drift continuously
rather than being re-fit fresh each call with a fixed forgetting profile.

TARGET: the observation `y` fed into the Kalman update at each step is the CURRENT beta-adjusted
target (V10._beta_adjusted_target's last row, evaluated causally at that step) instead of the original
test's raw next-day idio return -- this is the one piece that must change to make this a fair test of
"Kalman vs v10", since v10's ridge ensemble is fit on the beta-adjusted target, not raw returns.

"Mechanism OFF" doesn't apply literally to a Kalman filter (there's no q_frac that reduces it to the
EW-ridge ensemble) -- so per this repo's convention for such cases (test_v9cand_rrr.py's rank=50,
test_v12cand_huber.py's huber_k=None), the sanity check instead verifies that the REST of the pipeline
(BLEND reversion, pairwise boost, rank-stability blend, ALGO leg -- all cached once, independent of
which ridge estimator is used) reproduces v10 exactly when combined with the ORDINARY ridge ensemble,
before layering in the Kalman-based estimator.
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

print("=== precompute: BLEND reversion, pairwise boost, rank-stability signal, ALGO leg -- IDENTICAL "
      "regardless of the ridge estimator, cached once ===", flush=True)
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
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        RS_SIG[:, t] = rs_sig
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def blend_final(wz, t):
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_pos_from_wz(wz_fn):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = blend_final(wz_fn(t), t)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def wz_ridge_ensemble(t):
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
    return np.mean(fs, 0)


print("\n=== sanity check: plain ridge ensemble (mechanism 'off') must reproduce SAFE_llboost_v10 "
      "exactly -- validates the shared REV/boost/rank-stability/ALGO cache before testing Kalman ===")
POS_base = build_pos_from_wz(wz_ridge_ensemble)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")

print("\n=== building beta-adjusted target series (causal, one call per day) for the Kalman "
      "observation ===", flush=True)
t0 = time.time()
Y_BA = np.full((nt, nIdio), np.nan)   # Y_BA[k+1] = beta-adjusted target for day-k+1 observation
for k in range(1, nt - 1):
    Yfull = V10._beta_adjusted_target(r[:, :k + 1])
    Y_BA[k + 1] = Yfull[-1]
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

R_OBS = float(np.nanvar(Y_BA))
print(f"  R_obs (empirical beta-adjusted-target variance) = {R_OBS:.6f}")


def kalman_wz_series(q_frac, pdiag0=1.0):
    Q = R_OBS * q_frac
    B = np.zeros((nIdio, nInst))
    Pdiag = np.full((nIdio, nInst), pdiag0)
    WZ = {}
    for k in range(0, nt - 1):
        x = r[:, k]
        pred = B @ x
        t_out = k + 1
        if t_out >= WARMUP:
            fi = pred - pred.mean()
            WZ[t_out] = fi / (fi.std() + 1e-12)
        y = Y_BA[k + 1]
        if np.isnan(y).any():
            continue
        Pdiag = Pdiag + Q
        Sigma = (Pdiag * (x ** 2)[None, :]).sum(1) + R_OBS
        Kg = Pdiag * x[None, :] / Sigma[:, None]
        resid = y - pred
        B = B + Kg * resid[:, None]
        Pdiag = (1 - Kg * x[None, :]) * Pdiag
    return WZ


def evaluate(nm, q_frac, verbose=True):
    t0 = time.time()
    WZK = kalman_wz_series(q_frac)
    Pz = build_pos_from_wz(lambda t: WZK[t])
    scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<20}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}  [{time.time()-t0:.0f}s]")
    return dict(name=nm, q_frac=q_frac, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse,
                passed=passed)


print("\n=== SWEEP: Kalman process-noise fraction q_frac ===")
QFRACS = [1e-5, 1e-4, 1e-3, 1e-2]
results = [evaluate(f"q_frac={qf:g}", qf) for qf in QFRACS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} q_frac values beat v10 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: q_frac={best['q_frac']:g}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  q_frac={c['q_frac']:<10g} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
