"""
test_batch100_B29_ebridge.py

B29: Re-test predictor-wise empirical-Bayes ridge shrinkage against v10, GAMMA swept 0 to 3.0.
Originally tested (test_v11cand_predictor_shrink.py) against SAFE_llboost_v9, which already had the
beta-adjusted target active -- so the target itself is unchanged since then; what's NEW is the
rank-stability blend (v10) sitting downstream of the ridge. Re-verifying directly against v10.

MECHANISM (identical to test_v11cand_predictor_shrink.py): replace the single scalar RIDGE_A with a
per-predictor penalty a_i = RIDGE_A * (mean(rel) / max(rel_i, floor)) ** GAMMA, where rel_i is
predictor i's trailing pooled |corr| against all 50 idio targets (window=250, matching the original).
GAMMA=0 reproduces uniform RIDGE_A exactly -- mandatory sanity check (must reproduce v10 exactly).

REV/boost/rank-stability/ALGO leg cached once (independent of the ridge estimator) and reused across
every GAMMA, exactly like test_v19cand_boost_ncandidates.py's caching pattern.
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


def _ewls_ridge_diag(X, Y, hl, a_vec):
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + np.diag(a_vec) + eps * np.eye(p), XtWY)
    return B, mx, my


def predictor_reliability(X, Y, window):
    Xw = X[-window:]; Yw = Y[-window:]
    Xc = Xw - Xw.mean(0); Yc = Yw - Yw.mean(0)
    Xs = Xc / (Xc.std(0) + 1e-12); Ys = Yc / (Yc.std(0) + 1e-12)
    C = (Xs.T @ Ys) / Xw.shape[0]
    return np.abs(C).mean(1)


def a_vec_from_r(rel, gamma, floor=0.01):
    if gamma == 0:
        return np.full(rel.shape, RIDGE_A)
    rbar = rel.mean()
    mult = (rbar / np.maximum(rel, floor)) ** gamma
    return RIDGE_A * mult


print("=== precompute: BLEND reversion, pairwise boost, rank-stability signal, ALGO leg -- IDENTICAL "
      "for every GAMMA (independent of the ridge estimator), cached once ===", flush=True)
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


def build_pos(gamma, window):
    POS = np.zeros((nInst, nt))
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Y = V10._beta_adjusted_target(rr_)
        xq = rr_[:, -1]
        if gamma != 0:
            rel = predictor_reliability(X, Y, min(window, X.shape[0]))
            a_vec = a_vec_from_r(rel, gamma)
        else:
            a_vec = np.full(X.shape[1], RIDGE_A)
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = _ewls_ridge_diag(X, Y, hl, a_vec)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
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


print("\n=== sanity check: gamma=0 must reproduce SAFE_llboost_v10 exactly ===")
POS_base = build_pos(0.0, 250)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  gamma=0: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: gamma=0 does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, gamma, window, verbose=True):
    Pz = build_pos(gamma, window); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<20}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, gamma=gamma, window=window, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                nworse=nworse, passed=passed)


print("\n=== SWEEP: GAMMA in [0, 3.0] (window=250, matching the original test) ===")
GAMMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
results = [evaluate(f"gamma={g}", g, 250) for g in GAMMAS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: gamma={best['gamma']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  gamma={c['gamma']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
