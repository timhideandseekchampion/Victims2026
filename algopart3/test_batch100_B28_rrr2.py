"""
test_batch100_B28_rrr.py

B28: Re-test reduced-rank regression (RRR) on the idio ridge ensemble against v10, with the CURRENT
beta-adjusted target as Y. The original RRR test (test_v9cand_rrr.py) was run against SAFE_llboost_v8
-- BEFORE the beta-adjusted target (v9) and rank-stability blend (v10) existed -- using the plain
unadjusted next-day-return target (Y = rs[:,1:].T). Re-verifying the same estimator with Y =
V10._beta_adjusted_target(...) instead, against the current best (v10).

ESTIMATOR: identical construction to test_v9cand_rrr.py (verified there by hand: given the per-half-life
fit's S = XtWX + (eps+a)*I and B = S^-1 XtWY, the ridge loss reduces to L(C) = const +
tr((C-B)'S(C-B)); minimizing subject to rank(C)<=r gives the S-weighted low-rank approximation
C_r = B @ V_r @ V_r', V_r = top-r right singular vectors of L'B where S = LL' via Cholesky). At
rank=nIdio=50 (=min(p,q)) this reproduces full-rank B exactly, hence SAFE_llboost_v10 exactly (with
boost + rank-stability layered on top, both independent of the ridge estimator, cached once).

EFFICIENCY: fit + Cholesky + SVD depend only on (day, half-life), not on rank r -- precomputed once;
every candidate rank is then a cheap O(q*r) projection.

RANKS swept: {1, 5, 15, 25, 35, 50} per the task.
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
n_hl = len(HALF_LIVES)
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


def _ewls_rrr_fit(X, Y, hl, a, xq):
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    S = XtWX + (eps + a) * np.eye(p)
    B = np.linalg.solve(S, XtWY)
    L = np.linalg.cholesky(S)
    D0 = L.T @ B
    _, _, Vt = np.linalg.svd(D0, full_matrices=False)
    full = (xq - mx) @ B
    return my, full, Vt


def _rrr_pred(my, full, Vt, rank):
    rr_ = min(rank, Vt.shape[0])
    coef = full @ Vt[:rr_].T
    proj = coef @ Vt[:rr_]
    return my + proj


print("=== precompute: RRR fit cache (B/S/Cholesky/SVD once per day/half-life, using the CURRENT "
      "beta-adjusted target) -- rank sweep will be cheap ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
n_days = len(days)
MY = np.zeros((n_hl, n_days, nIdio))
FULL = np.zeros((n_hl, n_days, nIdio))
VT = np.zeros((n_hl, n_days, nIdio, nIdio))
for di, t in enumerate(days):
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    for hi, hl in enumerate(HALF_LIVES):
        my, full, Vt = _ewls_rrr_fit(X, Y, hl, RIDGE_A, xq)
        MY[hi, di] = my; FULL[hi, di] = full; VT[hi, di] = Vt
print(f"  done ({time.time()-t0:.0f}s, {n_days} days x {n_hl} half-lives)", flush=True)

print("=== precompute: BLEND reversion, pairwise boost, rank-stability signal, ALGO leg -- IDENTICAL "
      "regardless of RRR rank (independent of the ridge estimator), cached once ===", flush=True)
t0 = time.time()
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


def build_pos(rank):
    POS = np.zeros((nInst, nt))
    for di, t in enumerate(days):
        fs = []
        for hi in range(n_hl):
            pred = _rrr_pred(MY[hi, di], FULL[hi, di], VT[hi, di], rank)
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


print("\n=== sanity check: rank=50 (full rank) must reproduce SAFE_llboost_v10 exactly ===")
POS_full = build_pos(50)
base_scs = scs_curve(POS_full)
base_wo, base_wn = wscore(POS_full, *OLD), wscore(POS_full, *NEW)
print(f"  rank=50: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: rank=50 does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, rank, verbose=True):
    Pz = build_pos(rank); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<12}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, rank=rank, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== RANK SWEEP vs SAFE_llboost_v10 ===")
RANKS = [1, 5, 15, 25, 35, 50]
results = [evaluate(f"r={rk}", rk) for rk in RANKS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} ranks beat v10 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: r={best['rank']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  r={c['rank']:<3} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
