"""
test_v19cand_boost_ncandidates.py

QUESTION: does BOOST_N_CANDIDATES=39 (the pairwise boost's leader-pool size) still sit at its
optimum against the CURRENT best (SAFE_llboost_v10), or would a different N do better?

N=39 was chosen in SAFE_llboost_v3/v5, swept only against the original SAFE_llboost baseline --
before the beta-adjusted ridge target (v9) and the rank-stability blend (v10) existed. _pairwise_boost
itself is unchanged code since v7 (it operates only on raw idio returns, independent of wz), so there
is little mechanistic reason to expect the optimum moved -- but that is an assumption, not a result.
Verifying directly against v10, not re-citing the old v3-era sweep.

Expensive precompute (ridge ensemble WZ w/ beta-adjusted target, BLEND reversion, ALGO leg,
rank-stability signal) does not depend on BOOST_N_CANDIDATES and is cached once; only the boost
itself is recomputed per candidate N.
"""
import numpy as np, pandas as pd, time
from scipy import stats
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
BOOST_ALPHA, BOOST_P, BOOST_SCALE_W, BOOST_IC_L = V10.BOOST_ALPHA, V10.BOOST_P, V10.BOOST_SCALE_W, V10.BOOST_IC_L
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT
SHIPPED_N = V10.BOOST_N_CANDIDATES


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

print("=== precompute: ridge WZ (beta-adjusted target) + BLEND reversion + ALGO leg + rank-stability "
      "signal -- unchanged, reused verbatim from v10; all independent of BOOST_N_CANDIDATES ===",
      flush=True)
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


def sig_threshold(n_samples, n_candidates):
    if n_samples < 10:
        return 1.0
    alpha_adj = BOOST_ALPHA / n_candidates
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


def boost_at_day(k, n_candidates):
    """Exact copy of SAFE_llboost_v10._pairwise_boost's body, with n_candidates parameterized and
    rs pre-truncated to day k (matching how getMyPosition truncates prcSoFar before calling it)."""
    rs_k = rs[:, :k]
    T = k
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples, n_candidates)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:n_candidates]
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    boost = np.zeros(nIdio)
    for j in range(nIdio):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


def build_pos(n_candidates):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * boost_at_day(t, n_candidates)
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print(f"\n=== sanity check: N={SHIPPED_N} (shipped) must reproduce SAFE_llboost_v10 exactly ===")
t0 = time.time()
POS_base = build_pos(SHIPPED_N)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  N={SHIPPED_N}: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(n_candidates, verbose=True):
    Pz = build_pos(n_candidates); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  N={n_candidates:<4}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
              f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")
    return dict(n=n_candidates, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== SWEEP: BOOST_N_CANDIDATES (dense around the shipped {SHIPPED_N}, sparser at the edges) ===")
SWEEP = sorted(set([15, 20, 25, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 50]))
SWEEP = [n for n in SWEEP if n <= nIdio]
t0 = time.time()
results = [evaluate(n) for n in SWEEP]
print(f"  sweep done ({time.time()-t0:.0f}s)")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} candidate-pool sizes beat v10 on OLD+NEW+rmean jointly.")
if passing:
    print("Passing values:")
    for c in passing:
        print(f"  N={c['n']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:8]:
        print(f"  N={c['n']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

best = max(results, key=lambda c: c["rm"])
print(f"\nBest by rolling mean: N={best['n']} (rmean={best['rm']:.1f} vs shipped N={SHIPPED_N} "
      f"rmean={base_scs.mean():.1f})")
