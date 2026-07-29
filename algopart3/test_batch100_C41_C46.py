"""
test_batch100_C41_C46.py

Batch of 6 ideas (C41-C46), all modifying only the leader-detection / combination step inside
_pairwise_boost, tested against SAFE_llboost_v10 (current best). Everything upstream of the boost
(idio ridge ensemble w/ beta-adjusted target, BLEND reversion, rank-stability blend, ALGO leg) is
reused VERBATIM from V10 and is unaffected by any idea here -- computed once, shared across all 6.

  C41: average top-2 candidate leaders per follower (weighted by |corr|) instead of single best.
  C42: average top-3 candidate leaders per follower (weighted by |corr|).
  C43: two-hop transitive boost -- if A leads B and B leads C, also try A as a direct candidate
       leader for C (in addition to C's own direct leader), combined weighted by |corr|.
  C44: cluster-restricted candidate pool -- k-means on the causal all-pairs idio corr matrix (rows
       as feature vectors), sweep k in {4,5,6}; restrict each follower's leader search to its OWN
       cluster (no global top-39-by-vol filter).
  C45: distance correlation (not Pearson) for leader detection. Vectorized via double-centered
       distance matrices; APPROXIMATION for tractability: uses a rolling W=250-day window (not the
       full growing causal history the Pearson version uses) and reuses the Pearson significance
       threshold (computed at n_samples=W) as a magnitude-matched cutoff (dCor's true null
       distribution differs from a correlation t-test -- flagged as an approximation appropriate to
       a screening pass).
  C46: graphical-lasso sparse inverse covariance (GraphicalLassoCV) replacing the top-39-by-vol pool
       with a sparse partial-correlation neighbor graph; refit every 25 days (CV fit costs ~0.5-1s,
       refitting daily over ~520 days is not a "screening pass" cost) -- graph held fixed between
       refits. Given each follower's neighbor set, still uses the SAME argmax-|corr| + significance-
       style downstream logic as v10, only the *candidate universe* changes.

All 6 reuse V10._ewls_ridge, V10._beta_adjusted_target, V10._algo_vol_shares,
V10._rank_stability_signal (via the verbatim precompute below), V10._corrmat, V10._sig_threshold,
and V10's constants directly. A `combine_leaders()` helper generalizes _pairwise_boost's per-follower
scale/power-transform + IC>0 gate + apply-lead_boost[-1] body to an arbitrary list of
(candidate, weight) pairs -- this is _pairwise_boost's own inner-loop body lifted out and
parameterized, not a from-scratch reimplementation; at a single candidate with weight 1.0 it is
bit-identical to _pairwise_boost's inner loop (verified below via the K=1 internal check).

NOTE: this supersedes an earlier partial/incomplete attempt at this same batch
(test_batch100_c41_c48.py, left in place untouched, also covered C47/C48 which are NOT in this
assignment) -- written fresh under a new filename per instructions, and actually executed
synchronously in the foreground here with real printed numbers below.
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
BOOST_ALPHA, BOOST_P, BOOST_SCALE_W, BOOST_IC_L = V10.BOOST_ALPHA, V10.BOOST_P, V10.BOOST_SCALE_W, V10.BOOST_IC_L
BOOST_N_CANDIDATES = V10.BOOST_N_CANDIDATES
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
days = list(range(WARMUP, nt))

print("=== precompute (shared across all 6 ideas): ridge WZ (beta-adjusted target) + BLEND reversion "
      "+ ALGO leg + rank-stability signal -- unchanged, reused verbatim from v10 ===", flush=True)
t0 = time.time()
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_PRE = np.full((nIdio, nt), np.nan)
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


def rs_blend(wz, t):
    s = RS_SIG[:, t]
    if not np.isfinite(s).all():
        return wz
    sstd = s.std()
    s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
    return (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)


def build_pos(boost_by_day):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * boost_by_day[:, t]
        wz = rs_blend(wz, t)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, boost_arr, base_wo, base_wn, base_scs, verbose=True):
    Pz = build_pos(boost_arr); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<30}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}", flush=True)
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=bool(passed))


print("\n=== MANDATORY sanity check: mechanism OFF (V10._pairwise_boost verbatim) must reproduce "
      "SAFE_llboost_v10 exactly ===", flush=True)
t0 = time.time()
BOOST_BASE = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST_BASE[:, k] = V10._pairwise_boost(rs[:, :k])
POS_BASE = build_pos(BOOST_BASE)
base_scs = scs_curve(POS_BASE)
base_wo, base_wn = wscore(POS_BASE, *OLD), wscore(POS_BASE, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding. This sanity check underlies EVERY idea below "
          "(all 6 share this exact precompute; only the boost/leader-detection call site differs).")


# =========================================================================================
# Generic helper: _pairwise_boost's own per-follower body (scale/power transform, IC>0 gate,
# apply lead_boost[-1]), lifted out and generalized to combine MULTIPLE weighted candidates.
# At a single candidate with weight 1.0, this is bit-identical to _pairwise_boost's inner loop.
# =========================================================================================
def combine_leaders(rsl, j, T, cand_weight_pairs):
    contribs, weights = [], []
    for i, w in cand_weight_pairs:
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        contribs.append(lead_boost[-1]); weights.append(w)
    if not contribs:
        return 0.0
    return float(np.average(contribs, weights=weights))


def sig_threshold_n(n_samples, n_candidates):
    """Same formula as V10._sig_threshold, generalized to an arbitrary simultaneous-test count
    (needed whenever the candidate pool size isn't the fixed BOOST_N_CANDIDATES=39), matching the
    convention already established in test_v19cand_boost_ncandidates.py."""
    if n_samples < 10:
        return 1.0
    alpha_adj = BOOST_ALPHA / max(n_candidates, 1)
    tcrit = V10.stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


# =========================================================================================
# C41 / C42: top-K candidate leaders averaged, weighted by |corr|. K=1 must reduce EXACTLY to
# V10._pairwise_boost (same argmax-then-threshold selection, same combine body at a single pair).
# =========================================================================================
def boost_topK_at_day(k, K):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        valid = ~np.isnan(col)
        if not valid.any(): continue
        idxs = np.where(valid)[0]
        order = idxs[np.argsort(-np.abs(col[idxs]))]
        picked = [ci for ci in order if abs(col[ci]) > thr][:K]
        if not picked: continue
        pairs = [(int(cand_idx[ci]), float(abs(col[ci]))) for ci in picked]
        boost[j] = combine_leaders(rsl, j, T, pairs)
    return boost


# =========================================================================================
# C43: two-hop transitive boost.
# =========================================================================================
def boost_twohop_at_day(k):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    pos_of = {int(name): p for p, name in enumerate(cand_idx)}
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    direct_leader = {}; direct_w = {}
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)):
            direct_leader[j] = None; continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            direct_leader[j] = None; continue
        direct_leader[j] = int(cand_idx[ci]); direct_w[j] = float(abs(col[ci]))
    for j in range(n):
        pairs = []
        B = direct_leader.get(j)
        if B is not None:
            pairs.append((B, direct_w[j]))
            A = direct_leader.get(B)
            if A is not None and A != j and A != B and A in pos_of:
                v = C[pos_of[A], j]
                if not np.isnan(v) and abs(v) > thr:
                    pairs.append((A, float(abs(v))))
        if not pairs: continue
        boost[j] = combine_leaders(rsl, j, T, pairs)
    return boost


# =========================================================================================
# C44: cluster-restricted candidate pool (k-means on the causal all-pairs corr matrix).
# =========================================================================================
from sklearn.cluster import KMeans


def boost_cluster_at_day(k, n_clusters):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    corr_all = np.corrcoef(Xi_full)
    corr_all = np.nan_to_num(corr_all, nan=0.0)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=0).fit(corr_all)
    labels = km.labels_
    C_full = V10._corrmat(Xi_full, Yj)
    idxall = np.arange(n)
    for j in range(n):
        pool = idxall[(labels == labels[j]) & (idxall != j)]
        if len(pool) == 0: continue
        thr = sig_threshold_n(n_samples, len(pool))
        col = C_full[pool, j]
        ci_local = int(np.argmax(np.abs(col)))
        if abs(col[ci_local]) <= thr: continue
        i = int(pool[ci_local])
        boost[j] = combine_leaders(rsl, j, T, [(i, float(abs(col[ci_local])))])
    return boost


# =========================================================================================
# C45: distance correlation (vectorized), rolling W=250-day window (approximation, see docstring).
# =========================================================================================
DCOR_W = 250


def _dmat_flat(rows):
    m, W = rows.shape
    diff = np.abs(rows[:, :, None] - rows[:, None, :])
    rmean = diff.mean(axis=2, keepdims=True)
    cmean = diff.mean(axis=1, keepdims=True)
    tmean = diff.mean(axis=(1, 2), keepdims=True)
    A = diff - rmean - cmean + tmean
    return A.reshape(m, W * W)


def dcor_matrix(Xrows, Yrows):
    W = Xrows.shape[1]
    AX = _dmat_flat(Xrows); AY = _dmat_flat(Yrows)
    dcov2 = (AX @ AY.T) / (W * W)
    varX = (AX * AX).sum(1) / (W * W)
    varY = (AY * AY).sum(1) / (W * W)
    denom = np.sqrt(np.outer(varX, varY))
    denom = np.where(denom < 1e-24, np.nan, denom)
    dcor2 = np.clip(dcov2 / denom, 0, None)
    return np.sqrt(dcor2)


def boost_dcor_at_day(k):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    if n_samples < DCOR_W + 10:
        return boost
    thr = V10._sig_threshold(DCOR_W)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xw = np.nan_to_num(Xi_full[cand_idx, -DCOR_W:], nan=0.0)
    Yw = np.nan_to_num(Yj[:, -DCOR_W:], nan=0.0)
    D = dcor_matrix(Xw, Yw)
    for j in range(n):
        col = D[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)): continue
        colf = np.nan_to_num(col, nan=-1.0)
        ci = int(np.argmax(colf))
        if np.isnan(col[ci]) or col[ci] <= thr: continue
        i = int(cand_idx[ci])
        boost[j] = combine_leaders(rsl, j, T, [(i, float(col[ci]))])
    return boost


# =========================================================================================
# C46: graphical-lasso sparse inverse covariance, refit every GLASSO_REFIT_EVERY days (CV fit is
# too costly to refit daily for a screening pass), graph held fixed between refits.
# =========================================================================================
from sklearn.covariance import GraphicalLassoCV

GLASSO_REFIT_EVERY = 25
_glasso_state = {"day": -10 ** 9, "adj": None}


def get_glasso_adj(k, Xi_full):
    if k - _glasso_state["day"] >= GLASSO_REFIT_EVERY:
        Z = Xi_full.T
        Z = Z[-600:]
        Z = np.nan_to_num(Z, nan=0.0)
        try:
            m = GraphicalLassoCV(cv=3, max_iter=200).fit(Z)
            prec = m.precision_
            d = np.sqrt(np.clip(np.diag(prec), 1e-24, None))
            pcorr = -prec / np.outer(d, d)
            np.fill_diagonal(pcorr, 0.0)
            adj = np.abs(pcorr) > 1e-8
        except Exception:
            adj = np.zeros((Xi_full.shape[0], Xi_full.shape[0]), dtype=bool)
        _glasso_state["day"] = k; _glasso_state["adj"] = adj
    return _glasso_state["adj"]


def boost_glasso_at_day(k):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    if n_samples < 100:
        return boost
    adj = get_glasso_adj(k, Xi_full)
    C_full = V10._corrmat(Xi_full, Yj)
    for j in range(n):
        neighbors = np.where(adj[j])[0]
        neighbors = neighbors[neighbors != j]
        if len(neighbors) == 0: continue
        thr = sig_threshold_n(n_samples, len(neighbors))
        col = C_full[neighbors, j]
        ci_local = int(np.argmax(np.abs(col)))
        if abs(col[ci_local]) <= thr: continue
        i = int(neighbors[ci_local])
        boost[j] = combine_leaders(rsl, j, T, [(i, float(abs(col[ci_local])))])
    return boost


# =========================================================================================
# RUN EVERYTHING
# =========================================================================================
results = {}

print("\n=== C41/C42 (+ K=1 internal check): top-K candidate leaders averaged, weighted by |corr| ===",
      flush=True)
for K in (1, 2, 3):
    t0 = time.time()
    BOOST = np.zeros((nIdio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        BOOST[:, k] = boost_topK_at_day(k, K)
    nm = f"topK={K}"
    if K == 1:
        max_abs_diff = float(np.nanmax(np.abs(BOOST - BOOST_BASE)))
        print(f"  [internal check] K=1 vs V10._pairwise_boost: max|diff|={max_abs_diff:.3g} "
              f"(should be ~0)", flush=True)
    res = evaluate(nm, BOOST, base_wo, base_wn, base_scs)
    print(f"    [{time.time()-t0:.0f}s]", flush=True)
    results[nm] = res

print("\n=== C43: two-hop transitive boost ===", flush=True)
t0 = time.time()
BOOST43 = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST43[:, k] = boost_twohop_at_day(k)
results["twohop"] = evaluate("twohop", BOOST43, base_wo, base_wn, base_scs)
print(f"    [{time.time()-t0:.0f}s]", flush=True)

print("\n=== C44: cluster-restricted candidate pool (k-means, k in {4,5,6}) ===", flush=True)
for ncl in (4, 5, 6):
    t0 = time.time()
    BOOST44 = np.zeros((nIdio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        BOOST44[:, k] = boost_cluster_at_day(k, ncl)
    nm = f"cluster_k{ncl}"
    results[nm] = evaluate(nm, BOOST44, base_wo, base_wn, base_scs)
    print(f"    [{time.time()-t0:.0f}s]", flush=True)

print(f"\n=== C45: distance correlation (rolling W={DCOR_W}) ===", flush=True)
t0 = time.time()
BOOST45 = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST45[:, k] = boost_dcor_at_day(k)
results["dcor"] = evaluate("dcor", BOOST45, base_wo, base_wn, base_scs)
print(f"    [{time.time()-t0:.0f}s]", flush=True)

print(f"\n=== C46: graphical lasso sparse precision (refit every {GLASSO_REFIT_EVERY}d) ===", flush=True)
t0 = time.time()
BOOST46 = np.zeros((nIdio, nt))
_glasso_state["day"] = -10 ** 9
for k in range(BOOST_MIN_DAY, nt):
    BOOST46[:, k] = boost_glasso_at_day(k)
results["glasso"] = evaluate("glasso", BOOST46, base_wo, base_wn, base_scs)
print(f"    [{time.time()-t0:.0f}s]", flush=True)

print("\n=== SUMMARY ===")
print(f"baseline: OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")
for nm, r_ in results.items():
    tag = "PASS" if r_["passed"] else "fail"
    print(f"  {nm:<12} OLD={r_['wo']:>7.1f} NEW={r_['wn']:>7.1f} rmean={r_['rm']:>7.1f} "
          f"rfloor={r_['rf']:>7.1f} n_worse={r_['nworse']}/61  [{tag}]")
print("\nDONE")
