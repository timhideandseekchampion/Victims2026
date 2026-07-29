"""
test_batch100_catB_altmetrics.py

Batch-100, category B: 9 "swap one piece of the leader-selection step inside `_pairwise_boost`"
candidates, sharing the same overall skeleton V10._pairwise_boost already uses -- candidate pool =
top BOOST_N_CANDIDATES=39 idio names by trailing (causal) vol, pick the single best-by-metric leader
per follower among those candidates, apply a significance/validity gate, `ic<=0: discard`, then the
SAME convex transform `sign(x)*(|x|/scale)**BOOST_P` on the chosen leader's own last-return value.
Only the leader-SELECTION metric/methodology changes per idea; the candidate pool, the downstream
realized-IC recheck, and the convex transform are reused verbatim (`generic_boost()` below lifts that
shared body out of `_pairwise_boost` once and takes a pluggable `score_fn` + optional `apply_fn`,
rather than duplicating the whole function 9 times).

Uses the pre-built, verified `_v10_harness.py`: WZ_PRE (ridge+REV, pre-boost) is reused unmodified;
each variant only replaces the BOOST array, combined as `wz = WZ_PRE[:,t] + BOOST_K*B[:,t]`, then
`rs_blend(wz,t)` (the real production rank-stability blend), then scored with `evaluate()` -- exactly
v10's own OLD/NEW/rolling-mean/rolling-floor/n_worse bar.

IMPORTANT HONESTY NOTE on overlap with prior work (checked before writing this file, since the
assignment brief said these metrics were "not yet tried"):
  - Idea 5 (distance correlation) substantially OVERLAPS a prior test, C45 in
    test_batch100_C41_C46.py, which also used a rolling W=250 window + the same "reuse the Pearson
    Bonferroni threshold as a magnitude-matched cutoff" approximation for tractability, for the same
    reason (dCor's true null distribution isn't a t-distribution). Re-tested here anyway (assignment
    asked for it explicitly), same window convention, independently re-implemented via a
    double-centering + single-matmul vectorization for speed rather than C45's per-day loop.
  - Idea 9 (partial correlation controlling for ALGO's same-day return AT the leader-selection step)
    substantially OVERLAPS a prior test, C53 in test_batch100_boostvariants.py -- essentially the
    same idea, same residualize-then-correlate construction. Re-tested here with one refinement: the
    significance threshold uses the CORRECT df=n-3 (partial corr loses one extra degree of freedom
    from the control variable) instead of C53's explicitly-flagged approximation of ignoring that.
  - Idea 3 (Granger-style selection) is DIFFERENT from a prior test, C47 in test_batch100_C47.py:
    C47 keeps Pearson argmax SELECTION and only replaces the significance GATE on the
    already-selected candidate with a real statsmodels Granger test. This idea instead uses the
    Granger-equivalent statistic (see below) AS the selection criterion across all 39 candidates, per
    the assignment's literal wording ("select the candidate with the strongest such improvement").
  - Ideas 1 (asymmetric lead-lag), 2 (decayed multi-day), 4 (split-sample validation), 6 (mutual
    information), 7 (Kendall's tau), 8 (tail-dependence) were NOT found anywhere in this repo's prior
    test files (grepped case-insensitively for their key terms across all *.py) -- genuinely new to
    this session, as far as this search could tell.

THE 9 IDEAS:

1. Asymmetric lead-lag by direction. Split each candidate's OWN return history into "leader-up" and
   "leader-down" days (by sign of the candidate's own return), correlate leader->follower separately
   on each subsample, and use whichever direction's estimate matches the candidate's ACTUAL sign as
   of the most recent day. Per-candidate significance uses the actual subsample size (min 30 days per
   direction, else that candidate is skipped for that follower -- chosen to keep at least half of the
   `ok.sum()<60` minimum-sample convention used elsewhere in this file's history, since splitting by
   sign roughly halves the sample).

2. Decayed multi-day leader signal. SELECTION is untouched (identical Pearson argmax + Bonferroni
   gate to V10). Only the boost VALUE applied changes: instead of the leader's single last-day
   convex-transformed value, use an EW-decayed average of its last 3 days' transformed values with
   decay ~0.5/day (weights 1, 0.5, 0.25, normalized).

3. Granger-causality-style selection. For each (candidate, follower) pair, tests whether the
   candidate's lag-1 return improves a regression of the follower's forward return beyond the
   follower's OWN lag-1 return alone. For a single added regressor this maxlag=1 Granger F-test is
   algebraically equivalent to the PARTIAL correlation of the candidate's lag-1 return with the
   follower's forward return, controlling for the follower's own lag-1 return (F = t**2, t derived
   from the partial-r, same identity used elsewhere in this repo, e.g. C53's partial-corr construction
   for a different control variable) -- implemented that way instead of calling statsmodels ~1M times
   (39 candidates x 50 followers x ~520 days), which would be a large multiple of every other idea's
   compute budget for an identical test statistic. Select the candidate maximizing |partial r| per
   follower; require it clears a Bonferroni-corrected threshold at df=n-3 (one extra dof lost to the
   control variable, adjusted from the plain corr threshold V10 itself uses).

4. Split-sample leader validation. At day t, split the causal (X_s, Y_{s+1}) history in half. Pick
   the best-|corr| leader using ONLY the first half (also required to be nominally significant on
   that half's own sample size -- otherwise "best" could just be the least-noisy garbage). Then
   REQUIRE that SAME (leader, follower) pair to independently clear the significance bar on the
   SECOND half alone (both halves strictly before t, own sample-size-adjusted threshold). No fallback
   search if the first-half pick fails the second-half check -- that day's boost for that follower is
   simply zero, matching "before trusting it" rather than a joint two-stage optimization over all
   candidates. Strictly stronger than BOOST_MIN_DAY's single-sample-size gate.

5. Distance correlation (standard double-centering estimator) instead of Pearson. APPROXIMATED for
   tractability, same convention as prior test C45 (see overlap note above): computed on a rolling
   W=250-day window rather than the full growing causal history, and the resulting dCor (already on a
   [0,1] magnitude scale comparable to |Pearson r|) is compared against the same Bonferroni-corrected
   Pearson threshold at n=W. Implemented via one double-centered-distance-matrix build per series
   (vectorized over all 39 candidates / 50 followers at once) then a single matmul to get the full
   (39,50) score matrix per day -- this is what keeps 520 days of O(W^2) distance matrices tractable
   (a naive per-pair Python loop calling a generic dCor routine 1950x/day would not be).

6. Mutual information: a SIMPLE histogram-binned plug-in estimator (5x5 equal-frequency bins), also
   on a rolling W=250-day window for the same tractability reason (binned MI needs a stable window to
   pick bin edges from, and this repo's established convention windows expensive dependence metrics
   rather than growing them). Converted to a Pearson-comparable magnitude via the standard Gaussian-
   copula identity rho_equiv = sqrt(1 - exp(-2*MI)) so the same magnitude-matched Bonferroni cutoff
   applies (same trick as C45/idea 5). CAVEAT, stated honestly: histogram plug-in MI has a well-known
   upward finite-sample bias (worse with more bins / less data), so this significance gate is
   optimistic relative to a bias-corrected or kNN estimator -- kept small bin count (5) specifically
   to limit that bias, but it is not eliminated.

7. Kendall's tau instead of Pearson. Computed EXACTLY (not approximated) via a vectorized tie-free
   tau-a estimator: build the sign(x_i - x_j) matrix for every series (ties are ~impossible on
   continuous float log-returns) once per candidate/follower, flatten, and get the full (39,50) score
   matrix via a single matmul -- the same vectorization trick as idea 5. Windowed at W=250 anyway (not
   for tractability this time -- the matmul trick makes even the full growing history cheap -- but for
   comparability with the other rank/nonlinear metrics in this file, all using the same window).
   Converted to a Pearson-comparable magnitude via rho_equiv = sin(pi*tau/2) (the standard Gaussian-
   copula tau<->rho relation) for the same magnitude-matched threshold.

8. Tail-dependence. For each candidate, take its largest-magnitude 10% of days (by |return|, min 30
   days), and measure how often the FOLLOWER's contemporaneous-next-day return has the SAME sign as
   the candidate's tail-day return (empirical hit-rate vs 50%). Gated by a two-sided binomial
   (normal-approximation) z-test on that hit-rate at the same Bonferroni-corrected alpha. Select the
   candidate with the largest |hit-rate - 0.5| per follower. (The downstream ic<=0 realized-IC recheck
   is untouched and will still discard a pair whose resulting lead_boost doesn't positively predict
   the follower going forward, even if the tail hit-rate happened to run the "wrong" way.)

9. Partial correlation controlling for ALGO's same-day return, computed AT the leader-selection step
   -- NOT "ALGO as leader" (a different, already-rejected idea): ALGO here is a control variable, never
   a boost candidate. Residualize each candidate's return (at day s) and each follower's forward
   return (at day s+1) against ALGO's OWN same-day return (at s and s+1 respectively), then correlate
   the residuals. See overlap note above re: prior test C53.

Every variant reuses V10's own BOOST_MIN_DAY/BOOST_N_CANDIDATES/BOOST_IC_L/BOOST_P/BOOST_SCALE_W/
BOOST_ALPHA/BOOST_K, WZ_PRE (idio ridge+REV), and rs_blend (rank-stability blend) verbatim -- nothing
upstream or downstream of the boost step is touched by this file.
"""
import time
import numpy as np
from scipy import stats
import _v10_harness as H

V10 = H.V10

# ==================================================================================================
# shared skeleton: candidate pool + downstream convex transform + ic<=0 realized-IC gate, lifted out
# of V10._pairwise_boost verbatim; only score_fn (leader SELECTION) and, optionally, apply_fn (how
# the chosen leader's transformed series maps to today's applied value) differ per idea.
# ==================================================================================================
def generic_boost(rs_k, score_fn, apply_fn=None):
    """score_fn(Xi, Yj_all, rs_k, cand_idx) -> (n_cand, n) score matrix, ALREADY NaN'd out wherever
    that (candidate, follower) pair fails its own significance/validity gate -- caller just takes
    argmax(|score|) per follower column (no separate outer threshold: every score_fn below does its
    own gating internally since different metrics need different-shaped significance tests)."""
    n, T = rs_k.shape
    boost = np.zeros(n)
    if T < V10.BOOST_MIN_DAY:
        return boost
    Xi_full = rs_k[:, :-1]
    Yj_all = rs_k[:, 1:]
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V10.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    S = score_fn(Xi, Yj_all, rs_k, cand_idx)
    for j in range(n):
        col = S[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V10.BOOST_P
        val = lead_boost[-1] if apply_fn is None else apply_fn(lead_boost)
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]
        ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = val
    return boost


def _thr(n_samples, extra_dof=0):
    """Bonferroni-corrected (over BOOST_N_CANDIDATES simultaneous tests) minimum |corr| at
    BOOST_ALPHA, given the actual sample size -- same formula V10._sig_threshold uses, generalized
    with an optional extra degrees-of-freedom loss for partial-correlation-style controls."""
    df = n_samples - 2 - extra_dof
    if df < 8:
        return 1.0
    alpha_adj = V10.BOOST_ALPHA / V10.BOOST_N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=df)
    return float(tcrit / np.sqrt(df + tcrit ** 2))


# ==================================================================================================
# idea 1: asymmetric lead-lag by direction
# ==================================================================================================
def score_asym(Xi, Yj_all, rs_k, cand_idx):
    n_cand, n_samples = Xi.shape
    n_follower = Yj_all.shape[0]
    S = np.full((n_cand, n_follower), np.nan)
    for ci in range(n_cand):
        x = Xi[ci]
        cur_sign = np.sign(x[-1])
        if cur_sign == 0:
            continue
        mask = (x > 0) if cur_sign > 0 else (x < 0)
        m = int(mask.sum())
        if m < 30:
            continue
        xm = x[mask]
        xc = xm - xm.mean()
        sxc = xc.std()
        if sxc < 1e-12:
            continue
        Ym = Yj_all[:, mask]
        Yc = Ym - Ym.mean(1, keepdims=True)
        sYc = Yc.std(1) + 1e-12
        corr = (Yc @ xc) / m / (sYc * sxc)
        thr = _thr(m)
        S[ci] = np.where(np.abs(corr) > thr, corr, np.nan)
    return S


# ==================================================================================================
# idea 2: decayed multi-day leader signal -- SELECTION unchanged (plain Pearson, self-gated exactly
# like V10), only apply_fn (the value applied) differs.
# ==================================================================================================
def score_pearson(Xi, Yj_all, rs_k, cand_idx):
    C = V10._corrmat(Xi, Yj_all)
    thr = _thr(Xi.shape[1])
    return np.where(np.abs(C) > thr, C, np.nan)


def apply_decayed(lead_boost):
    w = np.array([1.0, 0.5, 0.25])
    w = w / w.sum()
    vals = np.array([lead_boost[-1], lead_boost[-2], lead_boost[-3]])
    return float((w * vals).sum())


# ==================================================================================================
# idea 3: Granger-style selection (partial corr of candidate lag-1 vs follower forward return,
# controlling for the FOLLOWER'S OWN lag-1 -- the maxlag=1 Granger F-test's closed-form equivalent).
# ==================================================================================================
def score_granger(Xi, Yj_all, rs_k, cand_idx):
    Xi_full_all = rs_k[:, :-1]  # ALL 50 names' lag-1 returns -- needed for each follower's own lag
    n_cand, n_samples = Xi.shape
    n_follower = Yj_all.shape[0]
    S = np.full((n_cand, n_follower), np.nan)
    thr = _thr(n_samples, extra_dof=1)
    for j in range(n_follower):
        z = Xi_full_all[j]
        zc = z - z.mean()
        vz = (zc * zc).sum() + 1e-12
        Xc = Xi - Xi.mean(1, keepdims=True)
        b_x = (Xc @ zc) / vz
        Xr = Xc - b_x[:, None] * zc[None, :]
        y = Yj_all[j]
        yc = y - y.mean()
        b_y = (yc @ zc) / vz
        yr = yc - b_y * zc
        sx = Xr.std(1) + 1e-12
        sy = yr.std() + 1e-12
        pc = (Xr @ yr) / n_samples / (sx * sy)
        S[:, j] = np.where(np.abs(pc) > thr, pc, np.nan)
    return S


# ==================================================================================================
# idea 4: split-sample leader validation
# ==================================================================================================
def score_splitsample(Xi, Yj_all, rs_k, cand_idx):
    n_cand, n_samples = Xi.shape
    n_follower = Yj_all.shape[0]
    mid = n_samples // 2
    S = np.full((n_cand, n_follower), np.nan)
    if mid < 60 or (n_samples - mid) < 60:
        return S
    C1 = V10._corrmat(Xi[:, :mid], Yj_all[:, :mid])
    C2 = V10._corrmat(Xi[:, mid:], Yj_all[:, mid:])
    thr1 = _thr(mid)
    thr2 = _thr(n_samples - mid)
    for j in range(n_follower):
        col1 = C1[:, j]
        if np.all(np.isnan(col1)):
            continue
        ci = int(np.nanargmax(np.abs(col1)))
        if abs(col1[ci]) <= thr1:
            continue
        if abs(C2[ci, j]) <= thr2:
            continue
        S[ci, j] = col1[ci]
    return S


# ==================================================================================================
# idea 5: distance correlation (double-centering estimator), rolling W=250 window (see overlap note
# re: prior test C45), vectorized via one matmul across all (candidate, follower) pairs per day.
# ==================================================================================================
DCOR_W = 250


def _dcenter(Z):
    D = np.abs(Z[:, :, None] - Z[:, None, :])
    A = D - D.mean(2, keepdims=True) - D.mean(1, keepdims=True) + D.mean((1, 2), keepdims=True)
    return A.reshape(Z.shape[0], -1)


def score_dcor(Xi, Yj_all, rs_k, cand_idx):
    n_cand, n_samples = Xi.shape
    W = min(DCOR_W, n_samples)
    X = Xi[:, -W:]
    Y = Yj_all[:, -W:]
    Ax = _dcenter(X)
    Ay = _dcenter(Y)
    a = (Ax @ Ay.T) / (W * W)
    b = (Ax * Ax).sum(1) / (W * W)
    c = (Ay * Ay).sum(1) / (W * W)
    denom = np.sqrt(np.outer(b, c)) + 1e-18
    dcor = np.sqrt(np.clip(a, 0, None) / denom)
    thr = _thr(W)
    return np.where(dcor > thr, dcor, np.nan)


# ==================================================================================================
# idea 6: mutual information, simple 5x5 equal-frequency histogram plug-in estimator, W=250 window,
# converted to a Pearson-comparable magnitude for the shared Bonferroni cutoff.
# ==================================================================================================
MI_W = 250
MI_BINS = 5


def _qbins(x, nbins):
    qs = np.quantile(x, np.linspace(0, 1, nbins + 1))
    qs[0] -= 1e-9
    qs[-1] += 1e-9
    return np.clip(np.digitize(x, qs[1:-1]), 0, nbins - 1)


def score_mi(Xi, Yj_all, rs_k, cand_idx):
    n_cand, n_samples = Xi.shape
    n_follower = Yj_all.shape[0]
    W = min(MI_W, n_samples)
    X = Xi[:, -W:]
    Y = Yj_all[:, -W:]
    xb = np.array([_qbins(X[i], MI_BINS) for i in range(n_cand)])
    yb = np.array([_qbins(Y[j], MI_BINS) for j in range(n_follower)])
    thr = _thr(W)
    S = np.full((n_cand, n_follower), np.nan)
    for i in range(n_cand):
        xi_b = xb[i]
        for j in range(n_follower):
            idx = xi_b * MI_BINS + yb[j]
            counts = np.bincount(idx, minlength=MI_BINS * MI_BINS).astype(float).reshape(MI_BINS, MI_BINS)
            pxy = counts / W
            px = pxy.sum(1, keepdims=True)
            py = pxy.sum(0, keepdims=True)
            denom = px @ py
            nz = pxy > 0
            mi = float((pxy[nz] * np.log(pxy[nz] / denom[nz])).sum())
            mi = max(mi, 0.0)
            rho_eq = np.sqrt(1 - np.exp(-2 * mi))
            S[i, j] = rho_eq if rho_eq > thr else np.nan
    return S


# ==================================================================================================
# idea 7: Kendall's tau, exact tie-free tau-a via vectorized sign-matrix + matmul, W=250 window,
# converted to a Pearson-comparable magnitude via the Gaussian-copula rho<->tau relation.
# ==================================================================================================
KEND_W = 250


def score_kendall(Xi, Yj_all, rs_k, cand_idx):
    n_cand, n_samples = Xi.shape
    n_follower = Yj_all.shape[0]
    W = min(KEND_W, n_samples)
    X = Xi[:, -W:].astype(np.float32)
    Y = Yj_all[:, -W:].astype(np.float32)
    Sx = np.sign(X[:, :, None] - X[:, None, :]).reshape(n_cand, -1)
    Sy = np.sign(Y[:, :, None] - Y[:, None, :]).reshape(n_follower, -1)
    concord = Sx @ Sy.T
    tau = concord / (W * (W - 1))
    rho_eq = np.sin(np.pi * tau / 2)
    thr = _thr(W)
    return np.where(np.abs(rho_eq) > thr, rho_eq, np.nan)


# ==================================================================================================
# idea 8: tail-dependence -- candidate's largest-|return| 10% of days, sign-match hit-rate vs 50%,
# gated by a two-sided binomial (normal-approx) z-test at the same Bonferroni alpha.
# ==================================================================================================
TAIL_FRAC = 0.10


def score_taildep(Xi, Yj_all, rs_k, cand_idx):
    n_cand, n_samples = Xi.shape
    n_follower = Yj_all.shape[0]
    S = np.full((n_cand, n_follower), np.nan)
    z_crit = stats.norm.ppf(1 - (V10.BOOST_ALPHA / V10.BOOST_N_CANDIDATES) / 2)
    for ci in range(n_cand):
        x = Xi[ci]
        k_tail = max(30, int(np.ceil(TAIL_FRAC * n_samples)))
        tail_idx = np.argsort(-np.abs(x))[:k_tail]
        xt = np.sign(x[tail_idx])
        m = tail_idx.size
        Yt = np.sign(Yj_all[:, tail_idx])
        match = (Yt == xt[None, :]).mean(1)
        z = (match - 0.5) / (0.5 / np.sqrt(m))
        S[ci] = np.where(np.abs(z) > z_crit, match - 0.5, np.nan)
    return S


# ==================================================================================================
# idea 9: partial correlation controlling for ALGO's same-day return, AT the leader-selection step
# (overlaps prior test C53 -- see honesty note in module docstring).
# ==================================================================================================
def make_score_partial_algo(algo_r_full):
    def score_fn(Xi, Yj_all, rs_k, cand_idx):
        T = rs_k.shape[1]
        algo_r_k = algo_r_full[:T]
        zx = algo_r_k[:-1]
        zy = algo_r_k[1:]

        def resid(A, z):
            zc = z - z.mean()
            vz = (zc * zc).sum() + 1e-12
            Ac = A - A.mean(1, keepdims=True)
            b = (Ac @ zc) / vz
            return Ac - b[:, None] * zc[None, :]

        Xr = resid(Xi, zx)
        Yr = resid(Yj_all, zy)
        Xs = Xr / (Xr.std(1, keepdims=True) + 1e-12)
        Ys = Yr / (Yr.std(1, keepdims=True) + 1e-12)
        C = (Xs @ Ys.T) / Xi.shape[1]
        thr = _thr(Xi.shape[1], extra_dof=1)
        return np.where(np.abs(C) > thr, C, np.nan)

    return score_fn


# ==================================================================================================
# run all 9 variants
# ==================================================================================================
VARIANTS = [
    ("1_asymmetric_leadlag", score_asym, None),
    ("2_decayed_multiday", score_pearson, apply_decayed),
    ("3_granger_style_selection", score_granger, None),
    ("4_split_sample_validation", score_splitsample, None),
    ("5_distance_correlation", score_dcor, None),
    ("6_mutual_information", score_mi, None),
    ("7_kendall_tau", score_kendall, None),
    ("8_tail_dependence", score_taildep, None),
    ("9_partial_corr_vs_algo", make_score_partial_algo(H.algo_r_full), None),
]

print("\n=== batch100 cat B: 9 alt-metric leader-selection variants vs shipped v10 ===")
results = []
for name, score_fn, apply_fn in VARIANTS:
    t0 = time.time()
    B = np.zeros((H.nIdio, H.nt))
    for k in range(V10.BOOST_MIN_DAY, H.nt):
        B[:, k] = generic_boost(H.rs_full[:, :k], score_fn, apply_fn)
    WZ_full = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        wz = H.WZ_PRE[:, t] + V10.BOOST_K * B[:, t]
        WZ_full[:, t] = H.rs_blend(wz, t)
    res = H.evaluate(name, WZ_full)
    secs = time.time() - t0
    res["secs"] = secs
    print(f"    [{secs:.1f}s]", flush=True)
    results.append(res)

passing = [r for r in results if r["passed"]]
print(f"\n{len(passing)}/{len(results)} of the 9 alt-metric leader-selection ideas beat v10 on "
      f"OLD+NEW+rmean jointly.")
if passing:
    for r in passing:
        print(f"  PASS: {r['name']}  OLD={r['wo']:.1f}  NEW={r['wn']:.1f}  rmean={r['rm']:.1f}  "
              f"rfloor={r['rf']:.1f}  n_worse={r['nworse']}/61")
else:
    print("None passed. Ranked by rolling mean, closest first:")
    for r in sorted(results, key=lambda r: -r["rm"]):
        print(f"  {r['name']:<28} OLD={r['wo']:>7.1f}  NEW={r['wn']:>7.1f}  rmean={r['rm']:>7.1f}  "
              f"rfloor={r['rf']:>7.1f}  n_worse={r['nworse']}/61  [{r['secs']:.1f}s]")
