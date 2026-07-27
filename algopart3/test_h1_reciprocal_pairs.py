"""H1 (plan: revisit the pairwise family): does it matter whether a stock's significant leader
relationship is RECIPROCAL (its leader's own best match is it right back) vs ONE-DIRECTIONAL (its
leader's own best match is some other stock)? Mutual pairs are two independent argmax searches
agreeing -- plausibly a more genuine relationship than a one-directional pick, which could just mean
the leader happens to be the least-bad option among 48 noisy candidates for this one follower.

Stage 1 only: causal, day-by-day, using the EXACT significance-gate machinery from
SAFE_llboost.py/test_boost_subparam_sweep.py (fresh every day, no stale checkpoints). For every
currently-significant pair, tag it mutual or one-directional, then test the boost signal's OWN
predictive power (before any ic>0 sign-gate is applied) split by that tag -- pooled IC, permutation
(circular shift per stock), H1/H2 persistence.
"""
import numpy as np, pandas as pd, time
from scipy import stats

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
rs = r[1:]  # idio-stock returns, (49, T)
n, T = rs.shape

BOOST_MIN_DAY = 500
ALPHA = 0.05
N_CANDIDATES = 49
BOOST_P = 2.0
BOOST_SCALE_W = 1000


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== precompute: day-by-day full best-leader map (not just significant), tag mutual vs one-dir ===")
t0 = time.time()
# per day k: dict j -> (leader_i, is_significant, is_mutual, boost_value)
DAILY = {}
for k in range(BOOST_MIN_DAY, nt):
    Tn = k
    Xi = rs[:, :Tn - 1]; Yj = rs[:, 1:Tn]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    best_leader = np.full(n, -1, dtype=int)
    best_corr = np.zeros(n)
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        best_leader[j] = i
        best_corr[j] = col[i]
    entry = {}
    for j in range(n):
        if abs(best_corr[j]) <= thr:
            continue
        i = best_leader[j]
        mutual = bool(best_leader[i] == j)
        lead = rs[i, :Tn]
        scale = np.nanstd(lead[max(0, Tn - 1 - BOOST_SCALE_W):Tn - 1]) + 1e-12
        boost_val = float(np.sign(lead[-1]) * (np.abs(lead[-1]) / scale) ** BOOST_P)
        entry[j] = (i, mutual, boost_val)
    DAILY[k] = entry
print(f"done ({time.time()-t0:.0f}s)")

n_mutual = sum(1 for e in DAILY.values() for v in e.values() if v[1])
n_total = sum(len(e) for e in DAILY.values())
print(f"significant pairs found: {n_total}  (mutual: {n_mutual}, {n_mutual/n_total*100:.1f}%)")


def pooled_ic_perm(rows, label, n_perm=300):
    """rows: list of (j, t, boost_val) -> test boost_val -> rs[j, t+1]."""
    if len(rows) < 30:
        print(f"{label}: too few samples (n={len(rows)}), skipping")
        return
    X = np.array([x[2] for x in rows])
    Y = np.array([rs[x[0], x[1]] for x in rows])
    ok = ~np.isnan(X) & ~np.isnan(Y)
    X, Y = X[ok], Y[ok]
    ic = float(np.corrcoef(X, Y)[0, 1])
    n_ = len(X)
    ts = np.array([x[1] for x in rows])[ok]
    med_t = np.median(ts)
    m1 = ts < med_t; m2 = ~m1
    ic1 = float(np.corrcoef(X[m1], Y[m1])[0, 1]) if m1.sum() > 20 else float('nan')
    ic2 = float(np.corrcoef(X[m2], Y[m2])[0, 1]) if m2.sum() > 20 else float('nan')
    rng = np.random.default_rng(0)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        shift = rng.integers(1, n_ - 1)
        perm_ics[p] = np.corrcoef(X, np.roll(Y, shift))[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f} (n={n_})  H1={ic1:+.4f}  H2={ic2:+.4f}  "
          f"perm p={pval:.3f}  perm_std={perm_ics.std():.4f}")


# boost_val computed at day k uses rs[i, k-1] and (matching production's validated PnL alignment,
# POS[:,k] earns rs[:,k]) is INTENDED to predict rs[j, k] -- lag-1, not rs[j, k+1].
rows_mutual, rows_onedir = [], []
for k, entry in DAILY.items():
    if k >= T:  # rs has T columns (0..T-1); need rs[j, k] to be a valid index
        continue
    for j, (i, mutual, bv) in entry.items():
        row = (j, k, bv)
        (rows_mutual if mutual else rows_onedir).append(row)

print(f"\nmutual-pair rows: {len(rows_mutual)}   one-directional rows: {len(rows_onedir)}")
print("\n=== Stage 1: boost -> next-day return, MUTUAL pairs only ===")
pooled_ic_perm(rows_mutual, "  mutual")
print("\n=== Stage 1: boost -> next-day return, ONE-DIRECTIONAL pairs only ===")
pooled_ic_perm(rows_onedir, "  one-directional")
print("\n=== reference: ALL significant pairs pooled (current shipped population) ===")
pooled_ic_perm(rows_mutual + rows_onedir, "  all")
