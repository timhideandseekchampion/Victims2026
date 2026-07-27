"""H3 (plan: revisit the pairwise family): SAFE_llboost.py re-estimates each stock's leader fresh
from all history on every call. Does a leader relationship that has stayed IDENTICAL for many
consecutive days carry a more reliable signal than one that just flipped? Tracks, for each stock, a
"days since last leader change" counter (using the best_leader argmax, independent of significance,
so the stability measure reflects the underlying relationship's persistence, not just whether it
happened to clear the bar today), then splits the boost's own predictive power (target = rs[j,k],
matching the alignment confirmed correct in test_h1_reciprocal_pairs.py / test_h2) by stability
bucket, with permutation + H1/H2 persistence in each bucket.
"""
import numpy as np, pandas as pd, time
from scipy import stats

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
rs = r[1:]
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


print("=== precompute: day-by-day best-leader map + running stability counter per stock ===")
t0 = time.time()
run_leader = np.full(n, -1, dtype=int)
run_len = np.zeros(n, dtype=int)
rows = []  # (k, j, boost_val, stability, target)
for k in range(BOOST_MIN_DAY, min(nt, T)):
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
    # update running stability BEFORE using it (today's leader includes today's info, causal/valid)
    for j in range(n):
        if best_leader[j] == run_leader[j]:
            run_len[j] += 1
        else:
            run_leader[j] = best_leader[j]
            run_len[j] = 1
    for j in range(n):
        if abs(best_corr[j]) <= thr:
            continue
        i = best_leader[j]
        lead = rs[i, :Tn]
        scale = np.nanstd(lead[max(0, Tn - 1 - BOOST_SCALE_W):Tn - 1]) + 1e-12
        boost_val = float(np.sign(lead[-1]) * (np.abs(lead[-1]) / scale) ** BOOST_P)
        rows.append((k, j, boost_val, run_len[j], rs[j, k]))
print(f"done ({time.time()-t0:.0f}s); {len(rows)} significant-pair-day instances")

arr = np.array([(x[0], x[2], x[3], x[4]) for x in rows])
ks, bv, stab, target = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
print(f"\nstability distribution: min={stab.min():.0f} median={np.median(stab):.0f} "
      f"mean={stab.mean():.1f} max={stab.max():.0f}")
print(f"quartile boundaries: {np.percentile(stab, [25,50,75])}")


def pooled_ic_perm(X, Y, K, label, n_perm=300):
    if len(X) < 30:
        print(f"{label}: too few samples (n={len(X)}), skipping")
        return
    ic = float(np.corrcoef(X, Y)[0, 1])
    med_k = np.median(K)
    m1 = K < med_k; m2 = ~m1
    ic1 = float(np.corrcoef(X[m1], Y[m1])[0, 1]) if m1.sum() > 20 else float('nan')
    ic2 = float(np.corrcoef(X[m2], Y[m2])[0, 1]) if m2.sum() > 20 else float('nan')
    rng = np.random.default_rng(0)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        perm_ics[p] = np.corrcoef(X, rng.permutation(Y))[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f} (n={len(X)})  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}")


print("\n=== Stage 1: boost -> target, split by leader-stability quartile ===")
q = np.percentile(stab, [25, 50, 75])
buckets = [
    ("Q1 (least stable)", stab <= q[0]),
    ("Q2", (stab > q[0]) & (stab <= q[1])),
    ("Q3", (stab > q[1]) & (stab <= q[2])),
    ("Q4 (most stable)", stab > q[2]),
]
for name, mask in buckets:
    pooled_ic_perm(bv[mask], target[mask], ks[mask], f"  {name} (stab in this bucket)")

print("\n=== reference: all pooled (no stability split) ===")
pooled_ic_perm(bv, target, ks, "  all")
