"""H2 (plan: revisit the pairwise family): for a stock's already-established significant leader,
does that leader's return from TWO days ago add anything beyond yesterday's, once yesterday's is
already accounted for? Narrower, lower-dimensional version of the full-ridge lag-2 test
(test_lag2_ridge.py) that collapsed catastrophically -- here only ONE extra feature is added, and
only for the already-significant-pair population, not a global 51->102 feature doubling.

Uses the SAME significance-gate machinery as SAFE_llboost.py / test_boost_subparam_sweep.py (fresh
every day, no stale checkpoints), and the alignment CONFIRMED CORRECT via test_h1_reciprocal_pairs.py
(boost computed from rs[i, k-1] predicts rs[j, k] -- lag-1, matching production's validated PnL
convention exactly).

Stage 1: pool (lag1_raw, lag2_raw, target) triples across all currently-significant pairs, fit a
single pooled OLS target~lag1, take the residual, and correlate that residual against lag2 --
partial correlation testing whether lag2 explains anything lag1 doesn't. Permutation (global
shuffle of the residual/lag2 pairing) + H1/H2 persistence.
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


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== precompute: day-by-day significant pairs, collect (lag1_raw, lag2_raw, target) ===")
t0 = time.time()
rows = []  # (k, j, lag1, lag2, target)
for k in range(BOOST_MIN_DAY, min(nt, T)):
    Tn = k
    Xi = rs[:, :Tn - 1]; Yj = rs[:, 1:Tn]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr:
            continue
        if k < 2:
            continue
        lag1 = rs[i, k - 1]; lag2 = rs[i, k - 2]; target = rs[j, k]
        rows.append((k, j, lag1, lag2, target))
print(f"done ({time.time()-t0:.0f}s); {len(rows)} significant-pair-day instances")

rows_arr = np.array([(x[0], x[2], x[3], x[4]) for x in rows])
ks, lag1, lag2, target = rows_arr[:, 0], rows_arr[:, 1], rows_arr[:, 2], rows_arr[:, 3]

print("\n=== Stage 1: partial correlation of lag2 vs (target residualized on lag1) ===")
b = np.polyfit(lag1, target, 1)
resid = target - (b[0] * lag1 + b[1])
ic_full = float(np.corrcoef(lag2, resid)[0, 1])
print(f"raw corr(lag1, target) = {np.corrcoef(lag1, target)[0,1]:+.4f}  (sanity: should roughly match "
      f"the boost's own known predictive direction)")
print(f"partial corr(lag2, resid) = {ic_full:+.4f}  (n={len(rows)})")

med_k = np.median(ks)
m1 = ks < med_k; m2 = ~m1
b1 = np.polyfit(lag1[m1], target[m1], 1); resid1 = target[m1] - (b1[0]*lag1[m1]+b1[1])
ic1 = float(np.corrcoef(lag2[m1], resid1)[0, 1])
b2 = np.polyfit(lag1[m2], target[m2], 1); resid2 = target[m2] - (b2[0]*lag1[m2]+b2[1])
ic2 = float(np.corrcoef(lag2[m2], resid2)[0, 1])
print(f"H1 (early half): partial corr = {ic1:+.4f} (n={m1.sum()})")
print(f"H2 (late half):  partial corr = {ic2:+.4f} (n={m2.sum()})")

rng = np.random.default_rng(0)
n_perm = 500
perm_ics = np.empty(n_perm)
for p in range(n_perm):
    perm_resid = rng.permutation(resid)
    perm_ics[p] = np.corrcoef(lag2, perm_resid)[0, 1]
pval = float((np.abs(perm_ics) >= abs(ic_full)).mean())
print(f"permutation p-value = {pval:.3f}  (perm_std={perm_ics.std():.4f})")

print("\n=== also check: does |lag2| (magnitude only) explain residual VARIANCE (not sign)? ===")
abs_resid = np.abs(resid)
abs_lag2 = np.abs(lag2)
ic_mag = float(np.corrcoef(abs_lag2, abs_resid)[0, 1])
print(f"corr(|lag2|, |resid|) = {ic_mag:+.4f}")
