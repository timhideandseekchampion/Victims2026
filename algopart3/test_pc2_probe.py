"""Explore beyond the confirmed structure: is there a SECOND common factor (PC2/PC3) with its own
predictive power, the way ALGO (PC1) does? And do the current ridge model's residuals still have
exploitable temporal or cross-sectional structure it isn't using? Both checked with the same
permutation-test rigor used for the ALGO vol signal and the pairwise lead-lag network tonight.
"""
import numpy as np, pandas as pd
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)          # (51, T)

print("=== 1. PCA structure ===")
Rz = (r - r.mean(1, keepdims=True)) / r.std(1, keepdims=True)
cov = np.cov(Rz)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(-evals)
evals = evals[order]; evecs = evecs[:, order]
print("variance explained, top 5 PCs:", (evals[:5] / evals.sum()).round(3))
pc1 = evecs[:, 0]; pc2 = evecs[:, 1]; pc3 = evecs[:, 2]
print("PC1 loadings: mean", pc1.mean().round(3), "std", pc1.std().round(3), "(should be ~uniform positive -> ALGO-like)")
print("PC2 top +loadings:", [names[i] for i in np.argsort(-pc2)[:5]])
print("PC2 top -loadings:", [names[i] for i in np.argsort(pc2)[:5]])
print("PC3 top +loadings:", [names[i] for i in np.argsort(-pc3)[:5]])
print("PC3 top -loadings:", [names[i] for i in np.argsort(pc3)[:5]])

# PC scores over time (today's value of each component)
pc1_t = pc1 @ Rz; pc2_t = pc2 @ Rz; pc3_t = pc3 @ Rz

def ic_and_perm(feat_today, targets, s=0, e=None, N=300, seed=0):
    """feat_today[t] predicts targets[:,t+1] (cross-sectional average |corr| and permutation p)."""
    e = targets.shape[1] - 1 if e is None else e
    x = feat_today[s:e]; Y = targets[:, s+1:e+1]
    ok = ~np.isnan(x)
    x = x[ok]; Y = Y[:, ok]
    ics = np.array([np.corrcoef(x, Y[j])[0, 1] for j in range(Y.shape[0])])
    obs = np.abs(ics).mean()
    rng = np.random.default_rng(seed)
    null = np.empty(N)
    for i in range(N):
        xp = rng.permutation(x)
        null[i] = np.mean([abs(np.corrcoef(xp, Y[j])[0, 1]) for j in range(Y.shape[0])])
    p = float(np.mean(null >= obs))
    return obs, null.mean(), p

print("\n=== 2. does PC2 / PC3 (today) predict ANY stock's next-day return, like ALGO/PC1 does? ===")
for lbl, feat in [("PC1 (~ALGO)", pc1_t), ("PC2", pc2_t), ("PC3", pc3_t)]:
    obs, nullmean, p = ic_and_perm(feat[:-1], r[1:])   # predict stocks 1..50's next-day return
    print(f"  {lbl:<12} mean|IC| across 50 stocks = {obs:.4f}  (perm null mean {nullmean:.4f})  p={100*p:.0f}%")

print("\n=== 3. residual structure: fit SAFE's own ridge (hl=1000), check residual autocorr + xs-corr ===")
X = r[:, :-1].T; Y = r[1:, 1:].T
B, mx, my = SAFE._ewls_ridge(X, Y, hl=1000, a=SAFE.RIDGE_A)
pred = my + (X - mx) @ B
resid = Y - pred          # (T-1, 50): residual return per stock per day
resid_ac = np.array([np.corrcoef(resid[:-1, j], resid[1:, j])[0, 1] for j in range(50)])
print(f"  residual own lag-1 autocorr: mean {resid_ac.mean():+.4f}  frac>0 {(resid_ac>0).mean():.2f}")
avg_resid_corr = np.corrcoef(resid.T)
off = avg_resid_corr[np.triu_indices(50, 1)]
print(f"  avg pairwise residual cross-correlation (same-day): mean {off.mean():+.4f}  (near 0 -> ALGO factor fully removed)")
# does residual today predict residual tomorrow cross-sectionally at the PAIR level (leftover lead-lag)?
resid_ll = np.array([[np.corrcoef(resid[:-1, i], resid[1:, j])[0, 1] if i != j else np.nan
                       for j in range(50)] for i in range(50)])
print(f"  max |leftover pairwise lead-lag in residuals| = {np.nanmax(np.abs(resid_ll)):.4f}  "
      f"(compare: raw pre-model max was 0.171)")
