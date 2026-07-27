"""Genuinely untried angle: cluster the 49 STOCKS themselves (not days) by their IDIOSYNCRATIC
(post-ALGO-beta) return correlation profile, to look for sub-group/sector structure beyond the
single global ALGO factor and the individual pairwise lead-lag network already found. PCA (tested
earlier: PC2/PC3 not significant) only catches GLOBAL linear factors; a small group of 3-6 stocks
that share dynamics wouldn't necessarily show up as a significant PC even if it's real and tradeable.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
r0 = r[0]
T = r.shape[1]

print("computing idiosyncratic (post-ALGO-beta) residual returns for all 49 stocks ...")
beta = np.array([np.polyfit(r0, r[k], 1)[0] for k in range(1, nInst)])
resid = np.array([r[k] - beta[k - 1] * r0 for k in range(1, nInst)])  # (49, T)

print("clustering stocks by their residual correlation PROFILE (each stock's corr vector to all others) ...")
C = np.corrcoef(resid)  # (49,49) residual correlation matrix
np.fill_diagonal(C, 0.0)
print(f"off-diagonal residual corr: mean={C[np.triu_indices(49,1)].mean():+.4f}  "
      f"std={C[np.triu_indices(49,1)].std():.4f}  max={C.max():.3f}  min={C.min():.3f}")

for method, k in [("kmeans", 4), ("kmeans", 6), ("kmeans", 8), ("hierarchical", 6)]:
    if method == "kmeans":
        labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(C)
    else:
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(1 - C)
    print(f"\n=== {method}, k={k} ===")
    for c in range(k):
        members = [names[1:][i] for i in range(49) if labels[c if False else i] == c]
        if len(members) < 2: continue
        idx = [i for i in range(49) if labels[i] == c]
        within_corr = C[np.ix_(idx, idx)]
        avg_within = within_corr[np.triu_indices(len(idx), 1)].mean() if len(idx) > 1 else float('nan')
        print(f"  cluster {c} (n={len(members)}): {members[:8]}{'...' if len(members)>8 else ''}  "
              f"avg_within_corr={avg_within:+.4f}")

print("\n\n=== permutation test: is the clustering structure real, or does clustering find this much 'structure' in pure noise too? ===")
rng = np.random.default_rng(0)


def best_cluster_corr(resid_mat, k=6):
    Cx = np.corrcoef(resid_mat)
    np.fill_diagonal(Cx, 0.0)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Cx)
    best = -1.0
    for c in range(k):
        idx = [i for i in range(resid_mat.shape[0]) if labels[i] == c]
        if len(idx) < 2: continue
        wc = Cx[np.ix_(idx, idx)]
        avg = wc[np.triu_indices(len(idx), 1)].mean()
        best = max(best, avg)
    return best


obs_best = best_cluster_corr(resid, k=6)
print(f"observed best-cluster avg within-corr (k=6): {obs_best:.4f}")

null_best = []
for _ in range(100):
    resid_perm = np.array([rng.permutation(resid[i]) for i in range(resid.shape[0])])
    null_best.append(best_cluster_corr(resid_perm, k=6))
null_best = np.array(null_best)
p = float(np.mean(null_best >= obs_best))
print(f"permutation null (100 draws, EACH stock's time series independently shuffled): "
      f"mean={null_best.mean():.4f}  p95={np.percentile(null_best,95):.4f}  max={null_best.max():.4f}")
print(f"P(null best-cluster-corr >= observed) = {100*p:.0f}%")
