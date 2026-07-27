"""Unconventional angle: use K-means to discover market REGIME STATES directly from observable
features (vol level, vol-of-vol, momentum, cross-sectional dispersion) -- not by testing each
candidate signal's own IC (where small-sample statistical power was the wall in every prior gate
attempt). Then check whether vol/momentum/reversion have MEANINGFULLY DIFFERENT IC across the
discovered clusters -- if reversion is real in ONE cluster (even though it washes out pooled across
the whole file), that's a genuinely different kind of finding than anything tried so far. A Markov
chain on the cluster-label sequence then tells us how persistent/predictable regime membership is,
which determines whether this is even usable for trading (a regime that changes every day is useless
even if real).
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r = np.diff(logp, axis=1)
r0 = np.diff(lpA)
T = r.shape[1]
ret1 = np.full(len(lpA), np.nan); ret1[:-1] = lpA[1:] - lpA[:-1]

print("building regime-state features (vol, vol-of-vol, momentum, cross-sectional dispersion) ...")
vol20 = np.full(T, np.nan); vol20[19:] = M._roll_std(r0, 20)
volz = np.full(T, np.nan)
for s in range(80, T):
    w = vol20[s - 60:s]; volz[s] = (vol20[s] - w.mean()) / (w.std() + 1e-12)
vol_of_vol = np.full(T, np.nan)
for s in range(80, T):
    w = vol20[s - 60:s]; ok = ~np.isnan(w)
    if ok.sum() > 20: vol_of_vol[s] = w[ok].std() / (w[ok].mean() + 1e-12)
mom_raw = np.full(T, np.nan)
for t in range(10, T):
    mom_raw[t] = lpA[t] - lpA[t - 10]
momz = np.full(T, np.nan)
for s in range(80, T):
    w = mom_raw[s - 60:s]; ok = ~np.isnan(w)
    if ok.sum() > 20: momz[s] = (mom_raw[s] - w[ok].mean()) / (w[ok].std() + 1e-12)
dispersion = np.array([r[1:, t].std() for t in range(T)])
dispz = np.full(T, np.nan)
for s in range(80, T):
    w = dispersion[s - 60:s]; dispz[s] = (dispersion[s] - w.mean()) / (w.std() + 1e-12)

feat = np.column_stack([volz, vol_of_vol, momz, dispz])
valid = ~np.any(np.isnan(feat), axis=1)
print(f"valid days: {valid.sum()}/{T}")

print("\nrunning K-means (full-sample fit, diagnostic pass -- causal version comes after if promising) ...")
for k in (2, 3, 4):
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = np.full(T, -1)
    labels[valid] = km.fit_predict(feat[valid])

    print(f"\n=== k={k} ===")
    for c in range(k):
        mask = (labels == c) & (~np.isnan(ret1[:T]))
        n = mask.sum()
        if n < 30: continue
        # vol IC within this cluster
        okv = mask & ~np.isnan(volz)
        icv = np.corrcoef(volz[okv], ret1[:T][okv])[0, 1] if okv.sum() > 30 else float('nan')
        # momentum IC (z-scored 10d return, NOT faded -- same convention as SAFE_llvol's momentum leg)
        okm = mask & ~np.isnan(momz)
        icm = np.corrcoef(momz[okm], ret1[:T][okm])[0, 1] if okm.sum() > 30 else float('nan')
        print(f"  cluster {c}: n={n:>4}  mean_volz={feat[mask,0].mean():+.2f}  mean_vov={feat[mask,1].mean():.2f}  "
              f"mean_momz={feat[mask,2].mean():+.2f}  mean_disp={feat[mask,3].mean():+.2f}  "
              f"IC(vol)={icv:+.4f}  IC(mom)={icm:+.4f}")

print("\n\n=== persistence + Markov check on the k=4 clustering ===")
km4 = KMeans(n_clusters=4, n_init=10, random_state=0)
labels4 = np.full(T, -1)
labels4[valid] = km4.fit_predict(feat[valid])

ret1_T = ret1[:T]
half = T // 2
for c in range(4):
    m1 = (labels4[:half] == c); m2 = (labels4[half:] == c)
    ok1 = m1 & ~np.isnan(volz[:half]) & ~np.isnan(ret1_T[:half])
    ok2 = m2 & ~np.isnan(volz[half:]) & ~np.isnan(ret1_T[half:])
    ic1 = np.corrcoef(volz[:half][ok1], ret1_T[:half][ok1])[0, 1] if ok1.sum() > 20 else float('nan')
    ic2 = np.corrcoef(volz[half:][ok2], ret1_T[half:][ok2])[0, 1] if ok2.sum() > 20 else float('nan')
    print(f"cluster {c}: H1 IC(vol)={ic1:+.4f} (n={ok1.sum()})   H2 IC(vol)={ic2:+.4f} (n={ok2.sum()})")

print("\nMarkov transition matrix (row=today's cluster, col=tomorrow's cluster), self-persistence on diagonal:")
trans = np.zeros((4, 4))
for t in range(T - 1):
    a, b = labels4[t], labels4[t + 1]
    if a >= 0 and b >= 0:
        trans[a, b] += 1
trans_p = trans / trans.sum(axis=1, keepdims=True)
for c in range(4):
    print(f"  from cluster {c}: " + "  ".join(f"->{c2}:{trans_p[c,c2]:.2f}" for c2 in range(4)) +
          f"   (self-persistence: {trans_p[c,c]:.2f})")

print("\nmean days spent in each cluster per visit (1/(1-self_persistence)):")
for c in range(4):
    p = trans_p[c, c]
    print(f"  cluster {c}: ~{1/(1-p):.1f} days per visit" if p < 1 else f"  cluster {c}: absorbing")
