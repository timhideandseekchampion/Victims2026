"""Price/asset CLUSTERING: cluster the 50 names into groups (causal, on return co-movement),
then trade each name RELATIVE to its cluster (cluster-demeaned reversion & cluster-residual ridge)
instead of vs the whole market. If sub-group structure exists beyond the one market factor,
cluster-relative signals carry higher IC. Test IC on 250-500 AND 500-750 vs global signals."""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
A = RET[1:]                                                  # 50 tradeable names


def clusters(d, k, lb=250):
    """KMeans clusters of the 50 names on their return-correlation over trailing lb days (causal)."""
    W = A[:, max(0, d-lb):d]
    Wn = (W - W.mean(1, keepdims=True)) / (W.std(1, keepdims=True) + 1e-12)
    lab = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(Wn)
    return lab


def ic_of(sigfn, S, E, k, step=2):
    ics = []; lab = None; lastfit = -999
    for d in range(S, E-1, step):
        if d - lastfit >= 25:                               # recluster every 25 days
            lab = clusters(d, k); lastfit = d
        sig = sigfn(d, lab)
        fwd = A[:, d]                                        # next-day return (position set at d earns A[:,d])
        if sig is None or sig.std() < 1e-12: continue
        ics.append(np.corrcoef(sig, fwd)[0, 1])
    ics = np.array(ics)
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))+1e-12)


def cluster_demean(vec, lab):
    out = vec.copy()
    for c in np.unique(lab):
        m = lab == c; out[m] = vec[m] - vec[m].mean()
    return out


# signals (all use returns through A[:, d-1], predict A[:, d])
def glob_rev(h):    return lambda d, lab: -(A[:, d-h:d].sum(1) - A[:, d-h:d].sum(1).mean())
def clus_rev(h):    return lambda d, lab: -cluster_demean(A[:, d-h:d].sum(1), lab)
def clus_rev1():    return lambda d, lab: -cluster_demean(A[:, d-1], lab)

print("Cross-sectional IC (t): GLOBAL-relative vs CLUSTER-relative reversion\n")
print(f"{'signal':30} {'250-500 IC(t)':>18} {'500-750 IC(t)':>18}")
tests = [
    ("global rev-5d", glob_rev(5), None),
    ("global rev-20d", glob_rev(20), None),
    ("cluster rev-5d (k=5)", clus_rev(5), 5),
    ("cluster rev-20d (k=5)", clus_rev(20), 5),
    ("cluster rev-5d (k=7)", clus_rev(5), 7),
    ("cluster rev-1d (k=5)", clus_rev1(), 5),
    ("cluster rev-20d (k=3)", clus_rev(20), 3),
]
for name, fn, k in tests:
    kk = k if k else 5
    io, to = ic_of(fn, 250, 500, kk); inn, tn = ic_of(fn, nt-250, nt, kk)
    print(f"{name:30} {io:+.4f} (t{to:+.1f})   {inn:+.4f} (t{tn:+.1f})")
print("\nridge lead-lag reference IC ~ 0.068 (250-500) / 0.074 (500-750).")
print("Cluster-relative IC >> global => sub-group structure is a real, unused edge.")
