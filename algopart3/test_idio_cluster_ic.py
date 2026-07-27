"""Does the IDIO BOOK's ridge forecast quality (pooled across all 49 stocks) vary meaningfully
across the same market-regime clusters found for ALGO? Different question from the ALGO-only check:
here the target is each stock's own next-day return, pooled across all 49 names, within each
regime-day cluster.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
import SAFE, SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r = np.diff(logp, axis=1)
r0 = np.diff(lpA)
T = r.shape[1]

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

print("computing shipped SAFE idio wz series (all 49 names, full history) ...")
WZ = {}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    WZ[t] = wz
print("done")

print("\nrunning K-means (full-sample diagnostic pass) ...")
for k in (3, 4):
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = np.full(T, -1)
    labels[valid] = km.fit_predict(feat[valid])

    print(f"\n=== k={k}: pooled idio ridge IC (all 49 stocks) within each regime cluster ===")
    for c in range(k):
        xs = []; ys = []
        for t in range(SAFE.WARMUP, T - 1):
            if labels[t] != c: continue
            if (t + 1) not in WZ: continue
            xs.append(WZ[t + 1]); ys.append(r[1:, t + 1])
        if len(xs) < 5: continue
        X = np.concatenate(xs); Y = np.concatenate(ys)
        ok = ~np.isnan(X) & ~np.isnan(Y)
        ic = np.corrcoef(X[ok], Y[ok])[0, 1]
        n_days = len(xs)
        print(f"  cluster {c}: n_days={n_days:>4}  n_obs={ok.sum():>6}  mean_volz={feat[labels==c,0].mean():+.2f}  "
              f"mean_disp={feat[labels==c,3].mean():+.2f}  idio_pooled_IC={ic:+.4f}")
