import numpy as np, pandas as pd
import SAFE_llboost as SHIPPED

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape

for day in (600, 700, 800, 900):
    pos_shipped = np.asarray(SHIPPED.getMyPosition(P[:, :day+1]))
    print(f"day={day} shipped idio sign pattern (first 10): {np.sign(pos_shipped[1:11]).astype(int)}")

print("\n--- now compare against my from-scratch reconstruction (WZ + BOOST_AT) ---")
import SAFE
logp = np.log(P)
r = np.diff(logp, axis=1)
rs = r[1:]
n, T = rs.shape
from scipy import stats

def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = 0.05 / 49
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))

def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]

for day in (600, 700, 800, 900):
    k = day
    rr = r[:, :k]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    rv_ = logp[1:, k] - logp[1:, k - SAFE.REV_W]
    rv_ = rv_ - rv_.mean()
    rv = -rv_ / (rv_.std() + 1e-12)
    wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv

    T_ = k
    Xi = rs[:, :T_-1]; Yj = rs[:, 1:T_]
    thr = sig_threshold(Xi.shape[1])
    C = corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr: continue
        lead = rs[i, :T_]
        scale = np.nanstd(lead[max(0, T_-1-1000):T_-1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead)/scale)**2.0
        a = max(0, T_-1-190)
        xs = lead_boost[a:T_-1]; ys = rs[j, a+1:T_]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0,1])
        if ic <= 0: continue
        wz[j] += 1.5 * lead_boost[-1]
    print(f"day={day} reconstructed idio sign pattern (first 10): {np.sign(wz[:10]).astype(int)}")
