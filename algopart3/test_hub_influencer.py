"""New angle: hub/influencer stocks in the lead-lag network. SAFE_llboost.py's significance gate
gives each follower stock J a single best leader I; some stocks get picked as leader by MANY
followers (hubs), others by none. Never tested directly: is a stock's "degree" (fan-in count: how
many OTHER stocks currently have it as their significant leader) itself informative -- either (a)
about the hub's OWN next-day return, or (b) about how reliable the RIDGE's existing forecast
currently is for that hub stock. Both tested causally (fresh-every-day significance gate, matching
production exactly), with the same permutation + H1/H2 rigor as every other signal hypothesis.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
rs = r[1:]
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


print("=== precompute: day-by-day significance-gate + fan-in degree per stock ===")
t0 = time.time()
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
print(f"  WZ done ({time.time()-t0:.0f}s)")

rows = []  # (k, i, degree, target, wz_correct)
t0 = time.time()
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
        best_leader[j] = i; best_corr[j] = col[i]
    significant = np.abs(best_corr) > thr
    degree = np.zeros(n, dtype=int)
    for j in range(n):
        if significant[j]:
            degree[best_leader[j]] += 1
    wz = WZ[k]
    for i in range(n):
        target = rs[i, k]
        wz_sign_correct = int(np.sign(wz[i]) == np.sign(target)) if target != 0 else np.nan
        rows.append((k, i, degree[i], target, wz_sign_correct))
print(f"  done ({time.time()-t0:.0f}s); {len(rows)} (stock,day) instances")

arr_k = np.array([x[0] for x in rows])
arr_deg = np.array([x[2] for x in rows], dtype=float)
arr_target = np.array([x[3] for x in rows])
arr_correct = np.array([x[4] for x in rows], dtype=float)

print(f"\ndegree distribution: min={arr_deg.min():.0f} median={np.median(arr_deg):.0f} "
      f"mean={arr_deg.mean():.2f} max={arr_deg.max():.0f}")
print(f"% of (stock,day) with degree==0 (never anyone's leader today): {(arr_deg==0).mean()*100:.1f}%")


def pooled_ic_perm(X, Y, K, label, n_perm=300):
    ok = ~np.isnan(X) & ~np.isnan(Y)
    X, Y = X[ok], Y[ok]; K = K[ok]
    if len(X) < 30:
        print(f"{label}: too few samples, skipping"); return
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


print("\n=== H1: hub-degree(today) -> hub's OWN return(today) [degree computed causally, same day] ===")
pooled_ic_perm(arr_deg, arr_target, arr_k, "  degree -> own return")

print("\n=== H2: does hub-degree predict RIDGE FORECAST ACCURACY (sign hit rate)? ===")
q = np.percentile(arr_deg, [25, 50, 75, 90])
print(f"degree quartile/decile boundaries: {q}")
for lo, hi, name in [(0, 0, "degree=0 (never a leader)"), (1, q[1], "low (1 to median)"),
                      (q[1], q[2], "mid (median to Q3)"), (q[2], q[3], "high (Q3 to P90)"),
                      (q[3], arr_deg.max(), "top decile")]:
    mask = (arr_deg >= lo) & (arr_deg <= hi) & ~np.isnan(arr_correct)
    if mask.sum() < 30: continue
    hit_rate = arr_correct[mask].mean()
    print(f"  {name:<28} n={int(mask.sum()):>6}  ridge-sign hit rate={hit_rate*100:.1f}%")
