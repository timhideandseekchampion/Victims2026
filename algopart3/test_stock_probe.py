"""Probe individual stocks for signals, rather than the aggregate cross-sectional ridge / index-level
signals already investigated. Three angles, each checked for persistence/significance (not just
in-sample magnitude) to avoid the exact "selecting significant names in-sample is data-snooping"
trap this repo already flagged for the vol signal:
  1. Per-stock OWN-return autocorrelation at several lags (1,2,3,5,10,20) -- is there a momentum or
     reversion signal in a stock's own history the current cross-sectional model doesn't use?
  2. Pairwise lead-lag scan: corr(stock_i return today, stock_j return tomorrow) for all i != j pairs
     (2450 candidates on 50 non-ALGO names) -- any standout pairs, and do they persist H1 vs H2?
  3. Per-name IC of the ACTUAL shipped idio forecast (wz_i vs next-day return_i) -- is the edge broad
     across all 49 names (justifying "trade all of them"), or concentrated in a handful?
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)          # (nInst, nt-1)

print("=== 1. per-stock own-return autocorrelation, several lags ===")
for lag in (1, 2, 3, 5, 10, 20):
    acs = []
    for i in range(1, nInst):
        x = r[i, :-lag]; y = r[i, lag:]
        acs.append(np.corrcoef(x, y)[0, 1])
    acs = np.array(acs)
    t_stat = acs.mean() / (acs.std(ddof=1) / np.sqrt(len(acs)))
    print(f"  lag={lag:>3}: mean {acs.mean():+.4f}  frac>0 {(acs>0).mean():.2f}  t(cross-sec) {t_stat:+.2f}")

print("\n=== 2. pairwise lead-lag scan (stock_i today -> stock_j tomorrow), all i!=j among 1..50 ===")
n = nInst - 1
Xi = r[1:, :-1]   # (n, T-1) today's returns
Yj = r[1:, 1:]    # (n, T-1) tomorrow's returns
T1 = Xi.shape[1]
half = T1 // 2
def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]

C_full = corrmat(Xi, Yj)
np.fill_diagonal(C_full, np.nan)
C_h1 = corrmat(Xi[:, :half], Yj[:, :half]); np.fill_diagonal(C_h1, np.nan)
C_h2 = corrmat(Xi[:, half:], Yj[:, half:]); np.fill_diagonal(C_h2, np.nan)

flat_full = C_full.flatten(); ok = ~np.isnan(flat_full)
print(f"  all {ok.sum()} off-diagonal pairs: mean {flat_full[ok].mean():+.4f}  std {flat_full[ok].std():.4f}")
idx_sorted = np.argsort(-np.abs(flat_full))
seen = 0
print("  top 15 |corr| pairs (full-sample) with H1/H2 persistence check:")
for k in idx_sorted:
    if not ok[k]: continue
    i, j = divmod(k, n)
    full_c = C_full[i, j]; h1 = C_h1[i, j]; h2 = C_h2[i, j]
    print(f"    {names[i+1]:>6} -> {names[j+1]:<6} (lag1): full {full_c:+.3f}  H1 {h1:+.3f}  H2 {h2:+.3f}")
    seen += 1
    if seen >= 15: break

# multiple-testing sanity check: how big would we expect the max |corr| to be under pure noise?
rng = np.random.default_rng(0)
null_max = []
for _ in range(200):
    Yp = Yj[:, rng.permutation(T1)]
    Cn = corrmat(Xi, Yp); np.fill_diagonal(Cn, np.nan)
    null_max.append(np.nanmax(np.abs(Cn)))
null_max = np.array(null_max)
obs_max = np.nanmax(np.abs(flat_full))
print(f"\n  observed max |corr| = {obs_max:.3f}   permutation null max|corr|: mean {null_max.mean():.3f} "
      f"p95 {np.percentile(null_max,95):.3f}   P(null_max >= obs) = {100*np.mean(null_max>=obs_max):.0f}%")
