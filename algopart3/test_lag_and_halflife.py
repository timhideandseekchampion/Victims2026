"""Two new, distinct hypotheses (not a re-run of the failed pairwise-boost line):
  1. Does the 1-day lag assumption hold universally, or do some leader->follower pairs genuinely
     catch up over a LONGER horizon (2,3,5,7,10 days)? Full search across lags x pairs, with a
     max-corrected permutation test (same rigor as every other pairwise test tonight).
  2. Per-stock residual OU half-life: does each stock have its OWN characteristic mean-reversion
     speed (vs one shared REV_W=10 for the whole book), and is there a genuinely fast, exploitable
     subset?
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
n = nInst - 1

print("=== 1. does any pair have a STRONGER relationship at lag > 1 than at lag 1? ===")
def best_at_lag(lag):
    Xi = r[1:, :-lag]; Yj = r[1:, lag:]
    Xc = Xi - Xi.mean(1, keepdims=True); Yc = Yj - Yj.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    C = (Xs @ Ys.T) / Xi.shape[1]
    np.fill_diagonal(C, np.nan)
    return C

results_by_lag = {}
for lag in (1, 2, 3, 5, 7, 10):
    C = best_at_lag(lag)
    results_by_lag[lag] = C
    obs_max = np.nanmax(np.abs(C))
    ai, aj = np.unravel_index(np.nanargmax(np.abs(C)), C.shape)
    print(f"  lag={lag:>2}: max|corr|={obs_max:.4f} at {names[ai+1]}->{names[aj+1]}")

# for each follower, which lag gives ITS best leader the strongest |corr|, and is it ever NOT lag=1?
print("\n  per-follower best lag (which of 1,2,3,5,7,10 maximizes |corr| with its own best leader at that lag):")
best_lag_per_follower = []
for j in range(n):
    best = (0, None, 1)  # (|corr|, leader, lag)
    for lag in (1, 2, 3, 5, 7, 10):
        col = results_by_lag[lag][:, j]
        i = int(np.nanargmax(np.abs(col)))
        c = col[i]
        if abs(c) > best[0]:
            best = (abs(c), names[i+1], lag)
    best_lag_per_follower.append(best[2])
from collections import Counter
print("  distribution of best-lag across all 49 followers:", dict(Counter(best_lag_per_follower)))

print("\n  permutation test (max |corr| across ALL lags x ALL pairs simultaneously, 200 draws) ...")
rng = np.random.default_rng(0)
obs_global_max = max(np.nanmax(np.abs(results_by_lag[lag])) for lag in results_by_lag)
null_max = []
Yj_by_lag = {lag: r[1:, lag:] for lag in (1,2,3,5,7,10)}
Xi_by_lag = {lag: r[1:, :-lag] for lag in (1,2,3,5,7,10)}
for _ in range(200):
    seed_perm = rng.permutation(r.shape[1])
    mx = 0.0
    for lag in (1,2,3,5,7,10):
        Xi = Xi_by_lag[lag]
        Yj_perm = r[1:][:, seed_perm][:, lag:]  # shuffle full timeline once, then re-slice per lag (consistent null)
        Xc = Xi - Xi.mean(1, keepdims=True); Yc = Yj_perm - Yj_perm.mean(1, keepdims=True)
        Xs = Xc/(Xc.std(1,keepdims=True)+1e-12); Ys = Yc/(Yc.std(1,keepdims=True)+1e-12)
        m = min(Xs.shape[1], Ys.shape[1])
        C = (Xs[:,:m] @ Ys[:,:m].T) / m
        np.fill_diagonal(C, np.nan)
        mx = max(mx, np.nanmax(np.abs(C)))
    null_max.append(mx)
null_max = np.array(null_max)
p = float(np.mean(null_max >= obs_global_max))
print(f"  observed global max (any lag, any pair) = {obs_global_max:.4f}   null mean {null_max.mean():.4f} "
      f"p95 {np.percentile(null_max,95):.4f}   P(null>=obs)={100*p:.0f}%")
