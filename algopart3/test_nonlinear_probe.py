"""Non-linearity probe: for the pairwise lead-lag network, is the leader->follower relationship
STRONGER when the leader's move is large (a threshold/convex effect a linear model underfits), vs
roughly the same strength for small moves? Tested across the FULL 2450-pair grid (not just the top
pairs already found -- avoids repeating the "look only at candidates I already like" trap), with a
max-based permutation test and an H1/H2 persistence check on whatever comes out on top.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
logp = np.log(P)
r = np.diff(logp, axis=1)
n = 50
Xi = r[1:, :-1]   # (50, T-1) leader's return today
Yj = r[1:, 1:]    # (50, T-1) follower's return tomorrow
T1 = Xi.shape[1]


def corr_rows(x, Y):
    """corr(x, Y[j]) for every row j, vectorized."""
    xc = x - x.mean(); Yc = Y - Y.mean(1, keepdims=True)
    num = Yc @ xc
    den = np.sqrt((xc ** 2).sum()) * np.sqrt((Yc ** 2).sum(1)) + 1e-18
    return num / den


def hi_lo_diff_grid(X, Y):
    """(50,50) grid: |corr| in the top-tercile-|leader-move| days minus |corr| in the bottom tercile."""
    out = np.full((n, n), np.nan)
    for i in range(n):
        x = X[i]
        ax = np.abs(x)
        hi_thr = np.percentile(ax, 67); lo_thr = np.percentile(ax, 33)
        hi = ax >= hi_thr; lo = ax <= lo_thr
        c_hi = corr_rows(x[hi], Y[:, hi])
        c_lo = corr_rows(x[lo], Y[:, lo])
        out[i, :] = np.abs(c_hi) - np.abs(c_lo)
    np.fill_diagonal(out, np.nan)
    return out

DIFF = hi_lo_diff_grid(Xi, Yj)
flat = DIFF.flatten(); ok = ~np.isnan(flat)
obs_max = np.nanmax(np.abs(DIFF))
ai, aj = np.unravel_index(np.nanargmax(np.abs(DIFF)), DIFF.shape)
print(f"observed max |hi-tercile - lo-tercile| corr gap: {obs_max:.4f}  at {names[ai+1]} -> {names[aj+1]}")
print(f"distribution: mean {np.nanmean(np.abs(flat)):.4f}  p95 {np.nanpercentile(np.abs(flat),95):.4f}")

# does the KNOWN top pair (DUCT->AMRP) show this pattern too?
di = names.index("DUCT") - 1; aj_amrp = names.index("AMRP") - 1
print(f"DUCT->AMRP hi-lo diff: {DIFF[di, aj_amrp]:+.4f}")

print("\npermutation test (shuffle follower time-axis, 300 draws) ...")
rng = np.random.default_rng(0)
null_max = []
for _ in range(300):
    perm = rng.permutation(T1)
    Yp = Yj[:, perm]
    Dn = hi_lo_diff_grid(Xi, Yp)
    null_max.append(np.nanmax(np.abs(Dn)))
null_max = np.array(null_max)
p = float(np.mean(null_max >= obs_max))
print(f"null max: mean {null_max.mean():.4f}  p95 {np.percentile(null_max,95):.4f}  P(null_max >= obs) = {100*p:.0f}%")

# H1/H2 persistence check on the observed top pair
half = T1 // 2
def hi_lo_diff_one(x, y):
    ax = np.abs(x); hi_thr = np.percentile(ax,67); lo_thr = np.percentile(ax,33)
    hi = ax>=hi_thr; lo=ax<=lo_thr
    return abs(np.corrcoef(x[hi], y[hi])[0,1]) - abs(np.corrcoef(x[lo], y[lo])[0,1])
x_top = Xi[ai]; y_top = Yj[aj]
print(f"top pair {names[ai+1]}->{names[aj+1]} hi-lo diff: H1 {hi_lo_diff_one(x_top[:half], y_top[:half]):+.3f}  "
      f"H2 {hi_lo_diff_one(x_top[half:], y_top[half:]):+.3f}  full {hi_lo_diff_one(x_top, y_top):+.3f}")
