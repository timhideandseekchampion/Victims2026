"""New idea: earlier tonight, a bug in test_partial_pooling_boost.py used TODAY's leader return
(contemporaneous, b[t]*r[j,t]) instead of yesterday's (b[t-1]*r[j,t]) to size a trade -- an invalid
lookahead since you can't observe today's close before placing today's trade. But the REALIZED
same-day divergence, once today's close is in, IS valid lagged information for TOMORROW's decision.

This tests a specific, different hypothesis from every prior pairwise attempt tonight: not "does
yesterday's leader return predict today's follower return" (the ridge already captures this, and
~10 gate/blend/stack variants on top of it all failed) but "does the SIZE of today's SURPRISE --
how far a stock's realized move deviated from what its usual contemporaneous co-mover's move would
have predicted -- forecast a catch-up/reversion move tomorrow." This is a spread mean-reversion
question (pairs-trading style), not a raw leader-return-continuation question.

Method (fully causal): idiosyncratic residuals (remove ALGO beta) -> expanding-window co-mover map
(highest |corr| among OTHER stocks, re-estimated at checkpoints) -> expanding-window regression
coefficient beta_ji -> daily surprise = resid_j[t] - beta_ji*resid_i[t] -> z-score by trailing
window -> test IC vs resid_j[t+1], with permutation + H1/H2 persistence exactly as every other
finding tonight was validated.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
Praw = P.values.T.astype(float)
nInst, nt = Praw.shape
logp = np.log(Praw)
r = np.diff(logp, axis=1)
T = r.shape[1]
r0 = r[0]

print("=== 1. idiosyncratic residuals (remove ALGO beta, causal checkpoint-refit beta, matching the ===")
print("    checkpoint-refit convention used all night for leader maps / GBM retrains) ===")
CP = list(range(100, T, 50))


def beta_at(cp):
    v0 = r0[:cp]
    return np.array([np.polyfit(v0, r[j, :cp], 1)[0] if j > 0 else 1.0 for j in range(nInst)])


BETA_CP = {cp: beta_at(cp) for cp in CP}
print(f"checkpoints: {CP}")


def beta_for_day(t):
    valid = [c for c in CP if c <= t]
    return BETA_CP[valid[-1]] if valid else BETA_CP[CP[0]]


resid = np.full((nInst, T), np.nan)
for t in range(CP[0], T):
    b = beta_for_day(t)
    resid[:, t] = r[:, t] - b * r0[t]
resid[0, :] = np.nan  # ALGO has no "idiosyncratic residual" of its own here
print("done")

print("\n=== 2. causal contemporaneous co-mover map (highest |corr| of idio residuals among OTHER stocks) ===")


def comover_at(cp):
    X = resid[1:, :cp]
    ok = ~np.any(np.isnan(X), axis=0)
    Xc = X[:, ok]
    Xn = (Xc - Xc.mean(1, keepdims=True)) / (Xc.std(1, keepdims=True) + 1e-12)
    C = (Xn @ Xn.T) / Xn.shape[1]
    n = nInst - 1
    comov = {}; comov_beta = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        comov[j + 1] = i + 1
        # regression coefficient (not just correlation) for magnitude-consistent surprise
        xi = Xc[i]; xj = Xc[j]
        comov_beta[j + 1] = float(np.polyfit(xi, xj, 1)[0])
    return comov, comov_beta


COMOVE_CP = {cp: comover_at(cp) for cp in CP if cp >= 150}
CP2 = sorted(COMOVE_CP.keys())


def comove_for_day(t):
    valid = [c for c in CP2 if c <= t]
    return COMOVE_CP[valid[-1]] if valid else COMOVE_CP[CP2[0]]


print(f"co-mover checkpoints: {CP2}")
first_cp = CP2[0]
sample_comov, _ = COMOVE_CP[first_cp]
print(f"sample co-mover map at cp={first_cp}: {dict(list(sample_comov.items())[:5])} ...")

print("\n=== 3. daily surprise/divergence signal (causal), z-scored by trailing window ===")
START = first_cp + 10
surprise = np.full((nInst, T), np.nan)
for t in range(START, T):
    comov, comov_beta = comove_for_day(t)
    for j in range(1, nInst):
        i = comov[j]
        surprise[j, t] = resid[j, t] - comov_beta[j] * resid[i, t]

VOL_Z_W = 60
div_z = np.full((nInst, T), np.nan)
for j in range(1, nInst):
    s = surprise[j]
    for t in range(START + VOL_Z_W, T):
        w = s[t - VOL_Z_W:t]
        ok = ~np.isnan(w)
        if ok.sum() > 20:
            div_z[j, t] = (s[t] - w[ok].mean()) / (w[ok].std() + 1e-12)
print("done")

print("\n=== 4. pooled IC: does today's divergence z-score predict TOMORROW's idio residual return? ===")


def pooled_ic(feat, target, tmin, tmax):
    rows_x = []; rows_y = []
    for t in range(tmin, tmax):
        fx = feat[1:, t]; fy = target[1:, t + 1]
        ok = ~np.isnan(fx) & ~np.isnan(fy)
        if ok.sum() == 0: continue
        rows_x.append(fx[ok]); rows_y.append(fy[ok])
    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    return float(np.corrcoef(X, Y)[0, 1]), len(X)


tmin, tmax = START + VOL_Z_W, T - 1
ic_full, n_full = pooled_ic(div_z, resid, tmin, tmax)
print(f"full-sample pooled IC(div_z[t] -> resid[t+1]): {ic_full:+.4f}  (n={n_full})")

half = (tmin + tmax) // 2
ic_h1, n1 = pooled_ic(div_z, resid, tmin, half)
ic_h2, n2 = pooled_ic(div_z, resid, half, tmax)
print(f"H1 IC: {ic_h1:+.4f} (n={n1})   H2 IC: {ic_h2:+.4f} (n={n2})")

print("\n=== 5. permutation test (shuffle div_z's TIME AXIS per stock, preserve cross-sectional shape) ===")
rng = np.random.default_rng(0)
perm_ics = []
for p in range(200):
    div_shuf = div_z.copy()
    for j in range(1, nInst):
        col = div_shuf[j, tmin:tmax + 1].copy()
        rng.shuffle(col)
        div_shuf[j, tmin:tmax + 1] = col
    ic_p, _ = pooled_ic(div_shuf, resid, tmin, tmax)
    perm_ics.append(ic_p)
perm_ics = np.array(perm_ics)
pval = (np.abs(perm_ics) >= np.abs(ic_full)).mean()
print(f"permutation null: mean={perm_ics.mean():+.4f}  std={perm_ics.std():.4f}  "
      f"p-value (|null|>=|observed|) = {pval:.3f}")
