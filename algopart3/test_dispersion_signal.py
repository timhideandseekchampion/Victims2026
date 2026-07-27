"""New angle (not pairwise/lead-lag): cross-sectional DISPERSION -- how much the 49 idio stocks'
returns disagree with each other on a given day -- as a market-wide, aggregate predictive feature.
Genuinely different mechanism class from everything else tested this session: not a per-stock or
per-pair relationship, an aggregate "regime uncertainty" signal, in the same family as the ALREADY-
VALIDATED ALGO vol-in-mean effect (high realized vol -> higher next ALGO return) but built from
cross-sectional disagreement instead of ALGO's own time-series volatility.

Hypotheses tested, in order:
  1. dispersion(t) -> ALGO's own next-day return r0(t+1)          [connects to the known vol-in-mean edge]
  2. dispersion(t) -> ALGO's next-day REALIZED VOLATILITY          [risk/magnitude signal, not direction]
  3. dispersion(t) -> mean idio next-day return (book-wide tilt)   [orthogonal to existing per-stock ridge]
  4. dispersion(t) -> dispersion(t+1) (own-persistence)            [diagnostic only]
Each gets a full-sample pooled IC, a permutation test (circular shift, preserves autocorrelation),
and an H1/H2 persistence check -- the same bar every other finding this session was held to.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
r0 = r[0]
ridio = r[1:]  # (49, T)
T = r.shape[1]

print("=== 1. build causal cross-sectional dispersion + trailing z-score ===")
disp = ridio.std(axis=0)  # (T,) -- cross-sectional std across 49 stocks, each day (contemporaneous, valid EOD)
DISP_Z = 60
dispz = np.full(T, np.nan)
for s in range(DISP_Z, T):
    w = disp[s - DISP_Z:s]
    dispz[s] = (disp[s] - w.mean()) / (w.std() + 1e-12)
print(f"dispersion series built. mean={disp.mean():.4f} std={disp.std():.4f}")


def pooled_ic_perm(x, y, label, n_perm=500):
    ok = ~np.isnan(x) & ~np.isnan(y)
    xs, ys = x[ok], y[ok]
    ic = float(np.corrcoef(xs, ys)[0, 1])
    n = len(xs)
    half = n // 2
    ic1 = float(np.corrcoef(xs[:half], ys[:half])[0, 1])
    ic2 = float(np.corrcoef(xs[half:], ys[half:])[0, 1])
    rng = np.random.default_rng(0)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        shift = rng.integers(1, n - 1)
        ys_shift = np.roll(ys, shift)
        perm_ics[p] = np.corrcoef(xs, ys_shift)[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f} (n={n})  H1={ic1:+.4f}  H2={ic2:+.4f}  "
          f"perm p={pval:.3f}  perm_std={perm_ics.std():.4f}")
    return ic, pval


print("\n=== 2. hypothesis tests ===")
ret1 = np.full(T, np.nan); ret1[:-1] = r0[1:]  # ALGO's next-day return, aligned to day t
mean_idio_next = np.full(T, np.nan); mean_idio_next[:-1] = ridio[:, 1:].mean(axis=0)  # book-wide next-day tilt

vol20 = np.full(T, np.nan)
for s in range(20, T):
    vol20[s] = np.abs(r0[s - 20:s]).std()
algo_vol_next = np.full(T, np.nan); algo_vol_next[:-1] = vol20[1:]  # crude next-day-window vol proxy

print("H1: dispersion(t) -> ALGO next-day return")
pooled_ic_perm(dispz, ret1, "  dispz -> ALGO r(t+1)")

print("H2: dispersion(t) -> ALGO next-window realized vol (magnitude, not direction)")
pooled_ic_perm(dispz, algo_vol_next, "  dispz -> ALGO vol(t+1:t+20)")

print("H3: dispersion(t) -> mean idio next-day return (book-wide tilt)")
pooled_ic_perm(dispz, mean_idio_next, "  dispz -> mean_idio r(t+1)")

print("H4 (diagnostic): dispersion(t) -> dispersion(t+1) (own persistence / clustering)")
disp_next = np.full(T, np.nan); disp_next[:-1] = dispz[1:]
pooled_ic_perm(dispz, disp_next, "  dispz -> dispz(t+1)")
