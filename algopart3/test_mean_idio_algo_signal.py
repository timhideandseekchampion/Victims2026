"""Step 3 (plan: revisit ALGO leg): does the LEVEL (not spread) of the idio book's average return
predict ALGO's own next-day move -- a market-breadth/sentiment signal. Distinct from the already-
rejected dispersion tests (test_dispersion_signal.py tested cross-sectional SPREAD predicting ALGO
and predicting mean-idio-next, but never tested mean-idio-LEVEL predicting ALGO specifically).
Same Stage 1 rigor as every other signal hypothesis this session: pooled IC, permutation test
(circular shift), H1/H2 persistence split.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
r0 = r[0]
ridio = r[1:]  # (49, T)
T = r.shape[1]

print("=== building causal mean-idio-return level + trailing z-score ===")
mean_idio = ridio.mean(axis=0)  # (T,) -- cross-sectional mean return across 49 stocks, each day
Z_W = 60
mean_idio_z = np.full(T, np.nan)
for s in range(Z_W, T):
    w = mean_idio[s - Z_W:s]
    mean_idio_z[s] = (mean_idio[s] - w.mean()) / (w.std() + 1e-12)
print(f"mean_idio level series built. mean={mean_idio.mean():.5f} std={mean_idio.std():.5f}")


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


print("\n=== hypothesis tests ===")
ret1 = np.full(T, np.nan); ret1[:-1] = r0[1:]  # ALGO's next-day return, aligned to day t

print("H1: mean_idio(t) [raw level] -> ALGO next-day return")
pooled_ic_perm(mean_idio, ret1, "  mean_idio raw -> ALGO r(t+1)")

print("\nH2: mean_idio_z(t) [trailing-z-scored level] -> ALGO next-day return")
pooled_ic_perm(mean_idio_z, ret1, "  mean_idio_z -> ALGO r(t+1)")

print("\nH3 (diagnostic): does mean_idio(t) correlate with ALGO's OWN same-day return r0(t)?")
print("   (checks whether this is just picking up shared same-day ALGO exposure, not new info)")
pooled_ic_perm(mean_idio, r0, "  mean_idio(t) -> ALGO r0(t) [contemporaneous]")
