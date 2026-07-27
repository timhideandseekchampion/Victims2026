"""Next untried angle: per-stock return SKEWNESS / tail asymmetry as a predictor of that stock's
OWN next-day return. Distinct from mean (momentum/reversion), vol (already used), and pairwise
(already used/exhausted) signals -- this asks whether the SHAPE of a stock's recent return
distribution (lopsided toward big up-moves vs big down-moves) carries information, in either
direction: continuation (skew persists) or reversion (a skewed run of big moves snaps back).

Causal rolling skewness per stock (trailing SKEW_W days), tested both as a raw signal and pooled
cross-sectionally (all 49 stocks x all days), same rigor bar as every other finding this session:
full-sample pooled IC, permutation test (circular shift per stock, preserves each stock's own
autocorrelation), H1/H2 persistence split.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
ridio = r[1:]  # (49, T)
n, T = ridio.shape

SKEW_W = 60


def rolling_skew(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t - w:t]
        m = seg.mean(); s = seg.std()
        if s < 1e-12:
            continue
        out[t] = float(np.mean(((seg - m) / s) ** 3))
    return out


print("=== building causal rolling skewness per stock (60-day trailing window) ===")
skew = np.full((n, T), np.nan)
for j in range(n):
    skew[j] = rolling_skew(ridio[j], SKEW_W)
print(f"done. pooled mean skew={np.nanmean(skew):.3f}  std={np.nanstd(skew):.3f}")


def pooled_ic_perm(feat, target, label, n_perm=300):
    """feat, target: (n, T) arrays, tests feat(t) -> target(t+1), pooled across stocks and days."""
    rng = np.random.default_rng(0)
    rows_x, rows_y = [], []
    for t in range(T - 1):
        fx = feat[:, t]; fy = target[:, t + 1]
        ok = ~np.isnan(fx) & ~np.isnan(fy)
        if ok.sum() == 0: continue
        rows_x.append(fx[ok]); rows_y.append(fy[ok])
    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    ic = float(np.corrcoef(X, Y)[0, 1])
    half_t = T // 2
    def sub_ic(t0, t1):
        rx, ry = [], []
        for t in range(t0, min(t1, T - 1)):
            fx = feat[:, t]; fy = target[:, t + 1]
            ok = ~np.isnan(fx) & ~np.isnan(fy)
            if ok.sum() == 0: continue
            rx.append(fx[ok]); ry.append(fy[ok])
        xs = np.concatenate(rx); ys = np.concatenate(ry)
        return float(np.corrcoef(xs, ys)[0, 1])
    ic1 = sub_ic(0, half_t); ic2 = sub_ic(half_t, T)
    # permutation: circular-shift each stock's feature series independently (preserves cross-sectional
    # structure and each stock's own autocorrelation, breaks the feat(t)->target(t+1) timing link)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        feat_shift = np.empty_like(feat)
        for j in range(n):
            shift = rng.integers(1, T - 1)
            feat_shift[j] = np.roll(feat[j], shift)
        rx, ry = [], []
        for t in range(T - 1):
            fx = feat_shift[:, t]; fy = target[:, t + 1]
            ok = ~np.isnan(fx) & ~np.isnan(fy)
            if ok.sum() == 0: continue
            rx.append(fx[ok]); ry.append(fy[ok])
        xs = np.concatenate(rx); ys = np.concatenate(ry)
        perm_ics[p] = np.corrcoef(xs, ys)[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}  perm_std={perm_ics.std():.4f}")
    return ic, pval


print("\n=== H1: raw skewness(t) -> own next-day return (pooled across all 49 stocks) ===")
pooled_ic_perm(skew, ridio, "  skew -> r(t+1)")

print("\n=== H2: |skewness| (tail-asymmetry MAGNITUDE, either direction) -> next-day |return| (vol proxy) ===")
abs_skew = np.abs(skew)
abs_ret = np.abs(ridio)
pooled_ic_perm(abs_skew, abs_ret, "  |skew| -> |r(t+1)|")

print("\n=== H3: skewness z-scored (relative to the stock's OWN trailing skew history) -> next-day return ===")
SKEW_Z_W = 120
skewz = np.full((n, T), np.nan)
for j in range(n):
    for t in range(SKEW_W + SKEW_Z_W, T):
        w = skew[j, t - SKEW_Z_W:t]
        ok = ~np.isnan(w)
        if ok.sum() > 30:
            skewz[j, t] = (skew[j, t] - w[ok].mean()) / (w[ok].std() + 1e-12)
pooled_ic_perm(skewz, ridio, "  skewz -> r(t+1)")
