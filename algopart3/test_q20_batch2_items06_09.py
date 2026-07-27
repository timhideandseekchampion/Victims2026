"""Batch 2, items 6/7/9 [SIGNAL]: new regime features on ALGO's OWN price series (single series,
not per-stock -- distinct hypotheses from the already-rejected per-stock versions of vol-of-vol
and skewness). Item 8 [SIGNAL] is per-idio-stock (distance from trailing high/low), using the
same pooled-IC + permutation + H1/H2 framework as item 5.
Stage 1 only: causal pooled IC, circular-shift permutation test, H1/H2 persistence split.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
ridio = r[1:]
n, T = ridio.shape

logp0 = logp[0]
r0 = r[0]  # ALGO log returns, length nt-1


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


def single_series_ic_perm(feat, target, label, n_perm=1000, seed=0):
    """feat, target: 1D arrays, same length, NaN where undefined. Causal by construction of feat."""
    rng = np.random.default_rng(seed)
    ok_all = ~np.isnan(feat) & ~np.isnan(target)
    Tn = len(feat)
    half = Tn // 2

    def ic_of(f, t0, t1):
        ok = ok_all[t0:t1]
        if ok.sum() < 30: return np.nan
        return float(np.corrcoef(f[t0:t1][ok], target[t0:t1][ok])[0, 1])

    ic = ic_of(feat, 0, Tn)
    ic1 = ic_of(feat, 0, half)
    ic2 = ic_of(feat, half, Tn)
    perm_ics = np.empty(n_perm)
    valid_idx = np.where(~np.isnan(feat))[0]
    for p in range(n_perm):
        shift = rng.integers(1, Tn - 1)
        f_shift = np.roll(feat, shift)
        ok = ~np.isnan(f_shift) & ~np.isnan(target)
        perm_ics[p] = np.corrcoef(f_shift[ok], target[ok])[0, 1] if ok.sum() > 30 else 0.0
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}  (n={ok_all.sum()})")


# ---------- item 6: ALGO's own vol-of-vol ----------
VOL_W = 20
VOV_W = 60
vol = np.full(len(r0), np.nan)
for t in range(VOL_W, len(r0)):
    vol[t] = r0[t - VOL_W:t].std()
volvol = np.full(len(r0), np.nan)
for t in range(VOL_W + VOV_W, len(r0)):
    volvol[t] = vol[t - VOV_W:t].std()
# target: ALGO's own next-day return / |return|
tgt_ret = np.full(len(r0), np.nan); tgt_ret[:-1] = r0[1:]
tgt_absret = np.abs(tgt_ret)

print("### Item 6: ALGO's own vol-of-vol ###")
print("H1: vol-of-vol(t) -> ALGO next-day return")
single_series_ic_perm(volvol, tgt_ret, "  volvol -> r(t+1)")
print("H2: vol-of-vol(t) -> ALGO next-day |return| (does high vol-of-vol forecast a vol regime shift?)")
single_series_ic_perm(volvol, tgt_absret, "  volvol -> |r(t+1)|")

# ---------- item 7: ALGO's own return skewness ----------
SKEW_W = 60


def roll_skew(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t - w:t]
        m = seg.mean(); s = seg.std()
        if s < 1e-12: continue
        out[t] = np.mean(((seg - m) / s) ** 3)
    return out


skew = roll_skew(r0, SKEW_W)
print("\n### Item 7: ALGO's own return skewness ###")
print("H1: skew(t) -> ALGO next-day return")
single_series_ic_perm(skew, tgt_ret, "  skew -> r(t+1)")
print("H2: |skew(t)| -> ALGO next-day |return| (vol proxy)")
single_series_ic_perm(np.abs(skew), tgt_absret, "  |skew| -> |r(t+1)|")

# ---------- item 9: ALGO jump detection ----------
JUMP_W = 60
jump_z = np.full(len(r0), np.nan)
m_roll = np.full(len(r0), np.nan); s_roll = np.full(len(r0), np.nan)
for t in range(JUMP_W, len(r0)):
    seg = r0[t - JUMP_W:t]
    mu = seg.mean(); sd = seg.std()
    if sd < 1e-12: continue
    jump_z[t] = (r0[t] - mu) / sd  # today's z-scored move vs trailing history (causal: excludes today)

print("\n### Item 9: ALGO jump/discontinuity detection (|z| of today's move vs trailing 60d) ###")
print("H1: jump_z(t) -> ALGO next-day return (does a jump predict continuation(momentum) or reversal?)")
single_series_ic_perm(jump_z, tgt_ret, "  jump_z -> r(t+1)")
print("H2: |jump_z(t)| -> ALGO next-day |return| (does a jump forecast elevated near-term vol?)")
single_series_ic_perm(np.abs(jump_z), tgt_absret, "  |jump_z| -> |r(t+1)|")

# ---------- item 8: distance from trailing high/low (per idio stock, pooled) ----------
print("\n### Item 8: distance from trailing 250d high/low (per idio stock, pooled) ###")
HL_W = 250


def rolling_minmax(x, w):
    n_ = len(x)
    out_min = np.full(n_, np.nan); out_max = np.full(n_, np.nan)
    for t in range(w, n_):
        seg = x[t - w:t]
        out_min[t] = seg.min(); out_max[t] = seg.max()
    return out_min, out_max


dist = np.full((n, T), np.nan)
for j in range(n):
    px = np.exp(logp[j + 1])  # idio stock j price series (length nt)
    lo, hi = rolling_minmax(px, HL_W)
    rng_ = hi - lo
    d = np.where(rng_ > 1e-9, (px - lo) / (rng_ + 1e-12), np.nan)
    dist[j] = d[:T]  # align to return-space length T = nt-1
dist = dist - 0.5  # center: 0 = midrange, -0.5 = at trailing low, +0.5 = at trailing high


def pooled_ic_perm(feat, target, label, n_perm=300, seed=0):
    rng = np.random.default_rng(seed)
    def flat(f, tgt, t0, t1):
        rx, ry = [], []
        for t in range(t0, min(t1, T - 1)):
            fx = f[:, t]; fy = tgt[:, t + 1]
            ok = ~np.isnan(fx) & ~np.isnan(fy)
            if ok.sum() == 0: continue
            rx.append(fx[ok]); ry.append(fy[ok])
        return np.concatenate(rx), np.concatenate(ry)
    X, Y = flat(feat, target, 0, T)
    ic = float(np.corrcoef(X, Y)[0, 1])
    half_t = T // 2
    X1, Y1 = flat(feat, target, 0, half_t); ic1 = float(np.corrcoef(X1, Y1)[0, 1])
    X2, Y2 = flat(feat, target, half_t, T); ic2 = float(np.corrcoef(X2, Y2)[0, 1])
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        feat_shift = np.empty_like(feat)
        for j in range(n):
            shift = rng.integers(1, T - 1)
            feat_shift[j] = np.roll(feat[j], shift)
        Xp, Yp = flat(feat_shift, target, 0, T)
        perm_ics[p] = np.corrcoef(Xp, Yp)[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}")


print("H1: dist_from_range(t) -> own next-day return (mean-reversion from extremes?)")
pooled_ic_perm(dist, ridio, "  dist -> r(t+1)")
print("H2: |dist_from_range(t)| (distance from midrange) -> next-day |return|")
pooled_ic_perm(np.abs(dist), np.abs(ridio), "  |dist| -> |r(t+1)|")
