"""Batch of 80, Category B (items 21-40): new signal features not yet tried this session.
Stage 1 only: causal pooled IC + circular-shift permutation test + H1/H2 persistence split
(per-idio-stock features) or single-series version (ALGO-specific features).
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
ridio = r[1:]
n, T = ridio.shape
r0 = r[0]


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
    return pval


def single_series_ic_perm(feat, target, label, n_perm=1000, seed=0):
    rng = np.random.default_rng(seed)
    ok_all = ~np.isnan(feat) & ~np.isnan(target)
    Tn = len(feat); half = Tn // 2
    def ic_of(f, t0, t1):
        ok = ok_all[t0:t1]
        if ok.sum() < 30: return np.nan
        return float(np.corrcoef(f[t0:t1][ok], target[t0:t1][ok])[0, 1])
    ic = ic_of(feat, 0, Tn); ic1 = ic_of(feat, 0, half); ic2 = ic_of(feat, half, Tn)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        shift = rng.integers(1, Tn - 1)
        f_shift = np.roll(feat, shift)
        ok = ~np.isnan(f_shift) & ~np.isnan(target)
        perm_ics[p] = np.corrcoef(f_shift[ok], target[ok])[0, 1] if ok.sum() > 30 else 0.0
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}  (n={ok_all.sum()})")
    return pval


tgt_ret0 = np.full(len(r0), np.nan); tgt_ret0[:-1] = r0[1:]
tgt_absret0 = np.abs(tgt_ret0)

# ---------- item 21/22: idio & ALGO kurtosis ----------
KURT_W = 60
def roll_kurt(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t-w:t]; m = seg.mean(); s = seg.std()
        if s < 1e-12: continue
        out[t] = np.mean(((seg-m)/s)**4) - 3.0
    return out

print("### Item 21: idio stock's own rolling excess kurtosis (per stock, pooled) ###")
kurt_idio = np.full((n, T), np.nan)
for j in range(n):
    kurt_idio[j] = roll_kurt(ridio[j], KURT_W)
pooled_ic_perm(kurt_idio, ridio, "  kurt -> own r(t+1)")
pooled_ic_perm(np.abs(kurt_idio - np.nanmean(kurt_idio)), np.abs(ridio), "  |kurt-dev| -> |r(t+1)|")

print("\n### Item 22: ALGO's own rolling excess kurtosis ###")
kurt_algo = roll_kurt(r0, KURT_W)
single_series_ic_perm(kurt_algo, tgt_ret0, "  algo_kurt -> r(t+1)")
single_series_ic_perm(np.abs(kurt_algo), tgt_absret0, "  |algo_kurt| -> |r(t+1)|")

# ---------- item 23: Hurst-exponent-style long-memory measure on ALGO ----------
print("\n### Item 23: ALGO rolling Hurst-exponent-style (R/S) long-memory measure ###")
HURST_W = 120
def roll_hurst(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t-w:t]
        seg = seg - seg.mean()
        cs = np.cumsum(seg)
        R = cs.max() - cs.min()
        S = seg.std()
        if S < 1e-12: continue
        out[t] = np.log(R/S + 1e-12) / np.log(w)
    return out
hurst = roll_hurst(r0, HURST_W)
single_series_ic_perm(hurst, tgt_ret0, "  hurst -> r(t+1)")
single_series_ic_perm(hurst, tgt_absret0, "  hurst -> |r(t+1)| (trending vs mean-reverting regime -> vol?)")

# ---------- item 24: week-start jump magnitude (distinct from periodicity) ----------
print("\n### Item 24: 'week-start' (every-5th-day) jump MAGNITUDE (not periodicity) ###")
day_idx = np.arange(T) % 5
is_weekstart = (day_idx == 0).astype(float)
single_series_ic_perm(is_weekstart[:len(r0)]*1.0, tgt_absret0, "  is_weekstart -> |r(t+1)|")

# ---------- item 25: cross-sectional dispersion -> next-day dispersion / individual returns ----------
print("\n### Item 25: cross-sectional return dispersion (across 50 idio stocks) ###")
disp = np.nanstd(ridio, axis=0)  # (T,)
disp_tgt = np.full(T, np.nan); disp_tgt[:-1] = disp[1:]
single_series_ic_perm(disp, disp_tgt, "  disp(t) -> disp(t+1)")
disp_bcast = np.tile(disp, (n,1))
pooled_ic_perm(disp_bcast, ridio, "  disp(t) -> individual r(t+1)")

# ---------- item 26: rolling beta-to-ALGO stability ----------
print("\n### Item 26: idio stock's rolling beta-to-ALGO STABILITY (change in beta) ###")
BETA_W = 60
beta_roll = np.full((n, T), np.nan)
for j in range(n):
    for t in range(BETA_W, T):
        x = r0[t-BETA_W:t]; y = ridio[j, t-BETA_W:t]
        if x.std() < 1e-12: continue
        beta_roll[j, t] = np.cov(x, y)[0,1] / (x.var() + 1e-12)
beta_stability = np.full((n, T), np.nan)
beta_stability[:, 1:] = -np.abs(np.diff(beta_roll, axis=1))  # negative = more change = less stable
pooled_ic_perm(beta_stability, ridio, "  -|beta_change| -> r(t+1)")
pooled_ic_perm(beta_stability, np.abs(ridio), "  -|beta_change| -> |r(t+1)|")

# ---------- item 28: ALGO's own lag-1 autocorrelation coefficient ----------
print("\n### Item 28: ALGO's own trailing lag-1 autocorrelation coefficient ###")
def rolling_ac1(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t-w:t]
        out[t] = np.corrcoef(seg[:-1], seg[1:])[0,1]
    return out
ac1_algo = rolling_ac1(r0, 60)
single_series_ic_perm(ac1_algo, tgt_ret0, "  algo_ac1 -> r(t+1)")
single_series_ic_perm(np.abs(ac1_algo), tgt_absret0, "  |algo_ac1| -> |r(t+1)|")

# ---------- item 29: cross-sectional rank-of-volatility ----------
print("\n### Item 29: cross-sectional rank-of-volatility (per stock) ###")
VOL_W2 = 60
vol_idio = np.full((n, T), np.nan)
for j in range(n):
    for t in range(VOL_W2, T):
        vol_idio[j, t] = ridio[j, t-VOL_W2:t].std()
from scipy.stats import rankdata
rank_vol = np.full((n, T), np.nan)
for t in range(VOL_W2, T):
    rank_vol[:, t] = (rankdata(vol_idio[:, t]) - 1)/(n-1) - 0.5
pooled_ic_perm(rank_vol, ridio, "  rank_vol -> r(t+1)")
pooled_ic_perm(rank_vol, np.abs(ridio), "  rank_vol -> |r(t+1)|")

# ---------- item 30: ALGO realized range proxy vs realized vol ----------
print("\n### Item 30: ALGO realized 'range' proxy (max-min over trailing window) vs realized vol ###")
RANGE_W = 20
def roll_range(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t-w:t]
        out[t] = seg.max() - seg.min()
    return out
rng_algo = roll_range(r0, RANGE_W)
vol_algo = np.full(len(r0), np.nan)
for t in range(RANGE_W, len(r0)):
    vol_algo[t] = r0[t-RANGE_W:t].std()
range_vs_vol = rng_algo / (vol_algo + 1e-12) - np.sqrt(RANGE_W)  # excess "jumpiness" beyond normal-range expectation
single_series_ic_perm(range_vs_vol, tgt_ret0, "  range/vol excess -> r(t+1)")
single_series_ic_perm(range_vs_vol, tgt_absret0, "  range/vol excess -> |r(t+1)|")

print("\nCategory B batch complete.")
