"""Two more untried angles, genuinely different from anything tested tonight:

H1: EXTREME single-day move reversal. The shipped BLEND=0.3 leg fades a smoothed 10-day cumulative
return -- distinct mechanism from asking whether a stock's SINGLE extreme-day move (top/bottom X%
of its OWN historical |return| distribution, causal/expanding) snaps back specifically the next
day. A sharp one-day outlier could have different reversion dynamics than a smooth multi-day drift.

H2: per-stock volatility-of-volatility (VoV) as a predictor of THAT SAME stock's own next return.
VoV was already used as a DAY-LEVEL clustering feature (regime detection across all stocks at once)
earlier tonight, but never tested as a per-stock predictive feature for that stock's own future
return -- distinct question: "is MY OWN vol currently unstable" vs "is the whole market's vol regime
unstable."

Both tested with full-sample pooled IC + permutation (circular shift, preserves autocorrelation)
+ H1/H2 persistence split, the same bar as every other finding tonight.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
ridio = r[1:]  # (49, T)
n, T = ridio.shape


def pooled_ic_perm(feat, target, label, n_perm=300):
    rng = np.random.default_rng(0)
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
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}  perm_std={perm_ics.std():.4f}")
    return ic, pval


print("=== H1: extreme single-day move -> next-day reversal ===")
EXPAND_MIN = 100
extreme_signed = np.full((n, T), np.nan)  # signed indicator: +1 if extreme up, -1 if extreme down, 0 else
PCTL = 0.90  # top/bottom 10% by |return| relative to the stock's own expanding history
for j in range(n):
    for t in range(EXPAND_MIN, T):
        hist = np.abs(ridio[j, :t])
        thr = np.quantile(hist, PCTL)
        if np.abs(ridio[j, t]) >= thr:
            extreme_signed[j, t] = -np.sign(ridio[j, t])  # predicted reversal direction if mean-reverting
print(f"extreme-day flag rate: {np.nanmean(~np.isnan(extreme_signed)):.3f} (target ~{1-PCTL:.2f})")
# feature = extreme_signed (predicted reversal direction, 0 on non-extreme days)
feat = np.nan_to_num(extreme_signed, nan=0.0)
pooled_ic_perm(feat, ridio, "  predicted-reversal-dir -> r(t+1)")

# also check: among ONLY extreme days, what's the hit rate of the predicted reversal direction?
mask = ~np.isnan(extreme_signed)
pred = extreme_signed[mask]
actual_next = np.full((n, T), np.nan); actual_next[:, :-1] = ridio[:, 1:]
act = actual_next[mask]
ok = ~np.isnan(act)
hits = (np.sign(act[ok]) == pred[ok]).mean()
print(f"  among extreme-day flags only: next-day sign matches predicted-reversal {hits*100:.1f}% of the time "
      f"(n={ok.sum()}, chance=50%)")

print("\n=== H2: per-stock vol-of-vol -> that stock's own next-day return ===")
VOL_W, VOV_W = 20, 60
vol = np.full((n, T), np.nan)
for j in range(n):
    for t in range(VOL_W, T):
        vol[j, t] = ridio[j, t - VOL_W:t].std()
vov = np.full((n, T), np.nan)
for j in range(n):
    for t in range(VOL_W + VOV_W, T):
        w = vol[j, t - VOV_W:t]
        ok = ~np.isnan(w)
        if ok.sum() > 20:
            vov[j, t] = w[ok].std() / (w[ok].mean() + 1e-12)  # coefficient of variation of own vol
vovz = np.full((n, T), np.nan)
Z_W = 120
for j in range(n):
    for t in range(VOL_W + VOV_W + Z_W, T):
        w = vov[j, t - Z_W:t]
        ok = ~np.isnan(w)
        if ok.sum() > 30:
            vovz[j, t] = (vov[j, t] - w[ok].mean()) / (w[ok].std() + 1e-12)
pooled_ic_perm(vovz, ridio, "  own vovz -> r(t+1)")
abs_ridio = np.abs(ridio)
pooled_ic_perm(vovz, abs_ridio, "  own vovz -> |r(t+1)| (magnitude)")
