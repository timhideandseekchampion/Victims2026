"""Item 5 [SIGNAL]: does a stock's OWN trailing lag-1 return-autocorrelation COEFFICIENT (not the
autocorrelation relationship itself, which the ridge already captures) predict that stock's future
return or forecast reliability? Distinct question: is the LEVEL of a stock's autocorrelation itself
informative (e.g. "stocks currently showing strong momentum/reversion tendencies behave
differently going forward"), not just the sign-based trading signal already used.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
ridio = r[1:]
n, T = ridio.shape

AC_WIN = 60


def rolling_ac1(x, w):
    out = np.full(len(x), np.nan)
    for t in range(w, len(x)):
        seg = x[t - w:t]
        out[t] = np.corrcoef(seg[:-1], seg[1:])[0, 1]
    return out


print("computing causal trailing lag-1 autocorrelation coefficient per stock ...")
ac1 = np.full((n, T), np.nan)
for j in range(n):
    ac1[j] = rolling_ac1(ridio[j], AC_WIN)
print(f"done. mean ac1={np.nanmean(ac1):.4f}  std={np.nanstd(ac1):.4f}")


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
    print(f"{label}: IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}")


print("\nH1: ac1(t) -> own next-day return")
pooled_ic_perm(ac1, ridio, "  ac1 -> r(t+1)")

print("\nH2: |ac1(t)| (magnitude, either direction) -> next-day |return| (vol proxy)")
pooled_ic_perm(np.abs(ac1), np.abs(ridio), "  |ac1| -> |r(t+1)|")
