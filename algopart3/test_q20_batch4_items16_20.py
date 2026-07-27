"""Batch 4, items 16-20 (cheap/simple checks).
16 [SIGNAL] Day-of-week / periodicity effects -- generator-artifact check.
17 [SIGNAL] Cross-sectional rank-of-momentum (a stock's momentum RANK among the other 48).
18 [MECH] BOOST_N_CANDIDATES sensitivity -- restrict Bonferroni divisor to a liquid subset.
19 [SIGNAL] Idiosyncratic vol LEVEL (unconditional, not current regime) as a static reliability hint.
20 [MECH] Commission-rate-aware position rounding -- does price granularity waste $10k cap room?
"""
import numpy as np, pandas as pd
from scipy import stats

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
n, T = rs.shape

print("### Item 16: day-of-week / periodicity effects (generator-artifact check) ###")
for period in (5, 7, 10, 20):
    day_idx = np.arange(T) % period
    means = [rs[:, day_idx == d].mean() for d in range(period)]
    stds = [rs[:, day_idx == d].std() for d in range(period)]
    grand = rs.mean()
    f_stat, p_val = stats.f_oneway(*[rs[:, day_idx == d].ravel() for d in range(period)])
    print(f"  period={period}: per-phase means={[f'{m*1e4:+.2f}bp' for m in means]}  ANOVA p={p_val:.3f}")

print("\n### Item 17: cross-sectional rank-of-momentum ###")
MOM_W = 10
mom = np.full((n, T), np.nan)
for t in range(MOM_W, T):
    mom[:, t] = logp[1:, t] - logp[1:, t - MOM_W]
rank_mom = np.full((n, T), np.nan)
for t in range(MOM_W, T):
    col = mom[:, t]
    rank_mom[:, t] = (stats.rankdata(col) - 1) / (n - 1) - 0.5  # centered rank in [-0.5,+0.5]


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


pooled_ic_perm(rank_mom, rs, "  rank_mom -> r(t+1)")

print("\n### Item 19: idiosyncratic vol LEVEL (unconditional, long-run) as a static reliability hint ###")
vol_level = np.nanstd(rs, axis=1)  # (50,) full-sample per-stock vol, static
order = np.argsort(vol_level)
print(f"  per-stock vol level range: {vol_level.min()*1e4:.1f}bp .. {vol_level.max()*1e4:.1f}bp")
print(f"  lowest-vol 10: {vol_level[order[:10]]*1e4}")
print(f"  highest-vol 10: {vol_level[order[-10:]]*1e4}")
# does a stock's static vol level predict ITS OWN ridge-forecast reliability (proxy: |return| predictability)?
# use simple lag-1 own-return IC as the "predictability" measure, split stocks into vol terciles
own_ic = np.array([np.corrcoef(rs[j, :-1], rs[j, 1:])[0, 1] for j in range(n)])
terc = pd.qcut(vol_level, 3, labels=["low", "mid", "high"])
for lab in ["low", "mid", "high"]:
    mask = np.array(terc == lab)
    print(f"  vol-tercile={lab}: mean own-return IC={own_ic[mask].mean():+.4f}  n={mask.sum()}")
corr_vol_ic = np.corrcoef(vol_level, own_ic)[0, 1]
print(f"  correlation(vol_level, own_ic across 50 stocks) = {corr_vol_ic:+.3f}")

print("\n### Item 20: commission-rate-aware position rounding (mechanical check) ###")
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
cur_last = P_[:, -1]
lim = (dlr / cur_last)
achieved_dollars = np.floor(lim) * cur_last
waste_frac = 1 - achieved_dollars / dlr
print("  per-stock cap utilization (using final-day prices):")
order2 = np.argsort(-waste_frac)
for i in order2[:8]:
    print(f"    inst {i}: price=${cur_last[i]:.2f}  cap_shares={int(lim[i])}  "
          f"achieved=${achieved_dollars[i]:,.0f}  waste={waste_frac[i]*100:.2f}%")
print(f"  mean waste fraction across all 51 instruments: {waste_frac.mean()*100:.3f}%")
print(f"  max waste fraction: {waste_frac.max()*100:.3f}%  (inst {np.argmax(waste_frac)})")
