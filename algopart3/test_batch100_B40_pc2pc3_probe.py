"""
test_batch100_B40_pc2pc3_probe.py  [DIAGNOSTIC]

B40: Re-run the PC2/PC3 second-factor probe on v10's ACTUAL current residuals (after the beta-adjusted
target and rank-stability blend), not an earlier version's residual. The original probe
(test_pc2_probe.py) used a single static in-sample hl=1000 ridge fit on the RAW next-day-return target
(pre-beta-adjustment, pre-rank-stability) -- found PC1 (~ALGO) predictive but PC2/PC3 null. Re-checking
on the genuine walk-forward residual of what v10 ACTUALLY trades today: WZ_FULL (batch100_shared) is
the exact per-name, per-day forecast v10 uses (ridge ensemble on the beta-adjusted target + BLEND
reversion + pairwise boost + rank-stability blend, causal, walk-forward) -- so "residual" here is the
part of the REALIZED idio return WZ_FULL does not explain, using v10's real signal, not a proxy.

METHOD: fit one pooled scalar c (OLS, no intercept) converting WZ_FULL's unitless score to expected
return units; residual[j,t] = rs[j,t] - c*WZ_FULL[j,t] for every traded day. PCA the residual's
CROSS-STOCK covariance structure (computed from each stock's residual TIME SERIES, same convention as
the original probe): does PC2/PC3 of the residual (today) predict the residual (tomorrow), the way the
original PC1(~ALGO) predicted raw returns? Plus: max leftover pairwise lead-lag in the residual
(compare to the pre-model raw-price max and the original probe's post-ridge max).
"""
import numpy as np
from batch100_shared import nIdio, nt, days, rs, WZ_FULL, BOOST_MIN_DAY

# restrict to the range where the full v10 mechanism (incl. boost) is active
t0, t1 = BOOST_MIN_DAY, nt - 2  # need t+1 <= nt-2 so rs[:, t+1] is defined (rs has nt-1 cols, 0..nt-2)
ts = np.arange(t0, t1 + 1)
print(f"=== B40: v10's actual residual, active-mechanism window t in [{t0},{t1}] ({len(ts)} days) ===")

WZt = WZ_FULL[:, ts]          # (nIdio, ndays) -- v10's actual forecast made at day t
RSt = rs[:, ts]                # (nIdio, ndays) -- the realized return v10's forecast at t is FOR (day t)

# pooled OLS scalar: rs[j,t] ~= c * WZ_FULL[j,t] (no intercept; wz is causally ~mean-zero cross-sectionally)
Xf = WZt.ravel(); Yf = RSt.ravel()
c = float((Xf @ Yf) / (Xf @ Xf))
print(f"pooled OLS coefficient (wz -> return units): c={c:.6f}")

RESID = RSt - c * WZt   # (nIdio, ndays): v10's actual current residual
print(f"residual built, shape={RESID.shape}")
print(f"residual pooled IC check (sanity: wz should have SOME explanatory power): "
      f"corr(wz,rs)={np.corrcoef(Xf, Yf)[0,1]:+.4f}   corr(wz,resid)={np.corrcoef(Xf, RESID.ravel())[0,1]:+.4f}")

print("\n=== 1. PCA of the residual's cross-stock covariance (each stock's residual TIME SERIES) ===")
Rz = (RESID - RESID.mean(1, keepdims=True)) / (RESID.std(1, keepdims=True) + 1e-12)
cov = np.cov(Rz)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(-evals)
evals = evals[order]; evecs = evecs[:, order]
print("variance explained, top 5 PCs of the RESIDUAL:", (evals[:5] / evals.sum()).round(3))
pc1, pc2, pc3 = evecs[:, 0], evecs[:, 1], evecs[:, 2]
pc1_t = pc1 @ Rz; pc2_t = pc2 @ Rz; pc3_t = pc3 @ Rz


def ic_and_perm(feat_today, target, N=300, seed=0):
    """feat_today[t] (today) predicts target[:,t+1] (tomorrow's residual), pooled across all stocks."""
    x = feat_today[:-1]; Y = target[:, 1:]
    ics = np.array([np.corrcoef(x, Y[j])[0, 1] for j in range(Y.shape[0])])
    obs = np.abs(ics).mean()
    rng = np.random.default_rng(seed)
    null = np.empty(N)
    for i in range(N):
        xp = rng.permutation(x)
        null[i] = np.mean([abs(np.corrcoef(xp, Y[j])[0, 1]) for j in range(Y.shape[0])])
    p = float(np.mean(null >= obs))
    return obs, null.mean(), p


print("\n=== 2. does any residual PC (today) predict ANY stock's residual (tomorrow)? ===")
for lbl, feat in [("PC1 (of residual)", pc1_t), ("PC2 (of residual)", pc2_t), ("PC3 (of residual)", pc3_t)]:
    obs, nullmean, p = ic_and_perm(feat, RESID)
    print(f"  {lbl:<20} mean|IC| across {nIdio} stocks = {obs:.4f}  (perm null mean {nullmean:.4f})  "
          f"p={100*p:.0f}%")

print("\n=== 3. leftover pairwise lead-lag structure IN THE RESIDUAL (same-day resid[i,t] -> resid[j,t+1]) ===")
resid_ac = np.array([np.corrcoef(RESID[j, :-1], RESID[j, 1:])[0, 1] for j in range(nIdio)])
print(f"  residual own lag-1 autocorr: mean {resid_ac.mean():+.4f}  frac>0 {(resid_ac > 0).mean():.2f}")
avg_resid_corr = np.corrcoef(RESID)
off = avg_resid_corr[np.triu_indices(nIdio, 1)]
print(f"  avg pairwise residual cross-correlation (same-day): mean {off.mean():+.4f}")
resid_ll = np.array([[np.corrcoef(RESID[i, :-1], RESID[j, 1:])[0, 1] if i != j else np.nan
                       for j in range(nIdio)] for i in range(nIdio)])
print(f"  max |leftover pairwise lead-lag in v10's residual| = {np.nanmax(np.abs(resid_ll)):.4f}  "
      f"(original probe's post-ridge max, on the OLD raw-target residual, was 0.171)")

print("\n=== B40 interpretation ===")
print("  See printed p-values above for PC2/PC3 predictive power on v10's genuine walk-forward "
      "residual, and the max leftover lead-lag vs. the original probe's post-ridge number -- "
      "diagnostic only, no candidate mechanism or pass/fail here.")
