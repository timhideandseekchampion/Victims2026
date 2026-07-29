"""
test_batch100_I92.py

I92 (DIAGNOSTIC): paired bootstrap over trading days, testing the statistical significance of v10's
total improvement over the ORIGINAL SAFE_llboost.py (the pre-v2 baseline this whole file sequence
descends from). Paired design: same days, diff = v10's daily PnL - original's daily PnL, i.i.d.
day-level bootstrap resample of the diff series (matching the spirit of the existing N=25 paired
synthetic-draw bootstraps in stress_test_synthetic.py / test_v10_stress_synthetic.py, but here
resampling the REAL days directly rather than fresh synthetic draws).
"""
import numpy as np
import batch100_versions_shared as S

np.random.seed(0)
nt = S.nt
WIN = (500, nt)  # 500 days: 501-1000 (OLD+NEW combined eval window)

pnl_orig = S.daily_pnl(S.POS["orig"], *WIN)
pnl_v10 = S.daily_pnl(S.POS["v10"], *WIN)
diff = pnl_v10 - pnl_orig
n = len(diff)

print(f"Paired daily PnL diff (v10 - original), {n} days (501-1000):")
print(f"  mean diff/day: {diff.mean():.2f}   std: {diff.std():.2f}   total: {diff.sum():.0f}")
print(f"  win rate (days v10 > original): {100*(diff > 0).mean():.1f}%")

wo_orig, wn_orig = S.wscore(S.POS["orig"], *S.OLD), S.wscore(S.POS["orig"], *S.NEW)
wo_v10, wn_v10 = S.wscore(S.POS["v10"], *S.OLD), S.wscore(S.POS["v10"], *S.NEW)
print(f"\n  score(original): OLD={wo_orig:.1f} NEW={wn_orig:.1f}   (README ref 774.1/828.6)")
print(f"  score(v10):      OLD={wo_v10:.1f} NEW={wn_v10:.1f}   (README ref 871.0/912.6)")

N_BOOT = 20000
idx = np.arange(n)
boot_means = np.empty(N_BOOT)
for b in range(N_BOOT):
    samp = np.random.choice(idx, size=n, replace=True)
    boot_means[b] = diff[samp].mean()

lo, hi = np.percentile(boot_means, [2.5, 97.5])
p_le_zero = float((boot_means <= 0).mean())
print(f"\n=== paired bootstrap (day-level, i.i.d. resample, N={N_BOOT}) ===")
print(f"  mean of bootstrap means: {boot_means.mean():.2f}  (point estimate {diff.mean():.2f})")
print(f"  95% CI for mean daily diff: [{lo:.2f}, {hi:.2f}]")
print(f"  fraction of bootstrap draws with mean diff <= 0 (one-sided 'p-value' that v10 > original): "
      f"{p_le_zero:.4f}")

# paired t-test as a standard complement
from scipy import stats
tstat, pval = stats.ttest_1samp(diff, 0.0)
print(f"\n  paired t-test: t={tstat:.2f}, two-sided p={pval:.2e}")
