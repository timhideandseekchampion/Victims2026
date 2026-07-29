"""
test_batch100_F75.py

F75 (DIAGNOSTIC, not a sizing change): compute each idio name's trailing hit-rate (% of days the
traded signal's sign matched the next-day return's sign) and report whether any names are
structurally more/less reliable than others.

Uses the FULL v10 signal (WZ_V10, sign) vs the realized next-day idio return (rs[i,t], same t-index
alignment the shipped pipeline itself uses). Computed over the OLD window, the NEW window, and their
correlation across names (a structural-persistence check: if the SAME names are reliable in both
halves, that's "structural"; if the ranking scrambles, it's noise/regime-specific).
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]

CACHE = np.load("batch100_cache.npz")
WZ_V10 = CACHE["WZ_V10"]

OLD = (500, 750); NEW = (750, nt)


def hitrate(name_i, lo, hi):
    t_range = np.arange(lo, min(hi, rs.shape[1]))
    pred = np.sign(WZ_V10[name_i, t_range])
    act = np.sign(rs[name_i, t_range])
    valid = (pred != 0) & (act != 0)
    if valid.sum() == 0:
        return np.nan
    return float((pred[valid] == act[valid]).mean())


HR_full = np.array([hitrate(i, 500, nt) for i in range(nIdio)])
HR_old = np.array([hitrate(i, *OLD) for i in range(nIdio)])
HR_new = np.array([hitrate(i, *NEW) for i in range(nIdio)])

print("=== F75 DIAGNOSTIC: per-idio-name trailing hit-rate (sign of traded WZ_V10 vs next-day return) ===")
print(f"  full test span (500-{nt}): mean={HR_full.mean():.3f}  std={HR_full.std():.3f}  "
      f"min={HR_full.min():.3f}  max={HR_full.max():.3f}")
print(f"  OLD window (500-750):      mean={HR_old.mean():.3f}  std={HR_old.std():.3f}  "
      f"min={HR_old.min():.3f}  max={HR_old.max():.3f}")
print(f"  NEW window (750-{nt}):     mean={HR_new.mean():.3f}  std={HR_new.std():.3f}  "
      f"min={HR_new.min():.3f}  max={HR_new.max():.3f}")

order_full = np.argsort(-HR_full)
print("\n  top-5 names by full-span hit-rate: " +
      ", ".join(f"idio#{i+1}({HR_full[i]:.3f})" for i in order_full[:5]))
print("  bottom-5 names by full-span hit-rate: " +
      ", ".join(f"idio#{i+1}({HR_full[i]:.3f})" for i in order_full[-5:]))

valid_mask = ~np.isnan(HR_old) & ~np.isnan(HR_new)
persist_corr = float(np.corrcoef(HR_old[valid_mask], HR_new[valid_mask])[0, 1])
print(f"\n  cross-name correlation of OLD-window hit-rate vs NEW-window hit-rate: {persist_corr:.3f}  "
      f"(near 0 => no structural persistence, names that looked 'reliable' in OLD are not "
      f"reliably the same ones in NEW)")

# how far from the null (50%) is the spread, in a statistical sense: with ~n_days-per-window
# observations per name, what SE would pure noise give around 0.5?
n_obs_new = int(NEW[1]) - int(NEW[0])
se_noise = 0.5 / np.sqrt(n_obs_new)
print(f"\n  reference: with n~{n_obs_new} obs/name (NEW window) and a true 50% coin-flip, "
      f"binomial SE ~ {se_noise:.3f}; observed cross-name std of hit-rate in NEW = {HR_new.std():.3f} "
      f"-> {'consistent with' if HR_new.std() < 2*se_noise else 'wider than'} pure sampling noise around a "
      f"common ~50% rate.")
