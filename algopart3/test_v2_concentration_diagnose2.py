"""Cleaner diagnostic: cumulative ALGO-leg PnL difference (v2-v1) over a wide window, plus properly
fixed regime-switch frequency check.
"""
import numpy as np, pandas as pd
import SAFE_llboost, SAFE_llboost_v2

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r0 = np.diff(lpA)

print("computing ALGO leg positions for both strategies, days 350-700 ...")
algo_v1 = np.zeros(nt); algo_v2 = np.zeros(nt)
for k in range(350, 700):
    cur0 = P[0, k]
    algo_v1[k] = SAFE_llboost._algo_vol_shares(lpA[:k + 1], cur0, 100_000.0)
    algo_v2[k] = SAFE_llboost_v2._algo_vol_shares(lpA[:k + 1], cur0, 100_000.0)
print("done")

pnl_diff = np.zeros(nt)
for d in range(351, 700):
    pnl_diff[d] = (algo_v2[d - 1] - algo_v1[d - 1]) * (P[0, d] - P[0, d - 1])

print("\n=== cumulative (v2-v1) ALGO PnL, every 20 days, 350-700 ===")
cum = np.cumsum(pnl_diff[351:700])
for i in range(0, len(cum), 20):
    print(f"  day={351+i}: cumulative diff={cum[i]:>+9.1f}")

sign_diff = np.sign(algo_v2[350:700]) != np.sign(algo_v1[350:700])
print(f"\ndays where v1 and v2 hold OPPOSITE signs (350-700): {sign_diff.sum()}/{350} "
      f"({sign_diff.mean()*100:.1f}%)")

print("\n=== regime-switch frequency (v2's short/long MOM_LB regime) ===")
VOL_WIN, VOL_Z = 20, 60


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol = np.full(nt, np.nan); vol[VOL_WIN:] = roll_std(r0, VOL_WIN)
volz = np.full(nt, np.nan)
for s in range(VOL_WIN + VOL_Z, nt):
    wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)

regime = volz[590:680] > 0
flips = sum(1 for i in range(1, len(regime)) if regime[i] != regime[i - 1])
print(f"days 590-680: regime flips {flips} times in {len(regime)} days")
near_boundary = np.abs(volz[590:680]) < 0.3
print(f"days with |volz|<0.3 (borderline elevated/calm): {near_boundary.sum()}/90")

flips_full = sum(1 for i in range(1, len(volz[400:700]) - 1) if not np.isnan(volz[400+i]) and
                  (volz[400+i] > 0) != (volz[400+i-1] > 0))
print(f"regime flips over the FULL 400-700 stretch: {flips_full} times in 300 days "
      f"(~1 flip every {300/max(flips_full,1):.1f} days)")
