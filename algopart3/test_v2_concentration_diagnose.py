"""Diagnose the days 610-670 concentration where LLBOOST_V2 underperforms LLBOOST. Since the idio
book is identical between them, isolate the ALGO leg's day-by-day PnL specifically, and check
whether the short/long MOM_LB switch is whipsawing (flipping frequently) in this period -- the
same "coin flip near a hard boundary" pattern diagnosed earlier for the agree/disagree gate.
"""
import numpy as np, pandas as pd
import SAFE_llboost, SAFE_llboost_v2

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
logp = np.log(P)
lpA = logp[0]
r0 = np.diff(lpA)

print("computing ALGO leg positions for both strategies (idio book is identical, isolate leg 0) ...")
algo_v1 = np.zeros(nt); algo_v2 = np.zeros(nt)
for k in range(400, 700):
    cur0 = P[0, k]
    algo_v1[k] = SAFE_llboost._algo_vol_shares(lpA[:k + 1], cur0, 100_000.0)
    algo_v2[k] = SAFE_llboost_v2._algo_vol_shares(lpA[:k + 1], cur0, 100_000.0)
print("done")

print("\n=== day-by-day ALGO-leg PnL difference (v2 minus v1), days 400-700 ===")
pnl_diff = np.zeros(nt)
for d in range(401, 700):
    pl_v1 = algo_v1[d - 1] * (P[0, d] - P[0, d - 1])
    pl_v2 = algo_v2[d - 1] * (P[0, d] - P[0, d - 1])
    pnl_diff[d] = pl_v2 - pl_v1

cum_diff = np.cumsum(pnl_diff[401:700])
worst_start = 401 + int(np.argmin(cum_diff))
print(f"cumulative (v2-v1) ALGO PnL over 401-700 trends... checking the 590-680 sub-stretch:")
for d in range(590, 680, 5):
    print(f"  day={d}: algo_v1_pos={algo_v1[d]:>8.0f}  algo_v2_pos={algo_v2[d]:>8.0f}  "
          f"5day_pnl_diff={pnl_diff[d:d+5].sum():>+9.1f}")

print("\n=== MOM_LB switch frequency check (how often does v2's regime flip vs v1's fixed 10?) ===")
VOL_WIN, VOL_Z = 20, 60


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol = np.full(nt - 1, np.nan); vol[VOL_WIN:] = roll_std(r0, VOL_WIN)
volz = np.full(nt - 1, np.nan)
for s in range(VOL_WIN + VOL_Z, nt - 1):
    wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)

regime = np.where(volz[590:680] > 0, "SHORT(7)", "LONG(12)")
flips = sum(1 for i in range(1, len(regime)) if regime[i] != regime[i - 1])
print(f"days 590-680: regime flips {flips} times in {len(regime)} days "
      f"(~1 flip every {len(regime)/max(flips,1):.1f} days)")
print(f"volz values near the boundary (|volz|<0.3, i.e. borderline elevated/calm classification):")
near_boundary = [(590 + i, volz[590 + i]) for i in range(90) if abs(volz[590 + i]) < 0.3]
print(f"  {len(near_boundary)}/90 days have |volz| < 0.3 (borderline): {near_boundary[:10]}")
