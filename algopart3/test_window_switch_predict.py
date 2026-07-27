"""Can we predict WHEN to change the fixed window (VOL_Z) rather than using one forever? Compute
rolling causal performance (trailing IC) of several VOL_Z candidates over time, check whether the
"currently winning" window changes, and whether that's predictable from an observable state variable
(overall vol level, vol-of-vol / regime instability) rather than just noise.
"""
import numpy as np, pandas as pd
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r = np.diff(lpA)
T = len(lpA)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]

CANDIDATES = (40, 50, 60, 75, 90, 120)


def volz_for(VOL_Z, VOL_WIN=20):
    vol = np.full(T, np.nan); vol[VOL_WIN:] = M._roll_std(r, VOL_WIN)
    volz = np.full(T, np.nan)
    for s in range(VOL_WIN + VOL_Z, T):
        wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    return volz


VOLZ = {vz: volz_for(vz) for vz in CANDIDATES}


def trailing_ic(feat, tnow, L=120):
    a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60: return None
    xs, ys = xs[ok], ys[ok]
    if xs.std() < 1e-12: return None
    return float(np.corrcoef(xs, ys)[0, 1])


print("computing which VOL_Z window has the best trailing 120-day IC at each checkpoint ...")
checkpoints = list(range(250, T, 20))
winners = []
for k in checkpoints:
    ics = {vz: trailing_ic(VOLZ[vz], k) for vz in CANDIDATES}
    ics = {vz: v for vz, v in ics.items() if v is not None}
    if not ics: continue
    best_vz = max(ics, key=lambda vz: abs(ics[vz]))
    winners.append((k, best_vz, ics[best_vz]))

from collections import Counter
print("distribution of the 'currently best' VOL_Z across all checkpoints:", Counter([w[1] for w in winners]))

# does the CURRENT best window persist for a while (predictable/sticky) or flip every checkpoint (noise)?
flips = sum(1 for i in range(1, len(winners)) if winners[i][1] != winners[i-1][1])
print(f"how often does the 'best window' change between consecutive checkpoints (every 20 days)? {flips}/{len(winners)-1}")

# is there a state variable that predicts whether a SHORT window (40-50) or LONG window (90-120) wins?
# candidate state variable: vol-of-vol (how much realized vol itself is fluctuating recently)
vol20 = np.full(T, np.nan); vol20[20:] = M._roll_std(r, 20)
vol_of_vol = np.full(T, np.nan)
for s in range(80, T):
    w = vol20[s-60:s]; ok = ~np.isnan(w)
    if ok.sum() > 20: vol_of_vol[s] = w[ok].std() / (w[ok].mean() + 1e-12)   # coefficient of variation of vol

short_wins = np.array([1 if w[1] <= 50 else 0 for w in winners])
vov_at_checkpoint = np.array([vol_of_vol[w[0]] if not np.isnan(vol_of_vol[w[0]]) else np.nan for w in winners])
ok = ~np.isnan(vov_at_checkpoint)
print(f"\ncorr(vol-of-vol, short-window-currently-wins): {np.corrcoef(vov_at_checkpoint[ok], short_wins[ok])[0,1]:.3f}")
print(f"mean vol-of-vol when SHORT window wins: {vov_at_checkpoint[ok][short_wins[ok]==1].mean():.3f}  "
      f"(n={int((short_wins[ok]==1).sum())})")
print(f"mean vol-of-vol when LONG window wins:  {vov_at_checkpoint[ok][short_wins[ok]==0].mean():.3f}  "
      f"(n={int((short_wins[ok]==0).sum())})")

# permutation check on that correlation
rng = np.random.default_rng(0)
obs_corr = np.corrcoef(vov_at_checkpoint[ok], short_wins[ok])[0,1]
null = [np.corrcoef(vov_at_checkpoint[ok], rng.permutation(short_wins[ok]))[0,1] for _ in range(500)]
null = np.array(null)
print(f"permutation p-value for that correlation: {100*np.mean(np.abs(null)>=abs(obs_corr)):.0f}%")
