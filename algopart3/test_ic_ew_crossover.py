"""Investigate the user's question: does a crossover between the two IC_EW_HL half-lives (20, 45)
-- one positive, one negative -- interact with calm-vol days to flip the ALGO position between
short and long? Computes the two fast EW-ICs SEPARATELY (not pre-averaged) alongside the slow IC
and the day's volz, to see concretely what drives sign flips.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r0 = np.diff(lpA)
T = len(lpA)

VOL_WIN, VOL_Z, IC_FAST, IC_EW_HL, IC_EW_W = 20, 60, 90, (20, 45), 200


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol = np.full(T, np.nan); vol[VOL_WIN:] = roll_std(r0, VOL_WIN)
volz = np.full(T, np.nan)
for s in range(VOL_WIN + VOL_Z, T):
    wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]


def _ic(feat, tnow, L):
    a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60: return None
    xs, ys = xs[ok], ys[ok]
    if xs.std() < 1e-12: return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _ic_ew(feat, tnow, HL, W):
    a = max(0, tnow - W); xs = feat[a:tnow]; ys = ret1[a:tnow]
    w = (0.5 ** (1.0 / HL)) ** ((tnow - 1) - np.arange(a, tnow))
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60: return None
    xs, ys, w = xs[ok], ys[ok], w[ok]; sw = w.sum()
    mx = (w * xs).sum() / sw; my = (w * ys).sum() / sw
    cxy = (w * (xs - mx) * (ys - my)).sum() / sw
    vx = (w * (xs - mx) ** 2).sum() / sw; vy = (w * (ys - my) ** 2).sum() / sw
    if vx < 1e-24 or vy < 1e-24: return None
    return float(cxy / np.sqrt(vx * vy))


rows = []
for tnow in range(300, T):
    icf = _ic(volz, tnow, IC_FAST)
    if icf is None or np.isnan(volz[tnow]): continue
    ice20 = _ic_ew(volz, tnow, 20, IC_EW_W)
    ice45 = _ic_ew(volz, tnow, 45, IC_EW_W)
    if ice20 is None or ice45 is None: continue
    ice_avg = (ice20 + ice45) / 2
    fh = float(np.clip(volz[tnow], -3, 3) / 3)
    sf = 1.0 if icf >= 0 else -1.0
    agree = (ice_avg >= 0) == (icf >= 0)
    position_sign = np.sign(sf * fh) if agree else 0.0
    crossover = (ice20 >= 0) != (ice45 >= 0)  # the two half-lives disagree with EACH OTHER
    rows.append((tnow, icf, ice20, ice45, ice_avg, fh, sf, agree, position_sign, crossover))

df = pd.DataFrame(rows, columns=["t", "icf", "ice20", "ice45", "ice_avg", "fh", "sf", "agree",
                                  "pos_sign", "crossover"])
print(f"total days: {len(df)}")
print(f"crossover days (hl20 and hl45 disagree with each other): {df['crossover'].sum()} "
      f"({df['crossover'].mean()*100:.1f}%)")
print(f"calm-vol days (fh < 0, i.e. volz negative): {(df['fh']<0).sum()} ({(df['fh']<0).mean()*100:.1f}%)")
print(f"calm-vol AND crossover days: {((df['fh']<0)&df['crossover']).sum()}")

print("\n=== does fh's OWN sign (calm vs elevated vol) flip the position, independent of crossover? ===")
print("sample of days where fh flips from + to - or vice versa (sf held at same value), showing pos_sign:")
sf_stable = df[df["sf"] == df["sf"].iloc[len(df)//2]]  # just look at a stretch with stable sf
print(sf_stable[["t", "fh", "sf", "agree", "pos_sign"]].iloc[100:115].to_string(index=False))

print("\n=== example CROSSOVER days during CALM vol (fh<0), showing all components ===")
cross_calm = df[(df["fh"] < 0) & df["crossover"]]
print(cross_calm[["t", "icf", "ice20", "ice45", "ice_avg", "fh", "sf", "agree", "pos_sign"]].head(10).to_string(index=False))
