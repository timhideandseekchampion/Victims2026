"""Demonstrate the discrete agree/disagree gate mechanism: for the ALGO leg's vol-switch _side()
function, the fast/slow IC agreement check `(ice >= 0) == (icf >= 0)` is a hard binary step -- when
they agree, trade at full size; when they disagree (even by a hair), go COMPLETELY flat. This shows
which specific days flip between "trade" and "flat" as IC_EW_W changes slightly, proving the
jaggedness comes from days where the fast EW-IC sits very close to zero (or icf does), so a tiny
window-length change flips its sign.
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r0 = np.diff(lpA)
T = len(lpA)

VOL_WIN, VOL_Z, IC_FAST, IC_EW_HL = 20, 60, 90, (20, 45)


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


print("=== for each day (from day 300 on), compute icf, ice at IC_EW_W=145 vs 150 vs 155 ===")
flips_145_150 = []
flips_150_155 = []
rows = []
for tnow in range(300, T):
    icf = _ic(volz, tnow, IC_FAST)
    if icf is None:
        continue
    ice_vals = {}
    for W in (145, 150, 155):
        ics = [_ic_ew(volz, tnow, hl, W) for hl in IC_EW_HL]
        if any(x is None for x in ics):
            ice_vals[W] = None
        else:
            ice_vals[W] = float(np.mean(ics))
    if any(v is None for v in ice_vals.values()):
        continue
    agree = {W: (ice_vals[W] >= 0) == (icf >= 0) for W in ice_vals}
    rows.append((tnow, icf, ice_vals, agree))
    if agree[145] != agree[150]:
        flips_145_150.append(tnow)
    if agree[150] != agree[155]:
        flips_150_155.append(tnow)

print(f"total days checked: {len(rows)}")
print(f"days where agree-status flips between W=145 and W=150: {len(flips_145_150)}")
print(f"days where agree-status flips between W=150 and W=155: {len(flips_150_155)}")

print("\n=== example flip days (W=145 vs 150), showing how close ice sits to zero ===")
for tnow in flips_145_150[:8]:
    icf, ice_vals, agree = next((r[1], r[2], r[3]) for r in rows if r[0] == tnow)
    print(f"  day={tnow}: icf={icf:+.4f}  ice(145)={ice_vals[145]:+.5f} (agree={agree[145]})  "
          f"ice(150)={ice_vals[150]:+.5f} (agree={agree[150]})")

print(f"\ntotal flip days across the whole file (145->150, 150->155): "
      f"{len(flips_145_150)} + {len(flips_150_155)} = {len(flips_145_150)+len(flips_150_155)} "
      f"(out of {len(rows)} tradeable days) -- each flip is a FULL SIZE <-> FLAT jump for that day")
