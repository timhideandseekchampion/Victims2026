"""Actually trade the window-switch idea: each day, causally measure current vol-of-vol (trailing,
using only past data) and pick EITHER a short VOL_Z (fast-reacting) or long VOL_Z (smoothed) for
that day's vol-signal computation, instead of one fixed VOL_Z=60 forever. Score with the same
eval-mirroring accounting used all night, ALGO-leg-only (isolating the effect, comparable to the
LLVOL_VO baseline) and full portfolio.
"""
import numpy as np, pandas as pd
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)
lpA = logp[0]
r = np.diff(lpA)
T = len(lpA)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


def volz_for(VOL_Z, VOL_WIN=20):
    vol = np.full(T, np.nan); vol[VOL_WIN:] = M._roll_std(r, VOL_WIN)
    volz = np.full(T, np.nan)
    for s in range(VOL_WIN + VOL_Z, T):
        wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    return volz


vol20 = np.full(T, np.nan); vol20[20:] = M._roll_std(r, 20)
vol_of_vol = np.full(T, np.nan)
for s in range(80, T):
    w = vol20[s - 60:s]; ok = ~np.isnan(w)
    if ok.sum() > 20: vol_of_vol[s] = w[ok].std() / (w[ok].mean() + 1e-12)


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


def _side(feat, tnow, fhv):
    icf = _ic(feat, tnow, M.IC_FAST)
    if icf is None: return None
    sf = 1.0 if icf >= 0 else -1.0
    ics = [_ic_ew(feat, tnow, hl, M.IC_EW_W) for hl in M.IC_EW_HL]
    if any(x is None for x in ics): return sf * fhv
    ice = float(np.mean(ics))
    return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0


SHORT_VZ, LONG_VZ, FIXED_VZ = 45, 100, 60
VOLZ_SHORT = volz_for(SHORT_VZ); VOLZ_LONG = volz_for(LONG_VZ); VOLZ_FIXED = volz_for(FIXED_VZ)

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_algo_pos(mode, vov_thresh_pctile=50):
    POS = np.zeros((nInst, nt))
    vov_hist = []
    for k in range(130, nt):
        cur = P[0, k]; lim0 = int(dlr[0] / cur)
        if mode == "fixed":
            volz = VOLZ_FIXED
        else:
            cur_vv = None if np.isnan(vol_of_vol[k]) else vol_of_vol[k]
            vov_hist.append(cur_vv)
            valid_hist = [v for v in vov_hist if v is not None]
            if cur_vv is None or len(valid_hist) < 30:
                volz = VOLZ_FIXED
            else:
                thresh = np.percentile(valid_hist[:-1], vov_thresh_pctile)   # causal: excludes today's own value from the threshold
                volz = VOLZ_LONG if cur_vv > thresh else VOLZ_SHORT
        fhv = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else np.nan
        if np.isnan(fhv):
            POS[0, k] = 0; continue
        sig = _side(volz, k, fhv)
        if sig is None:
            POS[0, k] = 0; continue
        av = M.SWITCH_GAIN * sig * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur, -lim0, lim0))
    return POS


print(f"{'config':<20}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}")
for mode in ("fixed", "switch"):
    POS = build_algo_pos(mode)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{mode:<20}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>8.1f}{min(scs):>9.1f}")
