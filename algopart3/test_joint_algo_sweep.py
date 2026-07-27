"""Score-maximization pass: SAFE_llvol's ALGO-leg parameters (SWITCH_GAIN, COMBINE_GAIN, IC_FAST,
IC_EW_HL) were tuned SEQUENTIALLY/one-at-a-time per the file's own docstring history. A joint grid
search might find a better combined peak that coordinate-wise tuning missed. This is cheap: the
vol+momentum ALGO leg needs only instrument 0's own price history (no idio ridge fit), so precompute
the shared feature series (volz, z10, and their IC statistics) ONCE, then sweep gains/windows for
(comparatively) free. Idio book held fixed at SAFE.py's shipped positions throughout (not part of
this sweep -- already separately validated as near-optimal earlier tonight).
"""
import numpy as np, pandas as pd
import SAFE, SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)


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


print("building idio-only position matrix (SAFE.py, unchanged, reused for every combo) ...")
idio_only = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_only[:, k] = np.clip(p, -lim, lim).astype(int)
print("done")

lpA = logp[0]; r = np.diff(lpA); T = len(lpA)
vol = np.full(T, np.nan); vol[M.VOL_WIN:] = M._roll_std(r, M.VOL_WIN)
volz = np.full(T, np.nan)
for s in range(M.VOL_WIN + M.VOL_Z, T):
    wv = vol[s - M.VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]
mom = np.full(T, np.nan); mom[M.MOM_LB:] = lpA[M.MOM_LB:] - lpA[:-M.MOM_LB]
z10 = np.full(T, np.nan)
for s in range(M.MOM_LB + M.VOL_Z, T):
    wm = mom[s - M.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)


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


print("precomputing per-day IC statistics for a few IC_FAST / IC_EW_HL configs ...")
CONFIGS_IC = [(90, (20, 45)), (60, (20, 45)), (120, (20, 45)), (90, (15, 30)), (90, (30, 60)), (150, (30, 60))]
SIDE_CACHE = {}
for IC_FAST, IC_EW_HL in CONFIGS_IC:
    sig = np.full(T, np.nan); msig = np.full(T, np.nan)
    for k in range(130, nt):
        fhv = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else np.nan
        fhm = np.clip(z10[k], -3, 3) / 3.0 if not np.isnan(z10[k]) else np.nan
        for feat, fh, W, out in ((volz, fhv, 200, "sig"), (z10, fhm, 200, "msig")):
            icf = _ic(feat, k, IC_FAST)
            if icf is None or np.isnan(fh):
                val = None
            else:
                sf = 1.0 if icf >= 0 else -1.0
                ics = [_ic_ew(feat, k, hl, W) for hl in IC_EW_HL]
                if any(x is None for x in ics):
                    val = sf * fh
                else:
                    ice = float(np.mean(ics))
                    val = (sf * fh) if (ice >= 0) == (icf >= 0) else 0.0
            if out == "sig": sig[k] = val if val is not None else np.nan
            else: msig[k] = val if val is not None else np.nan
    SIDE_CACHE[(IC_FAST, IC_EW_HL)] = (sig.copy(), msig.copy())
print("done")

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_and_score(sig, msig, SWITCH_GAIN, COMBINE_GAIN):
    POS = idio_only.copy()
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        s = sig[k]; m = msig[k]
        if np.isnan(s): av = 0.0
        elif np.isnan(m): av = SWITCH_GAIN * s * 100_000.0
        else: av = COMBINE_GAIN * (s + m) * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    return wo["score"], wn["score"], float(np.mean(scs)), float(min(scs))


results = []
wo0, wn0, rm0, rf0 = build_and_score(*SIDE_CACHE[(90, (20, 45))], 2.5, 3.0)
print(f"\nshipped (IC_FAST=90, HL=(20,45), SWITCH=2.5, COMBINE=3.0): OLD {wo0:.1f} NEW {wn0:.1f} rmean {rm0:.1f} rfloor {rf0:.1f}\n")

print(f"{'IC_FAST':>8}{'IC_EW_HL':>12}{'SWITCH':>8}{'COMBINE':>9}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}")
for IC_FAST, IC_EW_HL in CONFIGS_IC:
    sig, msig = SIDE_CACHE[(IC_FAST, IC_EW_HL)]
    for SWITCH_GAIN in (1.5, 2.0, 2.5, 3.0, 3.5):
        for COMBINE_GAIN in (2.0, 2.5, 3.0, 3.5, 4.0):
            wo, wn, rm, rf = build_and_score(sig, msig, SWITCH_GAIN, COMBINE_GAIN)
            results.append(((IC_FAST, IC_EW_HL, SWITCH_GAIN, COMBINE_GAIN), wo, wn, rm, rf))

results.sort(key=lambda x: -x[3])   # sort by rolling mean, descending
print("\ntop 15 by rolling mean:")
for cfg, wo, wn, rm, rf in results[:15]:
    print(f"{cfg[0]:>8}{str(cfg[1]):>12}{cfg[2]:>8}{cfg[3]:>9}{wo:>8.1f}{wn:>8.1f}{rm:>8.1f}{rf:>9.1f}")

results.sort(key=lambda x: -x[4])   # sort by rolling floor, descending
print("\ntop 15 by rolling floor:")
for cfg, wo, wn, rm, rf in results[:15]:
    print(f"{cfg[0]:>8}{str(cfg[1]):>12}{cfg[2]:>8}{cfg[3]:>9}{wo:>8.1f}{wn:>8.1f}{rm:>8.1f}{rf:>9.1f}")
