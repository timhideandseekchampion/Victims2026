"""Continuing the score-maximization pass: VOL_WIN/VOL_Z (feature construction for the vol signal)
have never been jointly swept against each other or against the gain parameters. Cheap: no ridge
needed. Idio book fixed (SAFE.py, unchanged). Uses COMBINE_GAIN=3.5 (the joint-sweep winner) and the
shipped IC config (IC_FAST=90, IC_EW_HL=(20,45)).
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


print("building idio-only position matrix (reused for every combo) ...")
idio_only = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_only[:, k] = np.clip(p, -lim, lim).astype(int)
print("done")

lpA = logp[0]; r = np.diff(lpA); T = len(lpA)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]
MOM_LB = M.MOM_LB
mom = np.full(T, np.nan); mom[MOM_LB:] = lpA[MOM_LB:] - lpA[:-MOM_LB]


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


def side_series(feat, IC_FAST=90, IC_EW_HL=(20, 45), W=200):
    sig = np.full(T, np.nan)
    for k in range(130, nt):
        fh = np.clip(feat[k], -3, 3) / 3.0 if not np.isnan(feat[k]) else np.nan
        if np.isnan(fh): continue
        icf = _ic(feat, k, IC_FAST)
        if icf is None: continue
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, k, hl, W) for hl in IC_EW_HL]
        if any(x is None for x in ics):
            sig[k] = sf * fh
        else:
            ice = float(np.mean(ics))
            sig[k] = (sf * fh) if (ice >= 0) == (icf >= 0) else 0.0
    return sig


z10 = np.full(T, np.nan)
for s in range(MOM_LB + 60, T):
    wm = mom[s - 60:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
msig = side_series(z10)

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))
COMBINE_GAIN = 3.5


def build_and_score(sig):
    POS = idio_only.copy()
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        s = sig[k]; m = msig[k]
        if np.isnan(s): av = 0.0
        elif np.isnan(m): av = 2.5 * s * 100_000.0
        else: av = COMBINE_GAIN * (s + m) * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    return wo["score"], wn["score"], float(np.mean(scs)), float(min(scs))


print(f"{'VOL_WIN':>8}{'VOL_Z':>7}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}")
results = []
for VOL_WIN in (10, 15, 20, 25, 30):
    for VOL_Z in (40, 50, 60, 75, 90, 120):
        vol = np.full(T, np.nan); vol[VOL_WIN:] = M._roll_std(r, VOL_WIN)
        volz = np.full(T, np.nan)
        for s in range(VOL_WIN + VOL_Z, T):
            wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
        sig = side_series(volz)
        wo, wn, rm, rf = build_and_score(sig)
        results.append(((VOL_WIN, VOL_Z), wo, wn, rm, rf))
        mark = "  <-- shipped" if (VOL_WIN, VOL_Z) == (20, 60) else ""
        print(f"{VOL_WIN:>8}{VOL_Z:>7}{wo:>8.1f}{wn:>8.1f}{rm:>8.1f}{rf:>9.1f}{mark}")

results.sort(key=lambda x: -x[3])
print("\ntop 10 by rolling mean:")
for cfg, wo, wn, rm, rf in results[:10]:
    print(f"  VOL_WIN={cfg[0]:<4} VOL_Z={cfg[1]:<5} OLD {wo:.1f} NEW {wn:.1f} rmean {rm:.1f} rfloor {rf:.1f}")
