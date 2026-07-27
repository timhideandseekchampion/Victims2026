"""Nested extension of the ALREADY-VALIDATED adaptive-combine architecture: SAFE_llvol's ALGO leg
combines vol + momentum by giving each its own _side() decision (slow-IC direction, gated by
fast-EW-IC agreement) and summing whatever survives. Add lead-lag net-$ skew as a THIRD input to
THIS SAME mechanism (not a naive sum, not a mechanism-switch -- both already tried and rejected) so
it only contributes when ITS OWN trailing evidence currently supports it, exactly like the other two.
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


print("building idio book (shipped SAFE.py, unchanged) and its net-$ skew each day ...")
idio_pos = np.zeros((nInst, nt)); netdol = np.full(nt, np.nan)
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_pos[:, k] = np.clip(p, -lim, lim).astype(int)
    netdol[k] = float((idio_pos[1:, k] * cur[1:]).sum())
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
netz = np.clip(netdol / 100_000.0, -3, 3)   # net-$ skew scaled to roughly the same [-3,3]-ish range as volz/z10


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


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos(use_netdol, GAIN):
    POS = idio_pos.copy()
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        fhv = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else np.nan
        fhm = np.clip(z10[k], -3, 3) / 3.0 if not np.isnan(z10[k]) else np.nan
        fhn = np.clip(netz[k], -3, 3) / 3.0 if not np.isnan(netz[k]) else np.nan
        sig = _side(volz, k, fhv) if not np.isnan(fhv) else None
        msig = _side(z10, k, fhm) if not np.isnan(fhm) else None
        nsig = _side(netz, k, fhn) if (use_netdol and not np.isnan(fhn)) else None
        parts = [x for x in (sig, msig, nsig) if x is not None]
        if not parts:
            av = 0.0
        elif len(parts) == 1:
            av = M.SWITCH_GAIN * parts[0] * 100_000.0
        else:
            av = GAIN * sum(parts) * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    return POS


print(f"\n{'config':<30}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}")
POS0 = build_pos(False, M.COMBINE_GAIN)
wo = window(POS0, *OLD); wn = window(POS0, *NEW)
scs = [window(POS0, E - NUMTEST, E)["score"] for E in end_days]
print(f"{'shipped (vol+mom only)':<30}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>8.1f}{min(scs):>9.1f}")

for GAIN in (1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
    POS = build_pos(True, GAIN)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{'vol+mom+netdol GAIN='+str(GAIN):<30}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>8.1f}{min(scs):>9.1f}")
