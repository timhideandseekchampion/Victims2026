"""Cross-draw robustness hypothesis: SAFE_llvol commits to ONE ALGO mechanism (vol-continuation)
that happens to validate on THIS draw. A sibling draw from plausibly the same generator had a
different ALGO dynamic (index reversion, confirmed absent here). Test adding 5-day z-score reversion
as a THIRD candidate signal through the SAME proven adaptive-combine architecture (each signal only
contributes when its OWN trailing IC currently confirms it) already used for vol+momentum. On THIS
draw it should cost ~nothing (the gate should suppress it, since we've shown reversion isn't real
here) -- the real test is whether it's free insurance, not whether it helps this specific file.
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


print("building idio book (shipped, unchanged) ...")
idio_pos = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_pos[:, k] = np.clip(p, -lim, lim).astype(int)
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

# reversion candidate: 5-day cumulative return, z-scored, FADED (this is the sibling draw's mechanism)
REV_K = 5
revmove = np.full(T, np.nan); revmove[REV_K:] = lpA[REV_K:] - lpA[:-REV_K]
zrev = np.full(T, np.nan)
for s in range(REV_K + M.VOL_Z, T):
    wv = revmove[s - M.VOL_Z:s]; zrev[s] = (revmove[s] - wv.mean()) / (wv.std() + 1e-12)
zrev_faded = -zrev   # fade: bet AGAINST the recent move (reversion), not with it


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


def build_pos(use_reversion, gain):
    POS = idio_pos.copy()
    n_rev_active = 0; n_days = 0
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        fhv = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else np.nan
        fhm = np.clip(z10[k], -3, 3) / 3.0 if not np.isnan(z10[k]) else np.nan
        fhr = np.clip(zrev_faded[k], -3, 3) / 3.0 if not np.isnan(zrev_faded[k]) else np.nan
        sig = _side(volz, k, fhv) if not np.isnan(fhv) else None
        msig = _side(z10, k, fhm) if not np.isnan(fhm) else None
        rsig = _side(zrev_faded, k, fhr) if (use_reversion and not np.isnan(fhr)) else None
        parts = [x for x in (sig, msig, rsig) if x is not None]
        if rsig is not None and rsig != 0.0:
            n_rev_active += 1
        n_days += 1
        if not parts:
            av = 0.0
        elif len(parts) == 1:
            av = M.SWITCH_GAIN * parts[0] * 100_000.0
        else:
            av = gain * sum(parts) * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    return POS, n_rev_active, n_days


print(f"\n{'config':<32}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}{'rev_active_days':>17}")
POS0, _, _ = build_pos(False, M.COMBINE_GAIN)
wo = window(POS0, *OLD); wn = window(POS0, *NEW)
scs = [window(POS0, E - NUMTEST, E)["score"] for E in end_days]
print(f"{'shipped (vol+mom only)':<32}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>8.1f}{min(scs):>9.1f}")

for gain in (M.COMBINE_GAIN, M.COMBINE_GAIN * 2 / 3):
    POS, n_active, n_days = build_pos(True, gain)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{'vol+mom+revK5 gain='+str(round(gain,2)):<32}{wo['score']:>8.1f}{wn['score']:>8.1f}"
          f"{np.mean(scs):>8.1f}{min(scs):>9.1f}{n_active:>17}/{n_days}")

print("\n--- full distribution check: how many windows does adding reversion actually help vs hurt? ---")
POS0, _, _ = build_pos(False, M.COMBINE_GAIN)
POS1, n_active, n_days = build_pos(True, M.COMBINE_GAIN)
scs0 = np.array([window(POS0, E - NUMTEST, E)["score"] for E in end_days])
scs1 = np.array([window(POS1, E - NUMTEST, E)["score"] for E in end_days])
n_worse = int((scs1 < scs0).sum()); n_better = int((scs1 > scs0).sum())
print(f"windows where adding reversion is WORSE: {n_worse}/{len(scs0)}")
print(f"windows where adding reversion is BETTER: {n_better}/{len(scs0)}")
print(f"biggest single-window improvement: {(scs1-scs0).max():.1f} at end_day {end_days[int(np.argmax(scs1-scs0))]}")
print(f"biggest single-window damage:      {(scs1-scs0).min():.1f} at end_day {end_days[int(np.argmin(scs1-scs0))]}")
