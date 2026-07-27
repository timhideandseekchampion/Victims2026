"""Full validation of the vol-adaptive MOM_LB finding (short=7, long=12) against the ACTUAL
getMyPosition-equivalent pathway, not just the backtest approximation. Builds a complete,
self-contained modified ALGO leg (mirroring SAFE_llvol.py exactly except for the adaptive MOM_LB)
and scores it with the exact same window() convention validated against eval_llboost.py all
session.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE, SAFE_llvol, SAFE_llboost

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


VOL_WIN, VOL_Z, IC_FAST, SWITCH_GAIN = 20, 60, 90, 2.5
IC_EW_HL, IC_EW_W, COMBINE_GAIN = (20, 45), 200, 3.5
SHORT_LB, LONG_LB = 7, 12


def algo_vol_shares_adaptive(lpA, cur0, cap_dol):
    """EXACT mirror of SAFE_llvol._algo_vol_shares, except MOM_LB switches between SHORT_LB
    (elevated vol) and LONG_LB (calm vol) based on today's volz sign."""
    T_ = len(lpA)
    if T_ < VOL_WIN + VOL_Z + 60:
        return 0
    r_ = np.diff(lpA)
    vol = np.full(T_, np.nan); vol[VOL_WIN:] = roll_std(r_, VOL_WIN)
    tnow = T_ - 1
    lo = max(VOL_WIN + VOL_Z, tnow - 250)
    volz = np.full(T_, np.nan)
    for s in range(lo, T_):
        wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T_, np.nan); ret1[:T_ - 1] = lpA[1:] - lpA[:-1]

    def _ic(feat, L):
        a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-12: return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _ic_ew(feat, HL, W):
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

    def _side(feat, fhv):
        icf = _ic(feat, IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, IC_EW_W) for hl in IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return 0
    sig = _side(volz, fh)
    if sig is None: return 0

    mom_lb = SHORT_LB if fh > 0 else LONG_LB
    mom = np.full(T_, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T_, np.nan)
    for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T_):
        wm = mom[s - VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        av = COMBINE_GAIN * (sig + msig) * 100_000.0
    else:
        av = SWITCH_GAIN * sig * 100_000.0
    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


def getMyPosition_adaptive(prcSoFar):
    """Full strategy: SAFE_llboost's idio ridge+significance-boost (unchanged, calls the actual
    module), plus the adaptive-MOM_LB ALGO leg."""
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nInst_, t = prcSoFar.shape
    full = np.asarray(SAFE_llboost.getMyPosition(prcSoFar))
    pos = full.copy()
    dlr_ = SAFE_llboost._limits(nInst_)
    cur = prcSoFar[:, -1]
    logp_ = np.log(prcSoFar)
    pos[0] = algo_vol_shares_adaptive(logp_[0], cur[0], dlr_[0])
    lim = (dlr_ / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)

print("computing REAL SAFE_llboost positions (baseline, via actual getMyPosition) ...")
t0 = time.time()
POS_base = np.zeros((nInst, nt))
for k in range(SAFE.WARMUP, nt):
    POS_base[:, k] = SAFE_llboost.getMyPosition(P_[:, :k + 1])
print(f"  done ({time.time()-t0:.0f}s)")

print("computing adaptive-MOM_LB positions (via getMyPosition_adaptive, real idio+boost + adaptive ALGO) ...")
t0 = time.time()
POS_adapt = np.zeros((nInst, nt))
for k in range(SAFE.WARMUP, nt):
    POS_adapt[:, k] = getMyPosition_adaptive(P_[:, :k + 1])
print(f"  done ({time.time()-t0:.0f}s)")


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<30}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


print("\n=== FULL VALIDATION via real getMyPosition pathway ===")
base_scs = report("real SAFE_llboost (baseline)", POS_base)
report("real + adaptive MOM_LB (short=7,long=12)", POS_adapt, base_scs)

print(f"\nsanity: real SAFE_llboost NEW should equal 828.6/822.16-ish official score: "
      f"{window(POS_base, *NEW):.2f}")
