"""Test: the ALGO leg (LLVOL) has a much lower Sharpe (~1.7-2.9 annualized) than the idio book
(~6) - so unlike the idio book (where the score formula mu*SR^2/(SR^2+1) is already saturated and
any variance-reduction trick just gives up mean for nothing), the ALGO leg is NOT saturated: cutting
its day-to-day PnL variance without giving up much mean should raise its score contribution.

Hypothesis: the leg's sizing ('av', the raw dollar target) is recomputed fresh every day from
today's clipped vol z-score + switch sign - it has no smoothing at all, so it can swing sharply
day-to-day even when the underlying regime hasn't changed. Test an EMA smooth on that raw dollar
target (post-signal, pre-clip) at a few half-lives vs the shipped no-smoothing baseline, using the
exact eval-mirroring accounting. Idio leg is untouched (identical every variant).
"""
import numpy as np, pandas as pd
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
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


def algo_raw_dollar(lpA, tnow):
    """Reproduce SAFE_llvol._algo_vol_shares up to the raw dollar target 'av' (pre-clip, pre-round)."""
    T = tnow + 1
    if T < M.VOL_WIN + M.VOL_Z + 60: return None
    r = np.diff(lpA[:T])
    vol = np.full(T, np.nan)
    vol[M.VOL_WIN:] = M._roll_std(r, M.VOL_WIN)
    lo = max(M.VOL_WIN + M.VOL_Z, tnow - M.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - M.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:T][:] - lpA[:T - 1]

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
        icf = _ic(feat, M.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        if not M.IC_BLEND: return sf * fhv
        ics = [_ic_ew(feat, hl, M.IC_EW_W) for hl in M.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return 0.0
    sig = _side(volz, fh)
    if sig is None: return 0.0
    mom = np.full(T, np.nan); mom[M.MOM_LB:] = lpA[:T][M.MOM_LB:] - lpA[:T][:-M.MOM_LB]
    z10 = np.full(T, np.nan)
    for s in range(max(M.MOM_LB + M.VOL_Z, tnow - M.IC_EW_W), T):
        wm = mom[s - M.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        return M.COMBINE_GAIN * (sig + msig) * 100_000.0
    return M.SWITCH_GAIN * sig * 100_000.0


print("computing raw daily $ target series (unsmoothed) ...")
RAW = np.full(nt, np.nan)
for k in range(130, nt):
    v = algo_raw_dollar(logp[0], k)
    RAW[k] = 0.0 if v is None else v
print("done")


def ema_smooth(x, hl):
    if hl <= 1: return x.copy()
    lam = 0.5 ** (1.0 / hl)
    out = np.full_like(x, np.nan)
    acc = 0.0; wsum = 0.0
    for i, v in enumerate(x):
        if np.isnan(v): out[i] = np.nan; continue
        acc = lam * acc + v; wsum = lam * wsum + 1.0
        out[i] = acc / wsum
    return out


def build_pos(smooth_hl):
    smoothed = ema_smooth(np.nan_to_num(RAW), smooth_hl)
    POS = np.zeros((nInst, nt))
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        av = float(np.clip(smoothed[k], -dlr[0], dlr[0]))
        algo_shares = int(np.clip(av / cur[0], -lim[0], lim[0]))
        # idio leg: shipped SAFE_llvol positions, unchanged
        full = np.asarray(M.getMyPosition(P[:, :k + 1]))
        pos = full.copy(); pos[0] = algo_shares
        POS[:, k] = np.clip(pos, -lim, lim).astype(int)
    return POS


OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

print(f"\n{'smooth_hl':>10} {'OLD':>8} {'NEW':>8} {'roll_mean':>10} {'roll_floor':>11}")
for hl in [1, 2, 3, 5, 8, 13, 21]:
    POS = build_pos(hl)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{hl:>10} {wo['score']:>8.1f} {wn['score']:>8.1f} {np.mean(scs):>10.1f} {min(scs):>11.1f}")
