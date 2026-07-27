"""Better gate: instead of requiring slow-IC and fast-EW-IC to merely AGREE IN SIGN (which lets a
truly-zero-IC candidate through ~83% of the time, since two noisy near-zero estimates coincidentally
share a sign far more often than intuition suggests), require the slow IC to be STATISTICALLY
SIGNIFICANT (a proper t-test, not just nonzero-signed) before the signal activates at all. This
should suppress the reversion candidate (true IC ~0 here) down toward its nominal false-positive
rate, while (hopefully) leaving the real vol/momentum signals largely intact.
"""
import numpy as np, pandas as pd
from scipy import stats
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
REV_K = 5
revmove = np.full(T, np.nan); revmove[REV_K:] = lpA[REV_K:] - lpA[:-REV_K]
zrev = np.full(T, np.nan)
for s in range(REV_K + M.VOL_Z, T):
    wv = revmove[s - M.VOL_Z:s]; zrev[s] = (revmove[s] - wv.mean()) / (wv.std() + 1e-12)
zrev_faded = -zrev


def _ic_n(feat, tnow, L):
    a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    n = int(ok.sum())
    if n < 60: return None, 0
    xs, ys = xs[ok], ys[ok]
    if xs.std() < 1e-12: return None, 0
    return float(np.corrcoef(xs, ys)[0, 1]), n


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


def is_sig(ic, n, alpha):
    if ic is None or n < 10: return False
    t = ic * np.sqrt(n - 2) / np.sqrt(max(1 - ic ** 2, 1e-12))
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    return abs(t) > tcrit


def _side_sig(feat, tnow, fhv, alpha):
    icf, n = _ic_n(feat, tnow, M.IC_FAST)
    if not is_sig(icf, n, alpha): return None
    sf = 1.0 if icf >= 0 else -1.0
    ics = [_ic_ew(feat, tnow, hl, M.IC_EW_W) for hl in M.IC_EW_HL]
    if any(x is None for x in ics): return sf * fhv
    ice = float(np.mean(ics))
    return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos(use_reversion, gain, alpha):
    POS = idio_pos.copy()
    n_rev_active = 0; n_vol_active = 0; n_mom_active = 0; n_days = 0
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        fhv = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else np.nan
        fhm = np.clip(z10[k], -3, 3) / 3.0 if not np.isnan(z10[k]) else np.nan
        fhr = np.clip(zrev_faded[k], -3, 3) / 3.0 if not np.isnan(zrev_faded[k]) else np.nan
        sig = _side_sig(volz, k, fhv, alpha) if not np.isnan(fhv) else None
        msig = _side_sig(z10, k, fhm, alpha) if not np.isnan(fhm) else None
        rsig = _side_sig(zrev_faded, k, fhr, alpha) if (use_reversion and not np.isnan(fhr)) else None
        if sig is not None and sig != 0.0: n_vol_active += 1
        if msig is not None and msig != 0.0: n_mom_active += 1
        if rsig is not None and rsig != 0.0: n_rev_active += 1
        n_days += 1
        parts = [x for x in (sig, msig, rsig) if x is not None]
        if not parts:
            av = 0.0
        elif len(parts) == 1:
            av = M.SWITCH_GAIN * parts[0] * 100_000.0
        else:
            av = gain * sum(parts) * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    return POS, n_rev_active, n_vol_active, n_mom_active, n_days


def report(nm, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])
    print(f"{nm:<38}OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={scs.mean():>8.1f}  rfloor={scs.min():>8.1f}")
    return scs


print(f"\n--- shipped vol+mom (no significance gate, no reversion) ---")
POS_ship, *_ = build_pos(False, M.COMBINE_GAIN, alpha=1.0)  # alpha=1.0 disables the sig filter (always "significant")
base_scs = report("shipped (vol+mom, plain agree-gate)", POS_ship)

print("\n--- vol+mom WITH significance gate (no reversion yet) -- check it doesn't break the real signals ---")
for alpha in (0.5, 0.3, 0.2, 0.1, 0.05):
    POS, _, n_vol, n_mom, n_days = build_pos(False, M.COMBINE_GAIN, alpha)
    scs = report(f"vol+mom, sig-gated alpha={alpha}", POS)
    print(f"    vol active {n_vol}/{n_days} ({100*n_vol/n_days:.0f}%)  mom active {n_mom}/{n_days} ({100*n_mom/n_days:.0f}%)")

print("\n--- NOW add reversion as a third candidate, WITH the significance gate ---")
for alpha in (0.5, 0.3, 0.2, 0.1, 0.05):
    POS, n_rev, n_vol, n_mom, n_days = build_pos(True, M.COMBINE_GAIN, alpha)
    scs = report(f"vol+mom+rev, sig-gated alpha={alpha}", POS)
    nworse = int((scs < base_scs).sum())
    print(f"    rev active {n_rev}/{n_days} ({100*n_rev/n_days:.0f}%)  n_worse_vs_shipped={nworse}/{len(scs)}")
