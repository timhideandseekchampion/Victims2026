"""Test whether MOM_LB should be adaptive/ensembled rather than a fixed constant. Two variants:
(1) ENSEMBLE: blend momentum z-scores from several different lookbacks (7,10,13 etc), analogous to
    how the ridge already ensembles multiple HALF_LIVES -- a well-motivated precedent since that
    ensemble genuinely helps for the ridge.
(2) VOL-REGIME-ADAPTIVE: switch between a SHORT and LONG lookback based on the CURRENT vol regime
    (causal, using the already-computed volz), on the theory that momentum decays faster in high-vol
    periods and persists longer in calm periods.
Reuses the fixed significance-boost + ridge WZ precompute; only the ALGO leg's momentum
construction varies.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE, SAFE_llvol

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]

BOOST_MIN_DAY = 500
ALPHA = 0.05
N_CANDIDATES = 49
BOOST_P = 2.0
BOOST_SCALE_W = 1000
BOOST_IC_L = 190
BOOST_K = 1.5


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


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== precompute (fixed idio side): shipped WZ + significance-gated boost ===")
t0 = time.time()
WZ = {}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    WZ[t] = wz
print(f"  WZ done ({time.time()-t0:.0f}s)")

n = rs.shape[0]
BOOST_AT = {}
for k in range(BOOST_MIN_DAY, nt):
    T = k
    Xi = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr:
            continue
        lead = rs[i, :T]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        entry[j] = lead_boost[-1]
    BOOST_AT[k] = entry
print("  significance-boost map done")

idio_pos = np.zeros((nInst, nt))
for k in range(SAFE.WARMUP, nt):
    cur = P_[:, k]; lim = (dlr / cur).astype(int)
    wz = WZ[k].copy()
    if k >= BOOST_MIN_DAY:
        for j, bv in BOOST_AT[k].items():
            wz[j] += BOOST_K * bv
    idio_pos[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
print("  fixed idio book (ridge+boost) done")


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


VOL_WIN, VOL_Z, IC_FAST, SWITCH_GAIN = 20, 60, 90, 2.5
IC_EW_HL, IC_EW_W, COMBINE_GAIN = (20, 45), 200, 3.5


def algo_vol_shares_flex(lpA, cur0, cap_dol, mode, mom_lbs=None, short_lb=None, long_lb=None):
    """mode='fixed' (single MOM_LB, mom_lbs=[v]), 'ensemble' (average z10 across mom_lbs),
    'vol_adaptive' (switch between short_lb and long_lb based on current volz)."""
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

    def z10_for(mom_lb):
        mom = np.full(T_, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
        z10 = np.full(T_, np.nan)
        for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T_):
            wm = mom[s - VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
        return z10

    if mode == "fixed":
        z10 = z10_for(mom_lbs[0])
    elif mode == "ensemble":
        z10s = [z10_for(lb) for lb in mom_lbs]
        stacked = np.vstack(z10s)  # (n_lb, T_) -- blend element-wise across the FULL history
        z10 = np.nanmean(stacked, axis=0)
        all_nan_cols = np.all(np.isnan(stacked), axis=0)
        z10[all_nan_cols] = np.nan
    elif mode == "vol_adaptive":
        use_lb = short_lb if fh > 0 else long_lb  # elevated vol -> shorter window, calm -> longer
        z10 = z10_for(use_lb)

    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        av = COMBINE_GAIN * (sig + msig) * 100_000.0
    else:
        av = SWITCH_GAIN * sig * 100_000.0
    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


def build_algo(**kw):
    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(algo_vol_shares_flex(logp[0, :k + 1], cur0, dlr[0], **kw), -lim0, lim0)
    return algo_pos


def full_pos(algo_pos):
    POS = idio_pos.copy()
    POS[0, :] = algo_pos
    return POS


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<32}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


print("\n=== baseline: MOM_LB=10 fixed (current shipped) ===")
base_scs = report("fixed MOM_LB=10 (shipped)", full_pos(build_algo(mode="fixed", mom_lbs=[10])))

print("\n=== ENSEMBLE: blend multiple MOM_LB lookbacks ===")
for combo in [(7, 10, 13), (8, 10, 12), (5, 10, 15), (7, 10, 13, 16), (9, 10, 11), (10, 13)]:
    report(f"ensemble {combo}", full_pos(build_algo(mode="ensemble", mom_lbs=list(combo))), base_scs)

print("\n=== VOL-REGIME-ADAPTIVE: short lookback in elevated vol, long in calm vol ===")
for short_lb, long_lb in [(7, 13), (5, 15), (7, 16), (5, 20), (8, 12)]:
    report(f"adaptive short={short_lb} long={long_lb}",
           full_pos(build_algo(mode="vol_adaptive", short_lb=short_lb, long_lb=long_lb)), base_scs)

print("\n=== neighbor-stability check around the promising vol-adaptive region (7-8 short, 12-16 long) ===")
for short_lb in (6, 7, 8, 9):
    for long_lb in (11, 12, 13, 14, 15, 16):
        report(f"adaptive short={short_lb} long={long_lb}",
               full_pos(build_algo(mode="vol_adaptive", short_lb=short_lb, long_lb=long_lb)), base_scs)

print("\n=== window-concentration diagnostic on the best candidate (short=7, long=12) ===")
best_POS = full_pos(build_algo(mode="vol_adaptive", short_lb=7, long_lb=12))
best_scs = scs_curve(best_POS)
diffs = best_scs - base_scs
worse_idx = np.where(diffs < 0)[0]
print(f"{len(worse_idx)}/{len(base_scs)} windows worse. worst 8 windows (end_day, baseline, candidate, diff):")
order = np.argsort(diffs)[:8]
for idx in order:
    print(f"    end_day={end_days[idx]:>4}  base={base_scs[idx]:>7.1f}  cand={best_scs[idx]:>7.1f}  diff={diffs[idx]:>+7.1f}")
print(f"spread of diffs across ALL worse windows: min={diffs[worse_idx].min():.1f} max={diffs[worse_idx].max():.1f} "
      f"mean={diffs[worse_idx].mean():.1f}")
better_idx = np.where(diffs > 0)[0]
print(f"spread of diffs across ALL better windows: min={diffs[better_idx].min():.1f} max={diffs[better_idx].max():.1f} "
      f"mean={diffs[better_idx].mean():.1f}  (n={len(better_idx)})")
