"""Follow-up to test_v7cand_algoresweep.py: that script's COMBINE_GAIN sweep (2.0-5.0) found a
MONOTONIC improvement on every metric (OLD/NEW/rmean/rfloor all rising together) with n_worse
staying at 0-2/61 throughout -- and never found a peak within the tested range. This extends the
sweep much further to find where it actually plateaus/peaks, and checks whether the mechanism
behind it is a genuine (bounded) effect or an unbounded artifact.

Mechanistic hypothesis for WHY this might be genuine, not overfit: COMBINE_GAIN just scales the
raw dollar target 'av = COMBINE_GAIN * (sig + msig) * 100_000' BEFORE it gets clipped to
+/-cap_dol (100_000). Since (sig+msig) is bounded (each component in roughly [-1,1] via fhv/fhm
clipping), raising COMBINE_GAIN just lowers the |sig+msig| threshold needed to hit the dollar cap
-- i.e. it pushes the ALGO leg from "magnitude-weighted sizing" towards "full-conviction sign-based
sizing whenever there's ANY nonzero signal", the SAME principle already validated everywhere else
in this repo (idio book sizing, batch80 catC: "full-conviction sign-based sizing keeps winning").
If that's really what's happening, the sweep MUST plateau exactly once every (sig+msig)!=0 day is
already saturated to the cap -- further increases would be a mathematical no-op, not something
that can be "overfit" past that point. This script confirms that story explicitly (fraction of days
saturated to +/-cap vs COMBINE_GAIN) rather than just trusting the score numbers.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE
import SAFE_llboost_v6 as V6

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
n = rs.shape[0]

BOOST_MIN_DAY = V6.BOOST_MIN_DAY
ALPHA = V6.BOOST_ALPHA
N_CANDIDATES = V6.BOOST_N_CANDIDATES
BOOST_P = V6.BOOST_P
BOOST_SCALE_W = V6.BOOST_SCALE_W
BOOST_IC_L = V6.BOOST_IC_L
BOOST_K = V6.BOOST_K
assert (BOOST_MIN_DAY, N_CANDIDATES, BOOST_IC_L, BOOST_SCALE_W, BOOST_K, BOOST_P) == \
       (480, 39, 250, 1000, 1.5, 2.0), "true v6 boost params drifted"


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


print("=== precompute (fixed idio side, ONCE): TRUE v6 ridge WZ + N=39/IC_L=250/MIN_DAY=480 boost ===")
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

t0 = time.time()
BOOST_AT = {}
for k in range(BOOST_MIN_DAY, nt):
    T = k
    Xi_full = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)): continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr: continue
        i = cand_idx[ci]
        lead = rs[i, :T]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0: continue
        entry[j] = lead_boost[-1]
    BOOST_AT[k] = entry
print(f"  boost map done ({time.time()-t0:.0f}s)")

idio_pos = np.zeros((nInst, nt))
for k in range(SAFE.WARMUP, nt):
    cur = P_[:, k]; lim = (dlr / cur).astype(int)
    wz = WZ[k].copy()
    if k >= BOOST_MIN_DAY:
        for j, bv in BOOST_AT[k].items():
            wz[j] += BOOST_K * bv
    idio_pos[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
print("  fixed TRUE-v6 idio book done -- frozen for every run below")


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


def algo_vol_shares_v6(lpA, cur0, cap_dol, COMBINE_GAIN, VOL_WIN=20, VOL_Z=60, IC_FAST=90,
                        SWITCH_GAIN=2.5, IC_EW_HL=(20, 45), MOM_LB_SHORT=7, MOM_LB_LONG=12,
                        IC_EW_W=200, track_saturation=None):
    T = len(lpA)
    if T < VOL_WIN + VOL_Z + 60:
        return 0
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan); vol[VOL_WIN:] = roll_std(r_, VOL_WIN)
    tnow = T - 1
    lo = max(VOL_WIN + VOL_Z, tnow - 250)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]

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
    mom_lb = MOM_LB_SHORT if fh > 0 else MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T):
        wm = mom[s - VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        av_raw = COMBINE_GAIN * (sig + msig) * 100_000.0
    else:
        av_raw = SWITCH_GAIN * sig * 100_000.0
    av = float(np.clip(av_raw, -cap_dol, cap_dol))
    if track_saturation is not None and msig is not None:
        track_saturation.append(abs(av_raw) >= cap_dol)
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


def build_algo(COMBINE_GAIN, track=None):
    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(
            algo_vol_shares_v6(logp[0, :k + 1], cur0, dlr[0], COMBINE_GAIN, track_saturation=track),
            -lim0, lim0)
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
    line = f"{nm:<20}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


print("\n=== sanity: real SAFE_llboost_v6.getMyPosition (COMBINE_GAIN=3.5) ===")
FIRST_DAY = 148
t0 = time.time()
POS_SHIPPED = np.zeros((nInst, nt))
for k in range(FIRST_DAY, nt):
    POS_SHIPPED[:, k] = V6.getMyPosition(P_[:, :k + 1])
print(f"  done ({time.time()-t0:.0f}s)")
base_scs = report("shipped v6 (real)", POS_SHIPPED)

print("\n=== extended COMBINE_GAIN sweep (2.0 -> 30.0), tracking cap-saturation fraction ===")
results = {}
for g in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0):
    track = []
    scs = report(f"COMBINE_GAIN={g}", full_pos(build_algo(g, track=track)), base_scs)
    sat_frac = float(np.mean(track)) if track else float("nan")
    nz_frac = float(np.mean([t or True for t in track])) if track else float("nan")
    print(f"    -> fraction of msig!=None days saturated to +/-cap: {sat_frac:.3f}  (n={len(track)})")
    results[g] = (scs.mean(), scs.min(), sat_frac)

print("\n=== summary: does it plateau? ===")
gs = sorted(results)
for i, g in enumerate(gs):
    rmean, rfloor, sat = results[g]
    delta = "" if i == 0 else f"  (d_rmean={rmean - results[gs[i-1]][0]:+.2f})"
    print(f"  G={g:<6} rmean={rmean:7.1f}  rfloor={rfloor:7.1f}  sat_frac={sat:.3f}{delta}")

print("\n=== neighbor-stability check around the apparent best region ===")
best_g = max(gs, key=lambda g: results[g][0])
print(f"best rmean at G={best_g} (rmean={results[best_g][0]:.1f}); checking +/-1 and +/-2 neighbors already "
      f"in the grid above for a plateau (not an isolated spike).")
