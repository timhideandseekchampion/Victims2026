"""
test_batch100_E67.py

E67: Test a CONTINUOUS blend across many momentum lookbacks (5,7,10,12,15,20) for ALGO's momentum
"combine" sub-signal (msig), replacing the binary MOM_LB_SHORT(7)/MOM_LB_LONG(12) switch -- which
v10 picks by the SIGN of the vol-regime feature fh -- with an unconditional AVERAGE of the
momentum-switch sub-signal computed independently at each of the 6 lookbacks (each still IC-gated
via the same _side() logic as v10). Everything else in _algo_vol_shares (vol-z regime detection,
the IC-gated vol-regime side signal `sig`, COMBINE_GAIN, SWITCH_GAIN fallback, the HOLD deadband)
is byte-identical to v10. One reasonable configuration tested (the exact lookback set given), no
sweep -- the "binary switch vs continuous blend" question doesn't need a parameter grid.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT

VOL_WIN, VOL_Z, IC_LOOKBACK, IC_FAST = V10.VOL_WIN, V10.VOL_Z, V10.IC_LOOKBACK, V10.IC_FAST
IC_BLEND, IC_EW_HL, IC_EW_W = V10.IC_BLEND, V10.IC_EW_HL, V10.IC_EW_W
MOM_LB_SHORT, MOM_LB_LONG, COMBINE_GAIN = V10.MOM_LB_SHORT, V10.MOM_LB_LONG, V10.COMBINE_GAIN
SWITCH_GAIN = V10.SWITCH_GAIN
DEADBAND_THRESH_FRAC, DEADBAND_MIN_DAY = V10.DEADBAND_THRESH_FRAC, V10.DEADBAND_MIN_DAY


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

print("=== precompute: idio WZ (verbatim v10) -- unaffected by the ALGO momentum-blend swap ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

WZ_V10 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    if t >= max(RS_SHORT_W, RS_LONG_W) + 5:
        short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
        long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
        sz = short_ret - short_ret.mean(); sstd = sz.std()
        lz = long_ret - long_ret.mean(); lstd = lz.std()
        if sstd > 1e-12 and lstd > 1e-12:
            sz = sz / sstd; lz = lz / lstd
            disagree = np.sign(lz) != np.sign(sz)
            rs_sig = np.where(disagree, -sz, 0.0)
            s_std = rs_sig.std()
            s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_idio_pos():
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    return POS


IDIO_POS = build_idio_pos()


def algo_baseline():
    algo = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
    return algo


ALGO_BASE = algo_baseline()
POS_base = IDIO_POS.copy(); POS_base[0, :] = ALGO_BASE
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"\n=== sanity check: verbatim reconstruction must reproduce SAFE_llboost_v10 exactly ===")
print(f"  OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  rfloor={base_scs.min():.1f}"
      f"   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: does NOT reproduce v10 -- do not trust results below. ***")
if not SANITY_OK:
    raise SystemExit("Sanity check failed -- aborting.")

# ============================================================================================
# ALGO leg with a continuous multi-lookback momentum blend replacing the binary short/long switch
# ============================================================================================


def _ic(feat, ret1, tnow, L):
    a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60: return None
    xs, ys = xs[ok], ys[ok]
    if xs.std() < 1e-12: return None
    return float(np.corrcoef(xs, ys)[0, 1])


def _ic_ew(feat, ret1, tnow, HL, W):
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


def _side(feat, ret1, tnow, fhv):
    icf = _ic(feat, ret1, tnow, IC_FAST)
    if icf is None: return None
    sf = 1.0 if icf >= 0 else -1.0
    if not IC_BLEND: return sf * fhv
    ics = [x for x in (_ic_ew(feat, ret1, tnow, hl, IC_EW_W) for hl in IC_EW_HL) if x is not None]
    if len(ics) < len(IC_EW_HL): return sf * fhv
    ice = float(np.mean(ics))
    return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0


def _mom_z(lpA, T, tnow, mom_lb):
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z = np.full(T, np.nan)
    for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T):
        wm = mom[s - VOL_Z:s]; z[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    return z


def _finalize(av, cur0, cap_dol, lim, state, tnow):
    have_prev = (tnow == state['t'] + 1)
    av = float(np.clip(av, -cap_dol, cap_dol))
    if have_prev and tnow >= DEADBAND_MIN_DAY and abs(av) < DEADBAND_THRESH_FRAC * cap_dol:
        shares = int(np.clip(state['shares'], -lim, lim))
    else:
        shares = int(np.clip(av / cur0, -lim, lim))
    state['shares'] = shares; state['t'] = tnow
    return shares


def algo_momblend(LBS):
    state = {'shares': 0, 't': -1}
    out = np.zeros(nt)
    lpA_full = logp[0]
    ret1_full = np.full(nt, np.nan); ret1_full[:nt - 1] = lpA_full[1:] - lpA_full[:-1]
    for k in range(130, nt):
        T = k + 1; tnow = T - 1
        cur0 = P_[0, k]; cap_dol = dlr[0]; lim = int(cap_dol / cur0)
        if T < VOL_WIN + VOL_Z + 60:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        lpA = lpA_full[:T]
        rr = np.diff(lpA)
        vol = np.full(T, np.nan)
        vol[VOL_WIN:] = V10._roll_std(rr, VOL_WIN)
        lo = max(VOL_WIN + VOL_Z, tnow - IC_LOOKBACK)
        volz = np.full(T, np.nan)
        for s in range(lo, T):
            wv = vol[s - VOL_Z:s]
            volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
        ret1 = ret1_full[:T]
        fh = np.clip(volz[tnow], -3, 3) / 3.0
        if np.isnan(fh):
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        sig = _side(volz, ret1, tnow, fh)
        if sig is None:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        msigs = []
        for lb in LBS:
            z = _mom_z(lpA, T, tnow, lb)
            fhm = np.clip(z[tnow], -3, 3) / 3.0
            if np.isnan(fhm): continue
            m = _side(z, ret1, tnow, fhm)
            if m is not None: msigs.append(m)
        if msigs:
            msig = float(np.mean(msigs))
            av = COMBINE_GAIN * (sig + msig) * 100_000.0
        else:
            av = SWITCH_GAIN * sig * 100_000.0
        out[k] = _finalize(av, cur0, cap_dol, lim, state, tnow)
    return out


LBS = (5, 7, 10, 12, 15, 20)
print(f"\n=== E67: continuous momentum blend across lookbacks {LBS} replacing the "
      f"MOM_LB_SHORT({MOM_LB_SHORT})/LONG({MOM_LB_LONG}) binary switch ===")
t0 = time.time()
algo_mb = algo_momblend(LBS)
Pz = IDIO_POS.copy(); Pz[0, :] = algo_mb
scs = scs_curve(Pz)
wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
nworse = int((scs < base_scs).sum())
print(f"  momblend {LBS}: OLD={wo:.1f}  NEW={wn:.1f}  rmean={scs.mean():.1f}  rfloor={scs.min():.1f}  "
      f"n_worse={nworse}/{len(scs)}  passed={passed}   [{time.time()-t0:.0f}s]")
print(f"\n  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}")
