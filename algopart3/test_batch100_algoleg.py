"""
test_batch100_algoleg.py

Batch-100 ideas E65, E66, E67, E69, E70 -- all modify ONLY the ALGO index leg's internal
mechanism (_algo_vol_shares). The idio book (ridge ensemble + beta-adjusted target + BLEND
reversion + pairwise boost + rank-stability blend) is IDENTICAL to SAFE_llboost_v10 for every
one of these and is precomputed ONCE and reused, exactly like test_v19cand_boost_ncandidates.py's
caching pattern.

Each idea is implemented as a standalone copy of _algo_vol_shares's body (per the house
convention established in test_v19cand_boost_ncandidates.py's `boost_at_day`: copy the body,
parameterize the piece under test) with local prev_shares/prev_t state instead of touching
V10's module globals, so results don't stomp on each other in the same process.

  E65: add a third, slow (500-day half-life) EW-IC estimator to the double-IC blend gate
       (IC_EW_HL=(20,45) -> (20,45,500), with a correspondingly longer window for the slow one).
  E66: regime-conditional COMBINE_GAIN -- different gain applied when fh>0 (elevated vol) vs
       fh<=0 (calm), instead of one fixed COMBINE_GAIN=16.
  E67: continuous multi-lookback momentum blend (5,7,10,12,15,20) replacing the binary
       MOM_LB_SHORT/LONG switch -- average the IC-gated side signal across all six lookbacks.
  E69: drawdown-based regime detection replacing the vol-z-based "fh>0" test used ONLY to pick
       the momentum lookback (mom_lb) -- everything else (sig, msig gating, combine formula)
       unchanged.
  E70: multi-horizon reversal blend (3,10,30-day fades) ADDED as a third, IC-gated component to
       the combine formula, on top of the existing vol-regime + momentum components.
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
_roll_std = V10._roll_std


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

# ============================================================================================
# shared precompute: idio WZ (ridge ensemble + BLEND + boost + rank-stability), verbatim v10,
# unaffected by anything tested in this file
# ============================================================================================
print("=== precompute: idio WZ (verbatim v10) -- unaffected by all ALGO-leg ideas here ===",
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
    """verbatim V10._algo_vol_shares, called sequentially so its module-level HOLD-deadband
    state advances correctly (matches v19/v20 template convention)."""
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


# ============================================================================================
# shared low-level helpers, generalized copies of _algo_vol_shares's internals
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


def _side(feat, ret1, tnow, fhv, ic_ew_hl=IC_EW_HL, ic_ew_w=IC_EW_W, use_blend=IC_BLEND):
    icf = _ic(feat, ret1, tnow, IC_FAST)
    if icf is None: return None
    sf = 1.0 if icf >= 0 else -1.0
    if not use_blend: return sf * fhv
    if isinstance(ic_ew_w, (list, tuple)):
        ics = [x for x in (_ic_ew(feat, ret1, tnow, hl, w) for hl, w in zip(ic_ew_hl, ic_ew_w)) if x is not None]
    else:
        ics = [x for x in (_ic_ew(feat, ret1, tnow, hl, ic_ew_w) for hl in ic_ew_hl) if x is not None]
    if len(ics) < len(ic_ew_hl): return sf * fhv
    ice = float(np.mean(ics))
    return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0


def _vol_volz_ret1(lpA, tnow, T):
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[VOL_WIN:] = _roll_std(r_, VOL_WIN)
    lo = max(VOL_WIN + VOL_Z, tnow - IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]
    return volz, ret1


def _mom_z(lpA, T, tnow, mom_lb):
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z = np.full(T, np.nan)
    for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T):
        wm = mom[s - VOL_Z:s]; z[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    return z


def _finalize(av, cur0, cap_dol, lim, state, tnow, min_day=DEADBAND_MIN_DAY, thresh=DEADBAND_THRESH_FRAC):
    have_prev = (tnow == state['t'] + 1)
    av = float(np.clip(av, -cap_dol, cap_dol))
    if have_prev and tnow >= min_day and abs(av) < thresh * cap_dol:
        shares = int(np.clip(state['shares'], -lim, lim))
    else:
        shares = int(np.clip(av / cur0, -lim, lim))
    state['shares'] = shares; state['t'] = tnow
    return shares


def evaluate(nm, algo_series, verbose=True):
    Pz = IDIO_POS.copy(); Pz[0, :] = algo_series
    scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<30}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


# ============================================================================================
# E65: third slow (500-day HL) EW-IC estimator added to the double-IC gate
# ============================================================================================
def algo_E65(ic_ew_hl=(20, 45, 500), ic_ew_w=(200, 200, 500)):
    state = {'shares': 0, 't': -1}
    out = np.zeros(nt)
    for k in range(130, nt):
        lpA = logp[0, :k + 1]; T = len(lpA); tnow = T - 1
        cur0 = P_[0, k]; cap_dol = dlr[0]; lim = int(cap_dol / cur0)
        if T < VOL_WIN + VOL_Z + 60:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        volz, ret1 = _vol_volz_ret1(lpA, tnow, T)
        fh = np.clip(volz[tnow], -3, 3) / 3.0
        if np.isnan(fh):
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        sig = _side(volz, ret1, tnow, fh, ic_ew_hl=ic_ew_hl, ic_ew_w=ic_ew_w)
        if sig is None:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        mom_lb = MOM_LB_SHORT if fh > 0 else MOM_LB_LONG
        z10 = _mom_z(lpA, T, tnow, mom_lb)
        fhm = np.clip(z10[tnow], -3, 3) / 3.0
        msig = _side(z10, ret1, tnow, fhm, ic_ew_hl=ic_ew_hl, ic_ew_w=ic_ew_w) if not np.isnan(fhm) else None
        av = COMBINE_GAIN * (sig + msig) * 100_000.0 if msig is not None else SWITCH_GAIN * sig * 100_000.0
        out[k] = _finalize(av, cur0, cap_dol, lim, state, tnow)
    return out


# ============================================================================================
# E66: regime-conditional COMBINE_GAIN
# ============================================================================================
def algo_E66(gain_high, gain_low):
    state = {'shares': 0, 't': -1}
    out = np.zeros(nt)
    for k in range(130, nt):
        lpA = logp[0, :k + 1]; T = len(lpA); tnow = T - 1
        cur0 = P_[0, k]; cap_dol = dlr[0]; lim = int(cap_dol / cur0)
        if T < VOL_WIN + VOL_Z + 60:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        volz, ret1 = _vol_volz_ret1(lpA, tnow, T)
        fh = np.clip(volz[tnow], -3, 3) / 3.0
        if np.isnan(fh):
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        sig = _side(volz, ret1, tnow, fh)
        if sig is None:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        mom_lb = MOM_LB_SHORT if fh > 0 else MOM_LB_LONG
        z10 = _mom_z(lpA, T, tnow, mom_lb)
        fhm = np.clip(z10[tnow], -3, 3) / 3.0
        msig = _side(z10, ret1, tnow, fhm) if not np.isnan(fhm) else None
        if msig is not None:
            gain = gain_high if fh > 0 else gain_low
            av = gain * (sig + msig) * 100_000.0
        else:
            av = SWITCH_GAIN * sig * 100_000.0
        out[k] = _finalize(av, cur0, cap_dol, lim, state, tnow)
    return out


# ============================================================================================
# E67: continuous multi-lookback momentum blend (replaces binary MOM_LB_SHORT/LONG switch)
# ============================================================================================
def algo_E67(mom_lbs=(5, 7, 10, 12, 15, 20)):
    state = {'shares': 0, 't': -1}
    out = np.zeros(nt)
    for k in range(130, nt):
        lpA = logp[0, :k + 1]; T = len(lpA); tnow = T - 1
        cur0 = P_[0, k]; cap_dol = dlr[0]; lim = int(cap_dol / cur0)
        if T < VOL_WIN + VOL_Z + 60:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        volz, ret1 = _vol_volz_ret1(lpA, tnow, T)
        fh = np.clip(volz[tnow], -3, 3) / 3.0
        if np.isnan(fh):
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        sig = _side(volz, ret1, tnow, fh)
        if sig is None:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        msigs = []
        for mlb in mom_lbs:
            if T < mlb + VOL_Z + 10:
                continue
            z10 = _mom_z(lpA, T, tnow, mlb)
            fhm = np.clip(z10[tnow], -3, 3) / 3.0
            if np.isnan(fhm):
                continue
            m = _side(z10, ret1, tnow, fhm)
            if m is not None:
                msigs.append(m)
        msig = float(np.mean(msigs)) if msigs else None
        av = COMBINE_GAIN * (sig + msig) * 100_000.0 if msig is not None else SWITCH_GAIN * sig * 100_000.0
        out[k] = _finalize(av, cur0, cap_dol, lim, state, tnow)
    return out


# ============================================================================================
# E69: drawdown-based regime detection replacing "fh>0" for the mom_lb pick ONLY
# ============================================================================================
def algo_E69(dd_window=VOL_Z):
    state = {'shares': 0, 't': -1}
    out = np.zeros(nt)
    for k in range(130, nt):
        lpA = logp[0, :k + 1]; T = len(lpA); tnow = T - 1
        cur0 = P_[0, k]; cap_dol = dlr[0]; lim = int(cap_dol / cur0)
        if T < VOL_WIN + VOL_Z + dd_window + 60:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        volz, ret1 = _vol_volz_ret1(lpA, tnow, T)
        fh = np.clip(volz[tnow], -3, 3) / 3.0
        if np.isnan(fh):
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        sig = _side(volz, ret1, tnow, fh)
        if sig is None:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        # -- drawdown-based regime indicator (replaces "fh>0" for mom_lb only) --
        dd = np.full(T, np.nan)
        for s in range(dd_window, T):
            dd[s] = lpA[s] - lpA[max(0, s - dd_window):s + 1].max()
        addl = np.abs(dd)
        lo_ = max(dd_window, tnow - VOL_Z)
        ddz = np.full(T, np.nan)
        for s in range(max(dd_window + VOL_Z, lo_), T):
            wv = addl[s - VOL_Z:s]
            ddz[s] = (addl[s] - wv.mean()) / (wv.std() + 1e-12)
        stress = ddz[tnow]
        regime_high = (not np.isnan(stress)) and stress > 0
        mom_lb = MOM_LB_SHORT if regime_high else MOM_LB_LONG
        z10 = _mom_z(lpA, T, tnow, mom_lb)
        fhm = np.clip(z10[tnow], -3, 3) / 3.0
        msig = _side(z10, ret1, tnow, fhm) if not np.isnan(fhm) else None
        av = COMBINE_GAIN * (sig + msig) * 100_000.0 if msig is not None else SWITCH_GAIN * sig * 100_000.0
        out[k] = _finalize(av, cur0, cap_dol, lim, state, tnow)
    return out


# ============================================================================================
# E70: multi-horizon reversal blend (3,10,30-day fades) added as a THIRD IC-gated component
# ============================================================================================
def _cum_z(lpA, T, tnow, L):
    cum = np.full(T, np.nan); cum[L:] = lpA[L:] - lpA[:-L]
    z = np.full(T, np.nan)
    for s in range(max(L + VOL_Z, tnow - IC_EW_W), T):
        wc = cum[s - VOL_Z:s]; z[s] = (cum[s] - wc.mean()) / (wc.std() + 1e-12)
    return z


def algo_E70(rev_lbs=(3, 10, 30), rev_gain=8.0):
    state = {'shares': 0, 't': -1}
    out = np.zeros(nt)
    for k in range(130, nt):
        lpA = logp[0, :k + 1]; T = len(lpA); tnow = T - 1
        cur0 = P_[0, k]; cap_dol = dlr[0]; lim = int(cap_dol / cur0)
        if T < VOL_WIN + VOL_Z + 60:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        volz, ret1 = _vol_volz_ret1(lpA, tnow, T)
        fh = np.clip(volz[tnow], -3, 3) / 3.0
        if np.isnan(fh):
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        sig = _side(volz, ret1, tnow, fh)
        if sig is None:
            state['t'] = tnow; state['shares'] = 0; out[k] = 0; continue
        mom_lb = MOM_LB_SHORT if fh > 0 else MOM_LB_LONG
        z10 = _mom_z(lpA, T, tnow, mom_lb)
        fhm = np.clip(z10[tnow], -3, 3) / 3.0
        msig = _side(z10, ret1, tnow, fhm) if not np.isnan(fhm) else None
        rev_sides = []
        for L in rev_lbs:
            if T < L + VOL_Z + 10:
                continue
            zL = _cum_z(lpA, T, tnow, L)
            fhL = np.clip(zL[tnow], -3, 3) / 3.0
            if np.isnan(fhL):
                continue
            sL = _side(zL, ret1, tnow, fhL)
            if sL is not None:
                rev_sides.append(sL)
        rev_component = float(np.mean(rev_sides)) if rev_sides else 0.0
        base_av = COMBINE_GAIN * (sig + msig) if msig is not None else SWITCH_GAIN * sig
        av = (base_av + rev_gain * rev_component) * 100_000.0
        out[k] = _finalize(av, cur0, cap_dol, lim, state, tnow)
    return out


# ============================================================================================
if not SANITY_OK:
    raise SystemExit("Sanity check failed -- aborting, fix precompute before trusting results.")

print("\n=== E65: third slow (HL=500,W=500) EW-IC estimator added to the double-IC gate ===")
t0 = time.time()
r65 = evaluate("E65 add HL=500,W=500", algo_E65())
print(f"  [{time.time()-t0:.0f}s]")

print("\n=== E66: regime-conditional COMBINE_GAIN (gain_high if fh>0 else gain_low) ===")
t0 = time.time()
e66_results = []
for gh, gl in [(20, 12), (12, 20), (22, 10), (10, 22), (18, 14)]:
    e66_results.append(evaluate(f"E66 high={gh},low={gl}", algo_E66(gh, gl)))
print(f"  [{time.time()-t0:.0f}s]")

print("\n=== E67: continuous multi-lookback momentum blend {5,7,10,12,15,20} ===")
t0 = time.time()
r67 = evaluate("E67 mom_lbs=5..20", algo_E67())
print(f"  [{time.time()-t0:.0f}s]")

print("\n=== E69: drawdown-based regime detection (replaces fh>0 for mom_lb pick only) ===")
t0 = time.time()
r69 = evaluate("E69 dd_window=60", algo_E69())
print(f"  [{time.time()-t0:.0f}s]")

print("\n=== E70: multi-horizon reversal blend {3,10,30} added as 3rd IC-gated component ===")
t0 = time.time()
e70_results = []
for rg in (4.0, 8.0, 12.0, 16.0, 20.0):
    e70_results.append(evaluate(f"E70 rev_gain={rg}", algo_E70(rev_gain=rg)))
print(f"  [{time.time()-t0:.0f}s]")

print("\n\n================ SUMMARY (baseline OLD={:.1f} NEW={:.1f} rmean={:.1f} rfloor={:.1f}) ================"
      .format(base_wo, base_wn, base_scs.mean(), base_scs.min()))
for label, res in [("E65", [r65]), ("E66", e66_results), ("E67", [r67]), ("E69", [r69]), ("E70", e70_results)]:
    best = max(res, key=lambda c: c["rm"])
    npass = sum(1 for c in res if c["passed"])
    print(f"{label}: {npass}/{len(res)} configs pass. Best by rmean: {best['name']} "
          f"OLD={best['wo']:.1f} NEW={best['wn']:.1f} rmean={best['rm']:.1f} rfloor={best['rf']:.1f} "
          f"n_worse={best['nworse']}/61 passed={best['passed']}")
