"""
test_batch100_catFG_rankstab_regime.py

Batch-100, categories F+G: rank-stability MECHANISM variants (F, ideas 1-4) and cross-leg
ALGO-regime <-> idio/ALGO-mechanism COUPLING ideas (G, ideas 5-8), tested against the real shipped
SAFE_llboost_v10.py via the shared, verified `_v10_harness.py`.

Context, stated honestly up front: two adjacent ideas were already tested and REJECTED this session
(see README.md) --
  - "ALGO-as-boost-leader": adding ALGO itself as a 40th candidate leader in the pairwise boost --
    rejected, small and net-negative where it fires.
  - "ALGO crossover extension" (`test_v17cand_algo_crossover.py` + diagnostics): applying the
    rank-stability short/long crossover construction to ALGO's OWN raw price series as a new
    time-series vote -- rejected on three independent, converging checks (unstable sign across
    lookback choice, 0/49 walk-forward robust configs, gate can't rescue it).
Both of those treat ALGO's PRICE PATTERN as a new predictive signal. Everything below is different:
ideas 1-4 are variant constructions of the existing rank-stability MECHANISM on the idio side (same
family the shipped blend already belongs to); ideas 5-8 use ALGO's existing internal REGIME STATE
(vol-regime feature `fh`, fast-IC sign `sf`) or the idio book's own pooled-IC state purely as a
CONDITIONING variable for other mechanisms' knobs -- never as a new leader/vote itself.

Category F -- rank-stability mechanism variants (build a standalone SIG[nIdio,nt] array, blend with
`H.blend_signal` on top of `wz = H.WZ_PRE[:,t] + V10.BOOST_K*H.BOOST_BASE[:,t]` -- i.e. each variant
REPLACES the shipped rank-stability blend for a clean apples-to-apples comparison against the same
v10 baseline, not an addition on top of it):
  1. Triple-timeframe confirmation: gate the shipped short8/long22 disagreement-fade so it only fires
     when long22's direction ALSO agrees with an even-longer window (40 or 50 days).
  2. Trade the agreement case too: fade (-short_z) on disagreement as shipped, but FOLLOW (+short_z)
     on agreement instead of zeroing it out -- always active, no gate.
  3. Severity-scaled fade: replace the binary gate with continuous scaling by disagreement severity,
     `-short_z * clip(|long_z-short_z|/typical_spread, 0, 2)`, typical_spread = that day's own
     cross-sectional std of |long_z-short_z| (a self-normalizing, unitless severity measure -- so
     the same clip range works regardless of the day's overall dispersion level).
  4. Residual-based crossover: identical short8/long22 crossover construction, applied to the
     ridge's own prediction residual instead of raw log-price returns. APPROXIMATION, stated
     explicitly: residual[i,t] = r[i,t] - sign(WZ_PRE[i,t])*|r[i,t]| (removes the return component
     "explained" by the ridge's own directional call), cumulative-summed into a residual "path" so
     the same short/long-difference construction applies. Not an exact ridge residual (which would
     require refitting), and not claimed to be.

Category G -- cross-leg regime coupling (genuinely new territory: ALGO's STATE conditions other
mechanisms' knobs, on top of the FULL shipped v10 pipeline):
  5. Scale BOOST_K by ALGO's vol-regime feature fh: K_t = BOOST_K*(1+GAIN*fh[t]) and the symmetric
     K_t = BOOST_K*(1+GAIN*|fh[t]|). GAIN=0 is a sanity anchor (must reproduce baseline exactly).
  6. Same idea, scaling RS_WEIGHT instead of BOOST_K.
  7. ALGO fast-IC sign bias: replicate the trailing-90-day plain correlation sign `sf` between
     ALGO's realized-vol feature and its own next return (the piece of `_side()` upstream of the
     EW-agreement gate) causally, then nudge the idio wz uniformly (all 50 names, same small
     additive term) toward that market-direction prior on days the fast IC is large in magnitude
     ("strongly one-signed").
  8. Reverse coupling: pooled trailing-60-day realized IC of the idio book's own final wz against
     realized idio returns (the idio-book equivalent of ALGO's own `_ic` helper, pooled across all
     50 names instead of one instrument) as a state variable, fed into the ALGO leg by scaling
     SWITCH_GAIN / COMBINE_GAIN. The repo's own diagnostics (double-IC-veto section, README.md)
     already found this pooled IC is extremely stable -- mean +0.0675, sd 0.0255, never negative in
     704 days -- so this is tested expecting (and reporting honestly if found) that the state
     variable barely moves and the coupling is close to inert.

Most of these are expected to fail (this repeats a now well-established pattern in this file's
history: idio's pooled edge is stationary and near-saturated, adaptive machinery only pays where
ALGO's genuinely non-stationary IC lives) -- reported below exactly as measured, pass or fail.
"""
import numpy as np, time
import _v10_harness as H

V10 = H.V10
t0 = time.time()


# ==================================================================================================
# shared small helpers
# ==================================================================================================
def zscore(x):
    m = x.mean(); s = x.std()
    return (x - m) / (s + 1e-12)


def build_wz_full(sig_fn, weight, base_wz_fn=None):
    """For category F: wz = WZ_PRE + BOOST_K*BOOST_BASE at every day t, blended with sig_fn(t)
    (returns an (nIdio,) array or None) via H.blend_signal at `weight`. Replaces the shipped
    rank-stability blend (this is a variant of it, not an addition on top of it)."""
    WZ = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        wz = H.WZ_PRE[:, t] + V10.BOOST_K * H.BOOST_BASE[:, t]
        sig = sig_fn(t)
        if sig is not None and np.isfinite(sig).all():
            wz = H.blend_signal(wz, sig, weight)
        WZ[:, t] = wz
    return WZ


# ==================================================================================================
# CATEGORY F -- rank-stability mechanism variants
# ==================================================================================================
print("=" * 90)
print("CATEGORY F -- rank-stability mechanism variants (replace shipped blend, compare vs real v10)")
print("=" * 90)

SHORT_W = V10.RS_SHORT_W  # 8
LONG_W = V10.RS_LONG_W    # 22


# --- idea 1: triple-timeframe confirmation ---------------------------------------------------------
def triple_confirm_signal(t, longest_w):
    if t < max(SHORT_W, LONG_W, longest_w) + 5:
        return None
    short_ret = H.logp[1:, t] - H.logp[1:, t - SHORT_W]
    long_ret = H.logp[1:, t] - H.logp[1:, t - LONG_W]
    longest_ret = H.logp[1:, t] - H.logp[1:, t - longest_w]
    sz = zscore(short_ret); lz = zscore(long_ret); llz = zscore(longest_ret)
    if short_ret.std() < 1e-12 or long_ret.std() < 1e-12 or longest_ret.std() < 1e-12:
        return None
    disagree = np.sign(lz) != np.sign(sz)
    confirm = np.sign(lz) == np.sign(llz)
    return np.where(disagree & confirm, -sz, 0.0)


print("\n--- idea 1: triple-timeframe confirmation (longest window in {40,50}) ---")
res_f1 = []
for longest_w in (40, 50):
    for w in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        WZ = build_wz_full(lambda t, lw=longest_w: triple_confirm_signal(t, lw), w)
        res_f1.append(H.evaluate(f"F1 longest={longest_w} w={w}", WZ))


# --- idea 2: trade the agreement case too (follow instead of zero) --------------------------------
def agree_follow_signal(t):
    if t < max(SHORT_W, LONG_W) + 5:
        return None
    short_ret = H.logp[1:, t] - H.logp[1:, t - SHORT_W]
    long_ret = H.logp[1:, t] - H.logp[1:, t - LONG_W]
    if short_ret.std() < 1e-12 or long_ret.std() < 1e-12:
        return None
    sz = zscore(short_ret); lz = zscore(long_ret)
    disagree = np.sign(lz) != np.sign(sz)
    return np.where(disagree, -sz, sz)


print("\n--- idea 2: always-active (fade disagreement, follow agreement) ---")
res_f2 = []
for w in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
    WZ = build_wz_full(agree_follow_signal, w)
    res_f2.append(H.evaluate(f"F2 w={w}", WZ))


# --- idea 3: severity-scaled fade -------------------------------------------------------------------
def severity_signal(t, cap=2.0):
    if t < max(SHORT_W, LONG_W) + 5:
        return None
    short_ret = H.logp[1:, t] - H.logp[1:, t - SHORT_W]
    long_ret = H.logp[1:, t] - H.logp[1:, t - LONG_W]
    if short_ret.std() < 1e-12 or long_ret.std() < 1e-12:
        return None
    sz = zscore(short_ret); lz = zscore(long_ret)
    diff = np.abs(lz - sz)
    typical_spread = diff.std()
    if typical_spread < 1e-12:
        return None
    severity = np.clip(diff / typical_spread, 0.0, cap)
    return -sz * severity


print("\n--- idea 3: severity-scaled fade (typical_spread = day's cross-sectional std of |lz-sz|) ---")
res_f3 = []
for cap in (1.0, 2.0, 3.0):
    for w in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        WZ = build_wz_full(lambda t, c=cap: severity_signal(t, c), w)
        res_f3.append(H.evaluate(f"F3 cap={cap} w={w}", WZ))


# --- idea 4: residual-based crossover ----------------------------------------------------------
print("\n--- idea 4: residual-based short8/long22 crossover (APPROXIMATE residual, see docstring) ---")
resid = np.full((H.nIdio, H.nt), np.nan)
for t in H.days:
    if t >= H.r.shape[1]:
        continue  # H.r[:,t] = return from day t->t+1; undefined at the last day (nt-1)
    wzp = H.WZ_PRE[:, t]
    if np.isfinite(wzp).all():
        resid[:, t] = H.r[1:, t] - np.sign(wzp) * np.abs(H.r[1:, t])

cumresid = np.zeros((H.nIdio, H.nt))
first_valid = H.days[0]
for t in H.days:
    prev = cumresid[:, t - 1] if t > first_valid else 0.0
    cumresid[:, t] = prev + (resid[:, t] if np.isfinite(resid[:, t]).all() else 0.0)


def residual_crossover_signal(t):
    if t < first_valid + max(SHORT_W, LONG_W) + 5:
        return None
    short_ret = cumresid[:, t] - cumresid[:, t - SHORT_W]
    long_ret = cumresid[:, t] - cumresid[:, t - LONG_W]
    if short_ret.std() < 1e-12 or long_ret.std() < 1e-12:
        return None
    sz = zscore(short_ret); lz = zscore(long_ret)
    disagree = np.sign(lz) != np.sign(sz)
    return np.where(disagree, -sz, 0.0)


res_f4 = []
for w in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
    WZ = build_wz_full(residual_crossover_signal, w)
    res_f4.append(H.evaluate(f"F4 w={w}", WZ))

print(f"\n[category F done, {time.time()-t0:.0f}s elapsed]")


# ==================================================================================================
# CATEGORY G -- cross-leg regime coupling (on top of the FULL shipped v10 pipeline)
# ==================================================================================================
print("\n" + "=" * 90)
print("CATEGORY G -- cross-leg ALGO-regime coupling (on top of full shipped v10)")
print("=" * 90)

FH = H.algo_fh_series()  # causal ALGO vol-regime feature, NaN before enough history


# --- idea 5: scale BOOST_K by ALGO's vol regime -----------------------------------------------
def build_wz_gainK(gain, use_abs):
    WZ = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        fh_t = FH[t]
        gmul = (1.0 + gain * abs(fh_t)) if use_abs else (1.0 + gain * fh_t)
        if np.isnan(fh_t):
            gmul = 1.0
        Kt = V10.BOOST_K * gmul
        wz = H.WZ_PRE[:, t] + Kt * H.BOOST_BASE[:, t]
        WZ[:, t] = H.rs_blend(wz, t)
    return WZ


print("\n--- idea 5: BOOST_K scaled by ALGO regime fh (GAIN=0 must reproduce baseline exactly) ---")
res_g5 = []
GAIN_LIST = [0.0, 0.5, 1.0, 2.0, -0.5, -1.0]
for use_abs in (False, True):
    tag = "abs" if use_abs else "signed"
    for gain in GAIN_LIST:
        WZ = build_wz_gainK(gain, use_abs)
        r_ = H.evaluate(f"G5 {tag} GAIN={gain}", WZ)
        res_g5.append(r_)
        if gain == 0.0:
            ok = abs(r_["wo"] - H.BASE_WO) < 0.01 and abs(r_["wn"] - H.BASE_WN) < 0.01
            print(f"    (GAIN=0 sanity: {'OK, exact baseline' if ok else 'MISMATCH -- BUG'})")


# --- idea 6: scale RS_WEIGHT by ALGO's vol regime ------------------------------------------------
def rs_blend_var_weight(wz, t, weight):
    rs_sig = V10._rank_stability_signal(H.logp[:, :t + 1])
    if rs_sig is None:
        return wz
    s_std = rs_sig.std()
    s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
    return (1 - weight) * wz + weight * s_z * (np.abs(wz).mean() + 1e-12)


def build_wz_gainRSW(gain, use_abs):
    WZ = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        fh_t = FH[t]
        gmul = (1.0 + gain * abs(fh_t)) if use_abs else (1.0 + gain * fh_t)
        if np.isnan(fh_t):
            gmul = 1.0
        w_t = float(np.clip(V10.RS_WEIGHT * gmul, 0.0, 1.0))
        wz = H.WZ_PRE[:, t] + V10.BOOST_K * H.BOOST_BASE[:, t]
        WZ[:, t] = rs_blend_var_weight(wz, t, w_t)
    return WZ


print("\n--- idea 6: RS_WEIGHT scaled by ALGO regime fh (GAIN=0 must reproduce baseline exactly) ---")
res_g6 = []
for use_abs in (False, True):
    tag = "abs" if use_abs else "signed"
    for gain in GAIN_LIST:
        WZ = build_wz_gainRSW(gain, use_abs)
        r_ = H.evaluate(f"G6 {tag} GAIN={gain}", WZ)
        res_g6.append(r_)
        if gain == 0.0:
            ok = abs(r_["wo"] - H.BASE_WO) < 0.01 and abs(r_["wn"] - H.BASE_WN) < 0.01
            print(f"    (GAIN=0 sanity: {'OK, exact baseline' if ok else 'MISMATCH -- BUG'})")


# --- idea 7: ALGO fast-IC sign bias on the idio wz ------------------------------------------------
print("\n--- idea 7: ALGO fast-IC sign (sf) as a market-direction prior, nudging idio wz uniformly ---")


def algo_fastic_sign_series():
    """Causal reconstruction of the `sf`/`icf` piece of V10._algo_vol_shares._side (the plain
    trailing-IC_FAST=90-day correlation sign between ALGO's realized-vol feature `volz` and its own
    next return -- upstream of the EW double-IC agreement gate, which this idea deliberately does NOT
    replicate, since the task is only to extract the fast-IC sign as a market-direction prior)."""
    lpA = H.logp[0]; T = len(lpA)
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[V10.VOL_WIN:] = V10._roll_std(r_, V10.VOL_WIN)
    volz = np.full(T, np.nan)
    for s in range(V10.VOL_WIN + V10.VOL_Z, T):
        wv = vol[s - V10.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]
    sf = np.full(T, np.nan); icf_arr = np.full(T, np.nan)
    for tnow in range(V10.VOL_WIN + V10.VOL_Z, T):
        a = max(0, tnow - V10.IC_FAST)
        xs = volz[a:tnow]; ys = ret1[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60:
            continue
        xs2, ys2 = xs[ok], ys[ok]
        if xs2.std() < 1e-12:
            continue
        icf = float(np.corrcoef(xs2, ys2)[0, 1])
        icf_arr[tnow] = icf
        sf[tnow] = 1.0 if icf >= 0 else -1.0
    return sf, icf_arr


SF, ICF = algo_fastic_sign_series()
valid_icf = ICF[np.isfinite(ICF)]
print(f"  ALGO fast IC (icf) over {len(valid_icf)} valid days: mean={valid_icf.mean():.4f} "
      f"sd={valid_icf.std():.4f} range=[{valid_icf.min():.4f},{valid_icf.max():.4f}] "
      f"frac_neg={100*(valid_icf<0).mean():.1f}%")

res_g7 = []
for thresh in (0.0, 0.05, 0.1):
    for bias in (0.02, 0.05, 0.1, 0.2):
        WZ = np.full((H.nIdio, H.nt), np.nan)
        for t in H.days:
            wz = H.WZ_PRE[:, t] + V10.BOOST_K * H.BOOST_BASE[:, t]
            wz = H.rs_blend(wz, t)
            icf_t = ICF[t]
            if np.isfinite(icf_t) and abs(icf_t) > thresh:
                nudge = bias * SF[t] * (np.abs(wz).mean() + 1e-12)
                wz = wz + nudge
            WZ[:, t] = wz
        res_g7.append(H.evaluate(f"G7 thresh={thresh} bias={bias}", WZ))

print(f"\n[category G idea 5-7 done, {time.time()-t0:.0f}s elapsed]")


# --- idea 8: reverse coupling -- idio pooled trailing-IC feeding ALGO's SWITCH_GAIN/COMBINE_GAIN ---
print("\n--- idea 8: idio-book pooled trailing-60d IC feeding ALGO's SWITCH_GAIN / COMBINE_GAIN ---")


def idio_pooled_ic_series(L=60):
    """Idio-book equivalent of ALGO's own `_ic` helper: pooled correlation between yesterday's final
    wz (H.BASE_WZ, the real shipped combined signal) and realized idio next-day returns, pooled
    across all 50 names, trailing L days."""
    out = np.full(H.nt, np.nan)
    for t in H.days:
        a = max(0, t - L)
        feat = H.BASE_WZ[:, a:t]; y = H.r[1:, a:t]
        ok = np.isfinite(feat) & np.isfinite(y)
        if ok.sum() < 60:
            continue
        xs = feat[ok]; ys = y[ok]
        if xs.std() < 1e-12:
            continue
        out[t] = float(np.corrcoef(xs, ys)[0, 1])
    return out


IC_IDIO = idio_pooled_ic_series(60)
valid_ic = IC_IDIO[np.isfinite(IC_IDIO)]
ic_mean, ic_std = valid_ic.mean(), valid_ic.std()
print(f"  idio pooled trailing-60d IC over {len(valid_ic)} valid days: mean={ic_mean:.4f} "
      f"sd={ic_std:.4f} range=[{valid_ic.min():.4f},{valid_ic.max():.4f}] "
      f"frac_neg={100*(valid_ic < 0).mean():.1f}%  (cf. README's 250d-pooled figure: "
      f"mean +0.0675, sd 0.0255, never negative in 704 days)")
IC_IDIO_Z = np.where(np.isfinite(IC_IDIO), (IC_IDIO - ic_mean) / (ic_std + 1e-12), 0.0)


def algo_raw_scaled(lpA, T, switch_mult, combine_mult):
    """Verbatim V10._algo_vol_shares raw-target logic through the 'av' computation, with
    SWITCH_GAIN and COMBINE_GAIN scaled by switch_mult/combine_mult (both 1.0 reproduces v10
    exactly)."""
    if T < V10.VOL_WIN + V10.VOL_Z + 60:
        return 0.0
    rr = np.diff(lpA[:T])
    vol = np.full(T, np.nan); vol[V10.VOL_WIN:] = V10._roll_std(rr, V10.VOL_WIN)
    tnow = T - 1
    lo = max(V10.VOL_WIN + V10.VOL_Z, tnow - V10.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V10.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:T] - lpA[:T - 1]

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
        icf = _ic(feat, V10.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        if not V10.IC_BLEND: return sf * fhv
        ics = [x for x in (_ic_ew(feat, hl, V10.IC_EW_W) for hl in V10.IC_EW_HL) if x is not None]
        if len(ics) < len(V10.IC_EW_HL): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        return 0.0
    sig = _side(volz, fh)
    if sig is None:
        return 0.0
    mom_lb = V10.MOM_LB_SHORT if fh > 0 else V10.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:T] - lpA[:T - mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V10.VOL_Z, tnow - V10.IC_EW_W), T):
        wm = mom[s - V10.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        return V10.COMBINE_GAIN * combine_mult * (sig + msig) * 100_000.0
    return V10.SWITCH_GAIN * switch_mult * sig * 100_000.0


def algo_shares_gain_coupled(gain, target):
    """target: 'switch' scales SWITCH_GAIN only, 'combine' scales COMBINE_GAIN only, both by
    (1 + gain*IC_IDIO_Z[t])."""
    lpA = H.logp[0]
    out = np.zeros(H.nt)
    prev = 0; prev_t = -1
    cur0_arr = H.P_[0]
    for k in range(130, H.nt):
        cur0 = cur0_arr[k]; lim = int(H.dlr[0] / cur0)
        gmul = 1.0 + gain * IC_IDIO_Z[k]
        smul = gmul if target == "switch" else 1.0
        cmul = gmul if target == "combine" else 1.0
        av = algo_raw_scaled(lpA, k + 1, smul, cmul)
        have_prev = (k == prev_t + 1)
        if have_prev and k >= V10.DEADBAND_MIN_DAY and abs(av) < V10.DEADBAND_THRESH_FRAC * H.dlr[0]:
            sh = int(np.clip(prev, -lim, lim))
        else:
            av_c = float(np.clip(av, -H.dlr[0], H.dlr[0]))
            sh = int(np.clip(av_c / cur0, -lim, lim))
        out[k] = sh; prev = sh; prev_t = k
    return out


print("  (rebuilding the ALGO leg per GAIN value -- this recomputes the full O(T^2) vol/IC scan, "
      "expect this section to take noticeably longer than categories F/G5-7)")
res_g8 = []
G8_GAIN_LIST = [0.0, 1.0, -1.0, 3.0, -3.0]
for target in ("switch", "combine"):
    for gain in G8_GAIN_LIST:
        tA = time.time()
        algo_arr = algo_shares_gain_coupled(gain, target)
        r_ = H.evaluate(f"G8 {target} GAIN={gain}", H.BASE_WZ, algo_pos_arr=algo_arr)
        res_g8.append(r_)
        if gain == 0.0:
            ok = abs(r_["wo"] - H.BASE_WO) < 0.01 and abs(r_["wn"] - H.BASE_WN) < 0.01
            print(f"    (GAIN=0 sanity: {'OK, exact baseline' if ok else 'MISMATCH -- BUG'}, "
                  f"{time.time()-tA:.0f}s)")

print(f"\n[all categories done, {time.time()-t0:.0f}s elapsed]")


# ==================================================================================================
# summary
# ==================================================================================================
def summarize(label, results):
    passing = [r for r in results if r["passed"]]
    print(f"\n{label}: {len(passing)}/{len(results)} configs pass.")
    if passing:
        best = max(passing, key=lambda r: r["rm"])
        print(f"  best pass by rmean: {best['name']}  OLD={best['wo']:.1f} NEW={best['wn']:.1f} "
              f"rmean={best['rm']:.1f} rfloor={best['rf']:.1f} n_worse={best['nworse']}/61")
    else:
        best = max(results, key=lambda r: r["rm"])
        print(f"  closest by rmean (still fails): {best['name']}  OLD={best['wo']:.1f} "
              f"NEW={best['wn']:.1f} rmean={best['rm']:.1f} rfloor={best['rf']:.1f} "
              f"n_worse={best['nworse']}/61")


print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
summarize("idea 1 (triple-timeframe confirmation)", res_f1)
summarize("idea 2 (trade agreement too)", res_f2)
summarize("idea 3 (severity-scaled fade)", res_f3)
summarize("idea 4 (residual-based crossover, approximate)", res_f4)
summarize("idea 5 (BOOST_K x ALGO regime)", res_g5)
summarize("idea 6 (RS_WEIGHT x ALGO regime)", res_g6)
summarize("idea 7 (ALGO fast-IC sign bias on idio wz)", res_g7)
summarize("idea 8 (idio pooled IC -> ALGO SWITCH/COMBINE_GAIN)", res_g8)

ALL = res_f1 + res_f2 + res_f3 + res_f4 + res_g5 + res_g6 + res_g7 + res_g8
passing_all = [r for r in ALL if r["passed"]]
print(f"\nTOTAL: {len(passing_all)}/{len(ALL)} configs across all 8 ideas pass OLD+NEW+rmean jointly "
      f"vs real SAFE_llboost_v10.")
if passing_all:
    print("PASSING CONFIGS:")
    for r in passing_all:
        print(f"  {r['name']:<28} OLD={r['wo']:.1f} NEW={r['wn']:.1f} rmean={r['rm']:.1f} "
              f"rfloor={r['rf']:.1f} n_worse={r['nworse']}/61")
