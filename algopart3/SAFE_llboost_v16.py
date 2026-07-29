"""
================================================================================
###  SAFE_llboost_v16.py  ·  v15 + an IC/t-stat significance gate on the fallback switch  ###
================================================================================
Built 2026-07-29, in direct response to a user-raised concern about SAFE_llboost_v15.py's
`_pick_at`: the "a fallback must actually be winning" check compares RAW trailing-60-day summed
PnL between champ and each fallback with NO significance test at all -- a fallback could "win" over
that window purely by lookback luck (a 60-day sample is small enough that noise alone can produce a
positive, champ-beating sum) and the switch would fire anyway. This repo already has the exact
IC + t-stat machinery to fix this -- `SAFE_lldollar.py`'s original validator (`_ic1`/`_ic`/`_tstat`)
-- it was just never wired into the PnL-sum-based `_pick_at` this lineage adopted later. Ported
verbatim (generalized via `_sig_at`, same as v14/v15's own generalization pattern) and added as an
ADDITIONAL required condition, not a replacement for the existing PnL-sum check.

SCOPE (explicitly NOT done here, per direct discussion): the kill switch (`_kill`) has the IDENTICAL
"bare threshold, no significance test" weakness -- it is UNCHANGED in this file, left exactly as
v15's for now, so the effect of gating the fallback-switch specifically can be isolated before
touching the kill switch too.

MECHANISM CHANGE: `_pick_at`'s fallback-selection loop now requires, for the candidate currently
winning on raw PnL, that its own trailing-ROT_W-day daily cross-sectional IC (correlation between
its forecast values and same-day realized returns, one IC value per day, exactly
`SAFE_lldollar.py`'s `_ic1`) has a t-stat (`_tstat`, same formula: mean/(std/sqrt(n))) exceeding
IC_TCRIT. This is checked ONLY for whichever fallback is currently leading on PnL (not all three
every day) -- cheap, and preserves the existing "best candidate" selection logic exactly, just adds
a veto.

IC_TCRIT=2.0 chosen as a conventional ~95%-one-sided significance level -- NOT swept against
alternatives in this file; a first pass, consistent with how this repo flags untuned thresholds
elsewhere (e.g. SAFE_llboost_v13's BOOST_SEL_FALLBACK_HL=1000).

VALIDATION (`test_v16_ic_gate.py`): same three-harness convention as v15 -- real prices.txt (does
this change anything on real data, where v15 was already silent every day?), change-point
reverse/rotate (does gating out low-significance switches cost the now-confirmed-real reverse-mode
edge, or does it filter noise without losing the real signal?), trend-regime momentum/flip/noise
(does it reduce the whipsaw mistiming identified in v15, where a stale momentum-fallback trade
persisted into a reversion phase -- would that specific trade have failed an IC significance check?).
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v16 (v15 + IC/t-stat significance gate on fallback switch)"

# --- idio ridge + ALGO adaptive-vol leg (identical to SAFE_llboost_v9-v15.py) ---
HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
WARMUP      = 96

VOL_WIN     = 20
VOL_Z       = 60
IC_LOOKBACK = 250
VOL_GAIN    = 15.0

VOL_MODE    = "switch"
IC_FAST     = 90
SWITCH_GAIN = 2.5

IC_BLEND    = True
IC_EW_HL    = (20, 45)
IC_EW_W     = 200

VOL_COMBINE = True
MOM_LB_SHORT = 7
MOM_LB_LONG  = 12
COMBINE_GAIN = 16.0

# --- ALGO min-conviction HOLD deadband (identical to SAFE_llboost_v8.py) ---
DEADBAND_THRESH_FRAC = 0.25
DEADBAND_MIN_DAY = 400

# --- beta-adjusted idio ridge target (identical to SAFE_llboost_v9.py) ---
BETA_DEMEAN_LAM = 0.6
BETA_DEMEAN_W = 500

# --- rank-stability trend/pullback blend (identical to SAFE_llboost_v10.py) ---
RS_SHORT_W = 8
RS_LONG_W = 22
RS_WEIGHT = 0.015

# --- pairwise boost parameters (plain, no decayed-selection fallback -- identical to v12/v15) ---
BOOST_K = 1.5
BOOST_MIN_DAY = 480
BOOST_N_CANDIDATES = 39
BOOST_IC_L = 250
BOOST_P = 2.0
BOOST_SCALE_W = 1000

# --- post-jump fixed-size fade (identical to SAFE_llboost_v12/v13/v14/v15.py) ---
FADE_W        = 40
FADE_K_SIGMA  = 2.0
FADE_EXTRA_W  = 0.06

# --- idio kill switch, PnL-sum trigger (UNCHANGED from v11/v12/v13/v14/v15.py -- deliberately not
#     gated in this file, see module docstring) ---
KILL_ON     = True
KILL_MARGIN = 0.0
KILL_P      = 1
ROT_W       = 60

# --- momentum/xsac insurance layer (identical to SAFE_llboost_v14/v15.py's Part B) ---
ROT_P       = 5        # consecutive-day persistence required before rotating away from champ
XSAC_W      = 40        # trailing window for the cross-sectional lag-1 autocorr regime detector
XSAC_TH     = 0.07       # real-data max is +0.061 -- 0 false flags on all real days at this threshold
XSAC_P      = 5          # xsac must stay above XSAC_TH for this many consecutive gradable days
MOMJT_L     = 120
MOMJT_S     = 20
RESIDM_L    = 120
RESIDM_S    = 20
FALLBACKS   = ("mom", "momJT", "residMom")

# --- NEW in v16: IC/t-stat significance gate on the fallback switch (_pick_at only) ---
IC_TCRIT    = 2.0       # min t-stat of the winning fallback's own trailing-ROT_W daily IC

# cache lookback: deepest history any of {kill, choose, xsac} needs
LOOKBACK  = ROT_W + max(KILL_P, ROT_P) + 6      # = 71
PRUNE_PAD = 10

_DLR = None
_PREV_ALGO_SHARES = 0
_PREV_T = -1

_SIG = {}   # n -> champion's final traded idio forecast (50-vec) built from prices[:, :n]
_FB  = {}   # n -> {"mom":vec, "momJT":vec, "residMom":vec}, cheap tail-slice fallback forecasts
_RET = {}   # n -> demeaned realized idio return over day n
_XC  = {}   # n -> cross-sectional lag-1 autocorr corr(_RET[n-1], _RET[n])
_PN  = {}   # (name, n) -> traded-PnL-sign proxy: sign(_sig_at(name,n)) . _RET[n]
_ICD = {}   # (name, n) -> daily cross-sectional IC of that signal's forecast vs realized return


def _limits(nInst):
    global _DLR
    if _DLR is None or len(_DLR) != nInst:
        _DLR = np.full(nInst, 10_000.0); _DLR[0] = 100_000.0
    return _DLR


def _ewls_ridge(X, Y, hl, a):
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


def _beta_adjusted_target(r):
    """Identical to SAFE_llboost_v9._beta_adjusted_target."""
    rs = r[1:]
    cf = rs.mean(0)
    m = rs.shape[1]
    hi = m - 1
    lo = max(0, hi - BETA_DEMEAN_W)
    if hi - lo >= 30:
        seg_y = rs[:, lo:hi]; seg_f = cf[lo:hi]
        vf = seg_f.var()
        if vf > 1e-24:
            beta = (seg_y * seg_f[None, :]).mean(1) - seg_y.mean(1) * seg_f.mean()
            beta = beta / vf
        else:
            beta = np.ones(rs.shape[0])
    else:
        beta = np.ones(rs.shape[0])
    Y = rs[:, 1:].T - BETA_DEMEAN_LAM * beta[None, :] * cf[1:m][:, None]
    return Y


def _rank_stability_signal(logp):
    """Identical to SAFE_llboost_v10-v15._rank_stability_signal."""
    t = logp.shape[1]
    if t < max(RS_SHORT_W, RS_LONG_W) + 5:
        return None
    short_ret = logp[1:, -1] - logp[1:, -1 - RS_SHORT_W]
    long_ret = logp[1:, -1] - logp[1:, -1 - RS_LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        return None
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    return np.where(disagree, -sz, 0.0)


def _roll_std(x, w):
    """std of every length-w window of x (population, ddof=0); out[i] = std(x[i:i+w])."""
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


def _algo_vol_shares(lpA, cur0, cap_dol):
    """Adaptive realized-vol leg -> integer share target for ALGO (instrument 0). Causal.
    Identical to SAFE_llboost_v9-v15 (unaffected by anything else in this file -- this leg has its
    own independent forecast and is never flattened/rotated)."""
    global _PREV_ALGO_SHARES, _PREV_T
    T = len(lpA)
    tnow = T - 1
    have_prev = (tnow == _PREV_T + 1)
    if T < VOL_WIN + VOL_Z + 60:
        _PREV_T = tnow
        _PREV_ALGO_SHARES = 0
        return 0
    r = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[VOL_WIN:] = _roll_std(r, VOL_WIN)
    lo = max(VOL_WIN + VOL_Z, tnow - IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
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
        if not IC_BLEND: return sf * fhv
        ics = [x for x in (_ic_ew(feat, hl, IC_EW_W) for hl in IC_EW_HL) if x is not None]
        if len(ics) < len(IC_EW_HL): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        _PREV_T = tnow
        _PREV_ALGO_SHARES = 0
        return 0
    if VOL_MODE == "switch":
        sig = _side(volz, fh)
        if sig is None:
            _PREV_T = tnow
            _PREV_ALGO_SHARES = 0
            return 0
        if VOL_COMBINE:
            mom_lb = MOM_LB_SHORT if fh > 0 else MOM_LB_LONG
            mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
            z10 = np.full(T, np.nan)
            for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T):
                wm = mom[s - VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
            fhm = np.clip(z10[tnow], -3, 3) / 3.0
            msig = _side(z10, fhm) if not np.isnan(fhm) else None
            if msig is not None:
                av = COMBINE_GAIN * (sig + msig) * 100_000.0
            else:
                av = SWITCH_GAIN * sig * 100_000.0
        else:
            av = SWITCH_GAIN * sig * 100_000.0
    else:
        ic = _ic(volz, IC_LOOKBACK)
        if ic is None:
            _PREV_T = tnow
            _PREV_ALGO_SHARES = 0
            return 0
        av = VOL_GAIN * max(0.0, ic) * fh * 100_000.0

    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)

    if have_prev and tnow >= DEADBAND_MIN_DAY and abs(av) < DEADBAND_THRESH_FRAC * cap_dol:
        shares = int(np.clip(_PREV_ALGO_SHARES, -lim, lim))
    else:
        shares = int(np.clip(av / cur0, -lim, lim))

    _PREV_ALGO_SHARES = shares
    _PREV_T = tnow
    return shares


BOOST_ALPHA = 0.05


def _sig_threshold(n_samples):
    """Min |corr| to be significant at BOOST_ALPHA, Bonferroni-corrected for BOOST_N_CANDIDATES
    simultaneous tests, given the ACTUAL number of return-pairs available right now -- causal, no
    full-sample look-ahead."""
    if n_samples < 10:
        return 1.0
    alpha_adj = BOOST_ALPHA / BOOST_N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def _corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


def _pairwise_boost(rs):
    """Identical to SAFE_llboost_v8/v9/v10/v11/v12/v15._pairwise_boost -- plain, undecayed,
    full-history-only candidate selection."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = _sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = _corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs[i]
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
        boost[j] = lead_boost[-1]
    return boost


def _idio_signal(prcSoFar):
    """The champion's FULL, final idio forecast: ridge ensemble + beta-adjusted target, BLEND
    reversion, plain pairwise boost, rank-stability blend, post-jump fixed-size fade. Identical to
    SAFE_llboost_v15._idio_signal. Returns a 50-vector (idio names only, ALGO excluded)."""
    logp = np.log(prcSoFar)
    r = logp[:, 1:] - logp[:, :-1]

    Y = _beta_adjusted_target(r)
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = _ewls_ridge(r[:, :-1].T, Y, hl, RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if BLEND > 0:
        rr = logp[1:, -1] - logp[1:, -1 - REV_W]
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        wz = (1 - BLEND) * wz + BLEND * rv

    boost = _pairwise_boost(r[1:])
    wz = wz + BOOST_K * boost

    rs_sig = _rank_stability_signal(logp)
    if rs_sig is not None:
        s_std = rs_sig.std()
        s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)

    idio_r = r[1:]
    if idio_r.shape[1] >= FADE_W + 1:
        sigma = idio_r[:, -1 - FADE_W:-1].std(axis=1)
        jump = idio_r[:, -1]
        flagged = np.abs(jump) > FADE_K_SIGMA * (sigma + 1e-12)
        if flagged.any():
            scale = np.abs(wz).mean() + 1e-12
            fade_dir = -np.sign(jump)
            wz = wz.copy()
            wz[flagged] = wz[flagged] + FADE_EXTRA_W * fade_dir[flagged] * scale

    return wz


def _fallback_signals(P):
    """Identical to SAFE_llboost_v14/v15.py's Part B: mom/momJT/residMom computed from a small
    trailing tail slice of P. Returns {name: 50-vec or None} -- None means insufficient history,
    caller substitutes champ."""
    lookback = max(REV_W, MOMJT_L, RESIDM_L) + 2
    n = P.shape[1]
    Ptail = P[:, max(0, n - lookback):n]
    logp = np.log(Ptail)
    r = logp[:, 1:] - logp[:, :-1]
    idio_logp = logp[1:]
    m = idio_logp.shape[1]

    out = {}

    if m >= REV_W + 1:
        rr = idio_logp[:, -1] - idio_logp[:, -1 - REV_W]
        out["mom"] = rr - rr.mean()
    else:
        out["mom"] = None

    if m >= MOMJT_L + 1:
        g = idio_logp[:, -1 - MOMJT_S] - idio_logp[:, -1 - MOMJT_L]
        g = g - g.mean()
        out["momJT"] = g / (g.std() + 1e-12)
    else:
        out["momJT"] = None

    if r.shape[1] >= RESIDM_L:
        Rwin = r[1:, -RESIDM_L:]; r0win = r[0, -RESIDM_L:]; r0c = r0win - r0win.mean()
        beta = (Rwin @ r0c) / (r0c @ r0c + 1e-12)
        resid = Rwin - beta[:, None] * r0win[None, :]
        cum = (resid[:, :RESIDM_L - RESIDM_S] if RESIDM_S > 0 else resid).sum(1)
        cum = cum - cum.mean()
        out["residMom"] = cum / (cum.std() + 1e-12)
    else:
        out["residMom"] = None

    return out


def _sig_at(name, n):
    return _SIG[n] if name == "champ" else _FB[n][name]


def _ensure_cache(P):
    """Fill _SIG/_FB/_RET over the trailing LOOKBACK window, then prune stale entries. Same
    monotonic-t / single-panel-per-process caveat as every sibling file."""
    T = P.shape[1]
    lo = max(WARMUP, T - LOOKBACK)
    for n in range(lo, T + 1):
        if n < WARMUP:
            continue
        if n not in _SIG:
            _SIG[n] = _idio_signal(P[:, :n])
        if n not in _FB:
            fb = _fallback_signals(P[:, :n])
            _FB[n] = {name: (fb[name] if fb[name] is not None else _SIG[n].copy())
                      for name in FALLBACKS}
        if n not in _RET and n < T:
            R = np.log(P[1:, n]) - np.log(P[1:, n - 1])
            _RET[n] = R - R.mean()
    cut = lo - PRUNE_PAD
    for d in (_SIG, _FB, _RET, _XC):
        for k in [k for k in d if k < cut]:
            del d[k]
    for k in [k for (nm, k) in _PN if k < cut]:
        for nm in ("champ",) + FALLBACKS:
            _PN.pop((nm, k), None)
    for k in [k for (nm, k) in _ICD if k < cut]:
        for nm in ("champ",) + FALLBACKS:
            _ICD.pop((nm, k), None)


def _pn1(name, n):
    """Traded-PnL-sign proxy for (signal name, day n): sign(_sig_at(name,n)).realized_return, summed
    over the 50 idio names. Identical to SAFE_llboost_v14/v15's generalized _pn1."""
    key = (name, n)
    v = _PN.get(key)
    if v is None:
        v = float((np.sign(_sig_at(name, n)) * _RET[n]).sum())
        _PN[key] = v
    return v


def _pn(name, lo, hi):
    return np.array([_pn1(name, n) for n in range(lo, hi)])


def _ic1(name, n):
    """NEW in v16: daily cross-sectional IC (Pearson correlation) between signal `name`'s forecast
    and the same day's realized idio return, across the 50 stocks. Verbatim formula from
    SAFE_lldollar.py's `_ic1`, generalized to any signal via `_sig_at` (same generalization pattern
    v14/v15 already used for `_pn1`)."""
    key = (name, n)
    v = _ICD.get(key)
    if v is None:
        f = _sig_at(name, n); g = _RET[n]
        fm = f - f.mean(); gm = g - g.mean()
        denom = np.sqrt((fm @ fm) * (gm @ gm))
        v = float(fm @ gm / denom) if denom > 1e-18 else 0.0
        _ICD[key] = v
    return v


def _ic(name, lo, hi):
    return np.array([_ic1(name, n) for n in range(lo, hi)])


def _tstat(x):
    """Verbatim from SAFE_lldollar.py's `_tstat`: mean/(std/sqrt(n)) of an array."""
    x = np.asarray(x, float)
    return float(x.mean() / (x.std() / np.sqrt(len(x)) + 1e-18))


def _xc1(n):
    """Cross-sectional lag-1 autocorr between realized idio returns of day n-1 and day n. Memoized.
    Identical to algopart2/SAFE_rotate.py's _xc1 -- signal-agnostic, needs only _RET."""
    v = _XC.get(n)
    if v is None:
        a = _RET.get(n - 1); b = _RET.get(n)
        if a is None or b is None:
            return None
        d = np.sqrt((a @ a) * (b @ b))
        v = float(a @ b / d) if d > 1e-18 else 0.0
        _XC[n] = v
    return v


def _xsac(a):
    """Trailing XSAC_W-day mean cross-sectional lag-1 autocorr as of day a (positive = momentum
    regime). Identical to algopart2/SAFE_rotate.py's _xsac."""
    vals = [_xc1(n) for n in range(a - XSAC_W + 1, a + 1)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) >= XSAC_W // 2 else None


def _xsac_flag(T):
    """Identical to algopart2/SAFE_rotate.py's _xsac_flag: sustained above XSAC_TH for XSAC_P days."""
    for a in range(T - XSAC_P, T):
        v = _xsac(a)
        if v is None or v <= XSAC_TH:
            return False
    return True


def _pick_at(a):
    """One day's validator verdict: 'champ' unless the champion is sick (trailing PnL-sum negative
    OR xsac flags a momentum regime) AND a fallback is beating it AND (NEW in v16) that fallback's
    own trailing-ROT_W daily IC has a t-stat exceeding IC_TCRIT -- a bare PnL-sum win is no longer
    enough by itself, guarding against a fallback that "won" the trailing window by lookback luck
    rather than real predictive power."""
    lo = a - ROT_W + 1
    if lo < WARMUP:
        return "champ"
    pn_c = _pn("champ", lo, a + 1)
    xs = _xsac(a)
    champ_sick = (pn_c.sum() < KILL_MARGIN) or (xs is not None and xs > XSAC_TH)
    if not champ_sick:
        return "champ"
    best = None; best_v = -1e18
    for name in FALLBACKS:
        pf = _pn(name, lo, a + 1)
        if (pf - pn_c).sum() > 0.0 and pf.sum() > 0.0 and pf.sum() > best_v:
            ic = _ic(name, lo, a + 1)
            if _tstat(ic) > IC_TCRIT:            # NEW: significance veto
                best_v = pf.sum(); best = name
    return best if best is not None else "champ"


def _choose(T):
    """Requires the SAME fallback to win _pick_at on each of the last ROT_P consecutive days before
    switching away from champ. Identical to SAFE_llboost_v14/v15's _choose."""
    picks = [_pick_at(a) for a in range(T - ROT_P, T)]
    if picks and picks[0] is not None and picks[0] != "champ" and all(p == picks[0] for p in picks):
        return picks[0]
    return "champ"


def _kill(T, chosen):
    """UNCHANGED from v15 -- deliberately not gated in this file, see module docstring. Flattens the
    idio book if whichever signal is currently chosen has its trailing-ROT_W summed realized
    PnL-sign proxy below KILL_MARGIN, re-evaluated fresh each day (KILL_P=1)."""
    if not KILL_ON:
        return False
    for a in range(T - KILL_P, T):
        lo = a - ROT_W + 1
        if lo < WARMUP:
            return False
        pn = _pn(chosen, lo, a + 1)
        if not (pn.sum() < KILL_MARGIN):
            return False
    return True


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nInst, t = prcSoFar.shape
    dlr = _limits(nInst)
    cur = prcSoFar[:, -1]
    pos = np.zeros(nInst)
    if t < WARMUP:
        return pos.astype(int)

    _ensure_cache(prcSoFar)

    ready = t >= WARMUP + ROT_W + max(ROT_P, KILL_P)
    chosen = _choose(t) if ready else "champ"
    wz = _sig_at(chosen, t)

    killed = ready and _kill(t, chosen)
    if not killed:
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    pos[0] = _algo_vol_shares(np.log(prcSoFar)[0], cur[0], dlr[0])

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
