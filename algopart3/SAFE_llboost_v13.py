"""
================================================================================
###  SAFE_llboost_v13.py  ·  SAFE_llboost_v11 + gated decayed-selection boost fallback  ###
================================================================================
Everything is IDENTICAL to SAFE_llboost_v11.py (idio ridge + beta-adjusted target, BLEND reversion,
rank-stability blend, ALGO leg, the PnL-sum idio kill switch) EXCEPT `_pairwise_boost`, which gets
one new, GATED fallback path -- see README's change-point section for the full motivation.

BACKGROUND: the boost's leader-*selection* step uses an undecayed FULL-HISTORY correlation, which
takes a median of ~1000-1200 synthetic post-change days to hand a genuinely new relationship the win
over the old one (confirmed via an extended-runway stress test -- see README). A candidate fix
(exponentially decaying the SAME selection step for every follower, every day) was tested and
REJECTED: it meaningfully speeds up reselection, but checked against this file's own n_worse-of-61
bar it costs real, non-trivial rolling-window consistency (14-27/61 windows worse depending on
half-life) that the averaged OLD/NEW/rmean numbers alone did not reveal.

THE IDEA THIS FILE TESTS: don't apply decay everywhere -- only fall back to a decayed candidate
search for a follower whose FULL-HISTORY path already failed to produce a boost that day (either no
candidate cleared Bonferroni significance, or the selected candidate failed its own trailing-250-day
validation IC). For any follower where the existing, real-data-validated full-history path still
works, behavior is IDENTICAL to v11/v10 by construction -- there is no way for this change to touch
a currently-healthy relationship. The decayed fallback only ever gets a chance to act where the
current mechanism was already contributing zero.

MECHANISM (`_pairwise_boost`, this file):
  1. Compute the candidate-selection correlation matrix TWICE per call: once full-history (exactly
     as v10/v11, undecayed, equal-weighted) and once exponentially decayed (half-life
     BOOST_SEL_FALLBACK_HL, same weighting style as `_ewls_ridge`), both against the SAME
     BOOST_N_CANDIDATES-sized candidate pool logic (recomputed per-weighting, since "most volatile"
     is itself weighting-dependent).
  2. For each follower j: try the full-history candidate + its trailing-250-day validation IC gate,
     IDENTICAL to v10/v11. If it produces a nonzero boost, use it and stop -- v11 behavior,
     unchanged.
  3. Only if step 2 produced zero (no significant full-history candidate, or its validation IC
     failed) -- try the DECAYED candidate for that SAME follower, same Bonferroni-style significance
     test (using the decayed weighting's effective sample size) and the SAME trailing-250-day
     validation gate. If it passes, use its boost value; otherwise the follower stays at zero, same
     as it would have anyway.

VALIDATION:
  1. Real prices.txt: does the fallback path ever actually trigger? If it never does (the
     full-history path never fails validation on real 1000-day data), this file is PROVABLY
     byte-identical to v11 by construction, not just empirically -- checked directly, see
     test_v13_gated_fallback.py.
  2. Synthetic change-point rotate scenario (changepoint_synthetic.py, corrected indexing): does
     reselection speed and the resulting oracle-gap PnL actually improve over v11/v10? See
     test_v13_gated_fallback.py and README for the numbers.

BOOST_SEL_FALLBACK_HL=1000 chosen to match the ridge ensemble's own longest half-life (a reused,
not arbitrary, timescale) -- not independently swept against alternatives in this file; see
test_v13_gated_fallback.py for what was checked.
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v13 (v11 + gated decayed-selection boost fallback, hl=1000)"

# --- idio ridge + ALGO adaptive-vol leg (identical to SAFE_llboost_v10/v11.py) ---
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

# --- pairwise boost parameters (identical to SAFE_llboost_v8.py) ---
BOOST_K = 1.5
BOOST_MIN_DAY = 480
BOOST_N_CANDIDATES = 39
BOOST_IC_L = 250
BOOST_P = 2.0
BOOST_SCALE_W = 1000

# --- NEW in v13: gated decayed-selection fallback (see docstring) ---
BOOST_SEL_FALLBACK_HL = 1000

# --- idio kill switch, PnL-sum trigger (identical to SAFE_llboost_v11.py) ---
KILL_ON     = True
KILL_MARGIN = 0.0
KILL_P      = 1
ROT_W       = 60

LOOKBACK  = ROT_W + KILL_P + 6      # = 67
PRUNE_PAD = 10

_DLR = None
_PREV_ALGO_SHARES = 0
_PREV_T = -1

_SIG = {}   # n -> final traded idio forecast (50-vec) built from prices[:, :n]
_RET = {}   # n -> demeaned realized idio return over day n
_PN  = {}   # n -> traded-PnL-sign proxy: sign(_SIG[n]) . _RET[n]


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
    """Identical to SAFE_llboost_v10/v11._rank_stability_signal."""
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
    Identical to SAFE_llboost_v9/v10/v11 (unaffected by the kill switch or the boost fallback, both
    of which only touch the idio book -- this leg has its own independent forecast)."""
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


def _corrmat_weighted(X, Y, w):
    """Weighted Pearson correlation matrix, same shape convention as _corrmat."""
    sw = w.sum()
    mx = (w[None, :] * X).sum(1, keepdims=True) / sw
    my = (w[None, :] * Y).sum(1, keepdims=True) / sw
    Xc, Yc = X - mx, Y - my
    vx = (w[None, :] * Xc * Xc).sum(1) / sw; vy = (w[None, :] * Yc * Yc).sum(1) / sw
    cov = (Xc * w[None, :]) @ Yc.T / sw
    denom = np.sqrt(vx[:, None] * vy[None, :]) + 1e-12
    return cov / denom


def _leader_boost_and_ic(rs, i, j, T):
    """Given a candidate leader i for follower j, build its boost-transformed series and its
    trailing-BOOST_IC_L-day validation IC vs follower j's realized return -- the SAME validation
    logic used by both the full-history and decayed selection paths (only candidate SELECTION
    differs between the two; validation is identical, as in v10/v11)."""
    lead = rs[i]
    scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
    lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
    a = max(0, T - 1 - BOOST_IC_L)
    xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60 or xs[ok].std() < 1e-12:
        return None, None
    ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
    return lead_boost, ic


def _pairwise_boost(rs):
    """v11's full-history candidate selection + validation, UNCHANGED, plus one new gated step:
    for any follower where the full-history path ends up contributing zero (no significant
    candidate, or the selected one fails its trailing-IC validation), try a SEPARATE, exponentially
    decayed candidate search for that SAME follower only. See module docstring."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]

    # --- primary: full-history selection, identical to v10/v11 ---
    thr_full = _sig_threshold(n_samples)
    vol_causal_full = np.nanstd(Xi_full, axis=1)
    cand_idx_full = np.argsort(-vol_causal_full)[:BOOST_N_CANDIDATES]
    Xi_f = Xi_full[cand_idx_full]
    C_full = _corrmat(Xi_f, Yj)

    # --- fallback pool: exponentially-decayed selection (only consulted where primary fails) ---
    lam = 0.5 ** (1.0 / BOOST_SEL_FALLBACK_HL)
    w = lam ** np.arange(n_samples - 1, -1, -1)
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    thr_dec = _sig_threshold(max(10, int(n_eff)))
    mean_dec = np.average(Xi_full, axis=1, weights=w)
    vol_causal_dec = np.sqrt(np.average((Xi_full - mean_dec[:, None]) ** 2, axis=1, weights=w))
    cand_idx_dec = np.argsort(-vol_causal_dec)[:BOOST_N_CANDIDATES]
    Xi_d = Xi_full[cand_idx_dec]
    C_dec = _corrmat_weighted(Xi_d, Yj, w)

    for j in range(n):
        filled = False

        col = C_full[:, j].copy()
        cand_pos = np.where(cand_idx_full == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if not np.all(np.isnan(col)):
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) > thr_full:
                i = cand_idx_full[ci]
                lead_boost, ic = _leader_boost_and_ic(rs, i, j, T)
                if lead_boost is not None and ic is not None and ic > 0:
                    boost[j] = lead_boost[-1]
                    filled = True

        if filled:
            continue

        colD = C_dec[:, j].copy()
        cand_posD = np.where(cand_idx_dec == j)[0]
        if len(cand_posD):
            colD[cand_posD[0]] = np.nan
        if np.all(np.isnan(colD)):
            continue
        ciD = int(np.nanargmax(np.abs(colD)))
        if abs(colD[ciD]) <= thr_dec:
            continue
        iD = cand_idx_dec[ciD]
        lead_boost, ic = _leader_boost_and_ic(rs, iD, j, T)
        if lead_boost is None or ic is None or ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


def _idio_signal(prcSoFar):
    """The FULL, final idio forecast (ridge ensemble + beta-adjusted target, BLEND reversion,
    pairwise boost with the gated decayed fallback, rank-stability blend) -- exactly what
    getMyPosition sizes into positions. Returns a 50-vector (idio names only, ALGO excluded)."""
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

    return wz


def _ensure_cache(P):
    """Fill _SIG/_RET over the trailing LOOKBACK window, then prune stale entries. Identical to
    SAFE_llboost_v11._ensure_cache (same monotonic-t / single-panel-per-process caveat applies)."""
    T = P.shape[1]
    lo = max(WARMUP, T - LOOKBACK)
    for n in range(lo, T + 1):
        if n < WARMUP:
            continue
        if n not in _SIG:
            _SIG[n] = _idio_signal(P[:, :n])
        if n not in _RET and n < T:
            R = np.log(P[1:, n]) - np.log(P[1:, n - 1])
            _RET[n] = R - R.mean()
    cut = lo - PRUNE_PAD
    for d in (_SIG, _RET, _PN):
        for k in [k for k in d if k < cut]:
            del d[k]


def _pn1(n):
    """Traded-PnL-sign proxy for day n: sign(forecast).realized_return, summed over the 50 idio
    names (same construction as SAFE_llboost_v11._pn1)."""
    v = _PN.get(n)
    if v is None:
        v = float((np.sign(_SIG[n]) * _RET[n]).sum())
        _PN[n] = v
    return v


def _pn(lo, hi):
    return np.array([_pn1(n) for n in range(lo, hi)])


def _kill(T):
    """Identical to SAFE_llboost_v11._kill."""
    if not KILL_ON:
        return False
    for a in range(T - KILL_P, T):
        lo = a - ROT_W + 1
        if lo < WARMUP:
            return False
        pn = _pn(lo, a + 1)
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
    wz = _SIG[t]

    ready = t >= WARMUP + ROT_W + KILL_P
    killed = ready and _kill(t)
    if not killed:
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    pos[0] = _algo_vol_shares(np.log(prcSoFar)[0], cur[0], dlr[0])

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
