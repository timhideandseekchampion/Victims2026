"""
================================================================================
###  SAFE_llboost_v12.py  ·  SAFE_llboost_v11 (idio kill switch) + post-jump fixed-size fade  ###
================================================================================
Two independent additions on top of SAFE_llboost_v10, combined here for the first time:

1. SAFE_llboost_v11's idio kill switch (PnL-sum trigger, ROT_W=60/KILL_P=1) -- UNCHANGED, copied
   verbatim (cache plumbing, `_kill`, `_ensure_cache`, `_pn1`). See SAFE_llboost_v11.py's own
   docstring for its full motivation/validation (a defensive flatten for the "actively hostile"
   regime-reversal failure mode; roughly neutral for the "rotate" mode; 0/904 false-positive kills
   on real prices.txt).

2. NEW (found via a systematic ~50-idea signal search, this session): a post-jump fixed-size fade.
   On any idio name whose most recent daily return exceeds FADE_K_SIGMA * (trailing FADE_W-day
   realized stdev, computed strictly BEFORE that return) add a fixed-size fade against the move,
   `FADE_EXTRA_W * (-sign(that return)) * mean(|wz|)` that day -- a discrete, event-triggered
   overlay distinct from the existing CONTINUOUS 10-day reversal leg (BLEND=0.3, REV_W=10), sized
   relative to that day's own forecast magnitude rather than an absolute number. Fires on ~5% of
   all name-days -- broadly distributed, not a handful of days.

WHERE THE FADE IS WIRED IN, AND WHY: added inside `_idio_signal` (v11's own factored-out helper for
"the full traded idio forecast"), immediately after the rank-stability blend -- i.e. the LAST step
before the signal is cached/sized. This means the kill switch's own trailing-PnL trigger (`_pn1`,
built from `_SIG[n]`) automatically evaluates the PnL of the signal INCLUDING the fade, not a stale
pre-fade version -- self-consistent by construction, no separate wiring needed.

VALIDATION (real prices.txt, real getMyPosition, FADE_W=40/FADE_K_SIGMA=2.0/FADE_EXTRA_W=0.06 --
best-by-rolling-mean point in a 5x4x7=140-config neighbor grid where 56/140 configs clear the strict
OLD+NEW+rolling-mean bar against v10, a broad plateau, not an isolated spike; see
test_batch100_catHIJK_vol_event_misc.py item 4 and validate_postjumpfade_full.py for the sweep this
was selected from):

                              OLD 501-750   NEW 751-1000   rolling mean   rolling floor   n_worse/61
  SAFE_llboost_v10 (real)          871.0         912.6         909.8           709.7           --
  v10 + fade only (real)           885.8         913.8         917.3           720.7          0/61
  SAFE_llboost_v11 (real)          871.0         912.6         909.8           709.7          0/61 (identical to v10 on real
                                                                                                       data -- kill switch
                                                                                                       never fires, 0/904 days)
  SAFE_llboost_v12 (this file)     885.8         913.8         917.3           720.7          0/61

v12 == "v10 + fade" exactly on real prices.txt, confirming the kill switch is still inert here (as
v11's own docstring reports) and the two additions compose without any real-data interaction.
n_worse=0/61 against v10 across every metric simultaneously -- clean.

Positions differ from v10 on only 40/852 graded-eligible days; NEW-window commission moves by only
+$12 (v10: $11,624 -> v12: $11,636) -- the fade is a small, targeted intervention, not a rewrite.

HONEST CAVEATS, stated plainly rather than buried:
  - The fade's headline GRADED-WINDOW gain (NEW) is modest, +1.2 -- the larger gains are OLD (+14.8)
    and the rolling floor (+11.0). Real and useful, not a blowout.
  - The kill switch's own caveats (from v11) still apply unchanged: first-pass, only lightly tuned,
    helps the reverse failure mode and is roughly neutral for rotate. Combining it with the fade does
    not fix or worsen either of those properties -- they are genuinely orthogonal mechanisms (one
    fires on trailing realized PnL sign at the whole-book level, the other on same-day per-name
    return magnitude), and this file does not re-validate v11's synthetic change-point results (see
    SAFE_llboost_v11.py / README for those) -- only that the two compose cleanly on real data.
  - The fade was found and swept via a backtest-approximation harness (_v10_harness.py,
    _v11-equivalent since the kill switch is inert on real data) and independently re-confirmed
    through the real, sequential getMyPosition pathway (validate_postjumpfade_full.py) before being
    wired in here -- not shipped on sweep numbers alone.
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v12 (v11 idio kill switch + post-jump fade W=40/k=2.0/extra_w=0.06)"

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

# --- idio kill switch, PnL-sum trigger (identical to SAFE_llboost_v11.py) ---
KILL_ON     = True
KILL_MARGIN = 0.0
KILL_P      = 1
ROT_W       = 60

LOOKBACK  = ROT_W + KILL_P + 6      # = 67
PRUNE_PAD = 10

# --- NEW in v12: post-jump fixed-size fade (see docstring above) ---
FADE_W        = 40
FADE_K_SIGMA  = 2.0
FADE_EXTRA_W  = 0.06

_DLR = None
_PREV_ALGO_SHARES = 0
_PREV_T = -1

_SIG = {}   # n -> final traded idio forecast (50-vec, incl. fade) built from prices[:, :n]
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
    Identical to SAFE_llboost_v9/v10/v11 (unaffected by the kill switch or the fade, both of which
    only touch the idio book -- this leg has its own independent forecast)."""
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
    """Identical to SAFE_llboost_v9/v10/v11._pairwise_boost."""
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
    """The FULL, final idio forecast: ridge ensemble + beta-adjusted target, BLEND reversion,
    pairwise boost, rank-stability blend (all identical to SAFE_llboost_v10/v11), PLUS the new
    post-jump fixed-size fade (NEW in v12, applied last -- see module docstring for why this
    placement keeps the kill switch's PnL-sign trigger self-consistent). Returns a 50-vector (idio
    names only, ALGO excluded)."""
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
    names (same construction as SAFE_llboost_v11._pn1) -- `_SIG[n]` here already includes the fade,
    so this proxy reflects the ACTUAL traded signal, not a stale pre-fade one."""
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
