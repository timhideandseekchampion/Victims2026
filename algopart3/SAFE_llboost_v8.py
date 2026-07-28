"""
================================================================================
###  SAFE_llboost_v8.py  ·  SAFE_llboost_v7 + ALGO min-conviction HOLD deadband ###
================================================================================
Everything (idio ridge+blend, the significance-gated pairwise boost, the ALGO leg's vol/momentum
signal construction and COMBINE_GAIN=16.0) is IDENTICAL to SAFE_llboost_v7.py. The ONLY change: the
ALGO leg's raw combine target is no longer resized into a small position on a near-cancellation day
-- it holds yesterday's share count instead.

Motivation (see algopart3/README.md, "v7 budget" + "ALGO deadband" sections, and
test_v7_algo_headroom.py): the ALGO leg runs at 95.1% utilisation of its $100k cap and is essentially
never flat, but the days where the raw combine target `av = COMBINE_GAIN*(sig+msig)*100_000` lands
under ~50% of the cap -- a near-cancellation of the vol-regime signal against the momentum signal --
lose money on average (-$81/day measured, vs +$188-309/day everywhere else). Those are simultaneously
low-conviction (small |av|) AND wrong-sign-prone days, not just small-position days.

THE GATE (test_v7_algo_deadband.py / test_v7_algo_deadband_v2.py):
  DEADBAND_THRESH_FRAC = 0.25  -- if |av| < 0.25 * $100k cap, HOLD yesterday's integer ALGO share
                                   count instead of resizing into the new, uncertain-sign target.
  DEADBAND_MIN_DAY     = 400   -- the gate is OFF (byte-identical to v7) before this many days of
                                   ALGO history exist, mirroring BOOST_MIN_DAY's own "don't trust an
                                   adaptive mechanism on thin history" philosophy. Without it, a joint
                                   (threshold x min_day) sweep found the SAME failure shape as
                                   BOOST_MIN_DAY's original motivation: every one of the 16 rolling
                                   windows the deadband made worse were the earliest ones (end_day
                                   400-470, before day ~400 of ALGO history), each losing ~15 points.

Joint sweep over threshold in {0.10..0.30} x min_day in {200..600} (test_v7_algo_deadband_v2.py):
32/40 configs clear OLD+NEW+rolling-mean jointly; min_day>=400 gives a clean plateau (n_worse=0/61
at min_day in {400,450,550,600} x thresh in {0.10,0.15,0.20,0.25}), not a lucky single point.
Selected thresh=0.25/min_day=400 by best rolling mean among the n_worse=0/61 configs; neighbor grid
(thresh in {0.20,0.25,0.30} x min_day in {300,400,450}) confirms it isn't an isolated spike.

                              OLD 501-750   NEW 751-1000   rolling mean   rolling floor   n_worse/61
  SAFE_llboost_v7 (shipped)        830.3         888.5         876.8           674.4           --
  SAFE_llboost_v8 (this file)      847.4         888.9         886.2           674.4          0/61

Every one of OLD/NEW/rolling-mean improves; the rolling FLOOR is UNCHANGED (674.4, to the decimal --
the deadband never touches the worst window at all), and n_worse=0/61 is as clean as v5's own
cleanest result. Validated on the actual getMyPosition pathway (`validate_llboost_v8_full.py`), not
just the backtest-equivalent reconstruction used for the sweep -- reproduces those numbers exactly
(OLD=847.4 NEW=888.9 rmean=886.2 rfloor=674.4 n_worse=0/61 vs v7). Official score via
`eval_llboost_v8.py`: **888.86** (vs v7's own 888.53).

Implementation note -- HOLD requires remembering yesterday's traded ALGO share count, which is
cross-call state (the same pattern `_limits`' `_DLR` already uses). This repo's full walk-up harnesses
(`validate_*_full.py`) call `getMyPosition` with a STRICTLY INCREASING day count from early history,
exactly like live sequential trading, so the cache builds correctly. But `eval_llboost_vN.py`'s
official-score convention calls `getPosition` ONLY over the graded test window (day 750+), skipping
the walk-up entirely -- a genuine cold start with no real "yesterday" to hold. Fixed by tracking
whether the current call is the immediate sequential successor of the last one (`tnow == _PREV_T+1`):
if not, the deadband is bypassed for that one call (computed exactly as v7 would) instead of silently
holding a fabricated 0 position, then HOLD behavior applies correctly from the next call on. Confirmed
this produces a consistent score under both harness conventions (888.86 official vs 888.9 full-walk,
the same rounding-level agreement v7 itself shows between its two conventions) -- see
`validate_llboost_v8_full.py` and `eval_llboost_v8.py`.
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v8 (v7 + ALGO min-conviction HOLD deadband, min_day=400, thresh=0.25)"

# --- idio ridge + ALGO adaptive-vol leg (identical to SAFE_llboost_v7.py) ---
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

# --- NEW in v8: ALGO min-conviction HOLD deadband (see docstring above) ---
DEADBAND_THRESH_FRAC = 0.25
DEADBAND_MIN_DAY = 400

# --- pairwise boost parameters (identical to SAFE_llboost_v7.py) ---
BOOST_K = 1.5
BOOST_MIN_DAY = 480
BOOST_N_CANDIDATES = 39
BOOST_IC_L = 250
BOOST_P = 2.0
BOOST_SCALE_W = 1000

_DLR = None
_PREV_ALGO_SHARES = 0
_PREV_T = -1


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


def _roll_std(x, w):
    """std of every length-w window of x (population, ddof=0); out[i] = std(x[i:i+w])."""
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


def _algo_vol_shares(lpA, cur0, cap_dol):
    """Adaptive realized-vol leg -> integer share target for ALGO (instrument 0). Causal.
    Identical to SAFE_llboost_v7 up to computing the raw dollar target `av`; then applies the v8
    HOLD deadband before the final int-share clip."""
    global _PREV_ALGO_SHARES, _PREV_T
    T = len(lpA)
    tnow = T - 1
    # HOLD needs a REAL "yesterday" -- only trust _PREV_ALGO_SHARES if this call is the immediate
    # sequential successor of the last one. Otherwise (a cold start: e.g. an evaluation harness that
    # only calls getMyPosition over the graded window, skipping the walk-up) fall through and
    # compute normally this one day, same "no adjustment until proven" default used elsewhere in
    # this repo (BOOST_MIN_DAY, the adaptive-BOOST_K candidate) -- never silently hold a fabricated 0.
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
        ics = [_ic_ew(feat, hl, IC_EW_W) for hl in IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
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
    """Identical to SAFE_llboost_v7._pairwise_boost."""
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


def getMyPosition(prcSoFar):
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nInst, t = prcSoFar.shape
    dlr = _limits(nInst)
    cur = prcSoFar[:, -1]
    pos = np.zeros(nInst)
    if t < WARMUP:
        return pos.astype(int)

    logp = np.log(prcSoFar)
    r = logp[:, 1:] - logp[:, :-1]

    fs = []
    for hl in HALF_LIVES:
        B, mx, my = _ewls_ridge(r[:, :-1].T, r[1:, 1:].T, hl, RIDGE_A)
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

    pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    pos[0] = _algo_vol_shares(logp[0], cur[0], dlr[0])

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
