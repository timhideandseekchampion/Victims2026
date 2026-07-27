"""
================================================================================
###  SAFE_llboost_v2.py  ·  SAFE_llboost + vol-regime-adaptive momentum       ###
###                         lookback (EXPERIMENTAL -- see caveat below)      ###
================================================================================
Identical to SAFE_llboost.py (idio ridge + significance-gated pairwise boost, unchanged) EXCEPT
for one change to the ALGO leg: the momentum-switch lookback (MOM_LB, fixed at 10 in SAFE_llboost)
now SWITCHES between a SHORT window (7 days) when today's realized vol is ELEVATED and a LONG
window (12 days) when it's CALM, instead of using the same fixed lookback always.

Motivation: momentum decaying faster in high-vol/high-information-flow regimes and persisting
longer in calm regimes is a plausible, common pattern in practice; SAFE_llboost never tested this
because MOM_LB was swept only as a single fixed constant (found to sit at an isolated optimum at
10 -- see SAFE_llboost.py's docstring and test_mom_lb_fine.py). This file tests making the window
itself regime-dependent rather than fixed.

VALIDATED ON THE ACTUAL getMyPosition PATHWAY (test_vol_adaptive_validate.py -- not just a backtest
approximation; sanity-checked to reproduce SAFE_llboost's own 828.60 official score for the
baseline before trusting the modified version):

                              OLD 501-750   NEW 751-1000   rolling mean   rolling floor   n_worse/61
  SAFE_llboost (baseline)          774.1         828.6         811.4           563.8            --
  SAFE_llboost_v2 (this file)      788.9         858.4         840.1           669.5          18/61

All four headline metrics improve substantially -- the floor jump (+105.7) is the largest single
improvement found in the whole investigation. Stable across a real parameter neighborhood too: the
short=7 lookback works well paired with every long lookback tested from 11 to 16 (test_adaptive_
mom_lb.py), not just this one specific pair -- not a lucky isolated point.

CAVEAT -- WHY THIS IS v2, NOT A REPLACEMENT FOR SAFE_llboost.py:
Unlike every parameter validated in SAFE_llboost.py itself, this does NOT clear n_worse=0/61 --
18 of 61 rolling windows are worse. The window-concentration diagnostic (test_adaptive_mom_lb.py)
found a REASSURING but not perfect shape: the 18 worse windows lose an average of only -9.8 (worst
case -19.4), while the 43 better windows gain an average of +44.8 (up to +114.5) -- a favorable
asymmetry, structurally different from every other candidate rejected this session (which typically
traded one metric for another, e.g. the top-2/3-leader idea's ~41-point NEW regression, or showed
symmetric/unfavorable loss magnitudes). The worse windows cluster mildly around days 610-670, but
the losses there are small, not the single-catastrophic-window pattern documented in SAFE_llmeta's
overfitting postmortem (README.md).

Bottom line: this is a genuinely promising result, not a clean pass. Kept as a separate file rather
than replacing SAFE_llboost.py so both can be compared/run side by side before deciding which to
actually submit.
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v2 (llboost + vol-regime-adaptive momentum lookback, EXPERIMENTAL)"

# --- idio ridge + ALGO adaptive-vol leg (identical to SAFE_llboost.py) ---
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
# MOM_LB is now regime-dependent instead of a single fixed constant (see _algo_vol_shares below):
MOM_LB_SHORT = 7   # used when today's realized vol is ELEVATED (fh > 0)
MOM_LB_LONG  = 12  # used when today's realized vol is CALM (fh <= 0)
COMBINE_GAIN = 3.5

# --- pairwise boost parameters (identical to SAFE_llboost.py, unchanged) ---
BOOST_K = 1.5
BOOST_MIN_DAY = 500
BOOST_N_CANDIDATES = 49
BOOST_IC_L = 190
BOOST_P = 2.0
BOOST_SCALE_W = 1000

_DLR = None


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
    Only change from SAFE_llboost.py: the momentum lookback switches between MOM_LB_SHORT
    (elevated vol) and MOM_LB_LONG (calm vol) instead of using a single fixed window."""
    T = len(lpA)
    if T < VOL_WIN + VOL_Z + 60:
        return 0
    r = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[VOL_WIN:] = _roll_std(r, VOL_WIN)
    tnow = T - 1
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
        return 0
    if VOL_MODE == "switch":
        sig = _side(volz, fh)
        if sig is None: return 0
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
        if ic is None: return 0
        av = VOL_GAIN * max(0.0, ic) * fh * 100_000.0
    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


BOOST_ALPHA = 0.05  # significance level, Bonferroni-corrected for BOOST_N_CANDIDATES simultaneous tests


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
    """rs: (49, T) idio-stock return matrix (ALGO excluded). Returns a length-49 array of today's
    raw boost value per stock (0.0 where no significant, min-history-qualified leader exists);
    caller scales by BOOST_K."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < BOOST_MIN_DAY:
        return boost
    Xi = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi.shape[1]
    thr = _sig_threshold(n_samples)
    C = _corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr:
            continue
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
        boost[j] = lead_boost[-1]  # today's boost value
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

    # ---- idio leg: ridge+blend forecast, plus the significance-gated pairwise boost ----
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

    # ---- ALGO index leg: vol-regime-adaptive momentum lookback (the only change vs SAFE_llboost) ----
    pos[0] = _algo_vol_shares(logp[0], cur[0], dlr[0])

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
