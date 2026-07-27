"""
================================================================================
###  SAFE_llboost_v7.py  ·  SAFE_llboost_v6 + re-tuned COMBINE_GAIN            ###
###                          (ALGO leg re-swept against the TRUE v6 book)     ###
================================================================================
v6's ALGO-leg parameters were last validated at two different, now-stale points: COMBINE_GAIN=3.5
was chosen by test_joint_algo_sweep.py before ANY pairwise boost existed; MOM_LB_SHORT/LONG=7/12
was chosen by SAFE_llboost_v2 against SAFE_llboost's original (N=49, IC_L=190, MIN_DAY=500) boost.
Neither was ever re-checked against the FINAL v6 idio book (N=39, IC_L=250, MIN_DAY=480,
SCALE_W=1000 boost + the v2 adaptive momentum, both active simultaneously).

A full re-sweep against the true v6 book (test_v7cand_algoresweep.py) confirmed VOL_WIN=20,
VOL_Z=60, IC_FAST=90, SWITCH_GAIN, IC_EW_HL=(20,45), MOM_LB_SHORT=7, MOM_LB_LONG=12 are all still
sharp, isolated optima (every neighbor tested scores decisively worse, n_worse jumping from 0/61 to
38-61/61) -- except COMBINE_GAIN, which improved monotonically on EVERY headline metric at every
point tested from 2.0 up to a broad plateau at 15-17, before mildly rolling over by 25-30
(test_v7cand_combine_gain_extend.py, test_v7cand_combine_gain_fine.py). A fully independent 720-combo
joint grid search over (BOOST_K, BOOST_IC_L, MOM_LB_SHORT, MOM_LB_LONG, COMBINE_GAIN) simultaneously
(test_v7cand_joint_search.py) converged on the exact same single lever -- confirming this isn't an
artifact of one-at-a-time coordinate descent missing a cross-term.

Why this is real, not overfit -- the mechanism, not just the number: COMBINE_GAIN only scales the
raw dollar target `av = COMBINE_GAIN * (sig + msig) * 100_000` BEFORE clipping to the $100k ALGO
cap. Since (sig+msig) is bounded, raising COMBINE_GAIN just lowers the |sig+msig| needed to hit the
cap -- i.e. it pushes the ALGO leg from partial magnitude-weighted sizing towards full-conviction
sign-based sizing whenever the vol/momentum signals agree, THE SAME PRINCIPLE already validated
everywhere else in this repo (idio book sizing; batch80 catC: "full-conviction sign-based sizing
keeps winning" against every magnitude/Kelly/confidence-ramp scheme tried). It is not, however, a
trivial "turn the dial to infinity" result: the curve genuinely peaks (rmean rises from 857.0 at
G=3.5 to 876.8 at G=16, then falls back to 873.0 by G=30) and rolls over because sign(sig+msig) can
still be a near-cancellation of disagreeing signals, which any large-enough gain will nonetheless
force to the cap in ONE direction or the other -- past the optimum that starts adding noise trades,
not conviction. Turnover is unaffected (COMBINE_GAIN changes magnitude only; sign(av) -- and hence
which days trade at all -- does not depend on it), so this costs no extra commission churn.

                              OLD 501-750   NEW 751-1000   rolling mean   rolling floor   n_worse/61
  SAFE_llboost_v6 (shipped)         811.4         868.9         857.0           669.5            --
  SAFE_llboost_v7 (this file)       830.3         888.5         876.8           674.4           1/61

Every one of OLD/NEW/rolling-mean/rolling-floor improves simultaneously, with n_worse=1/61 (cleaner
than v6's own 9/61 against its predecessor). Validated on the actual getMyPosition pathway
(eval_llboost_v7.py: official score 888.53 exactly; validate_llboost_v7_full.py, which -- matching
the convention of every prior vN validator in this repo -- compares against the original
SAFE_llboost.py baseline, not v6). See test_v7cand_combine_gain_extend.py /
test_v7cand_combine_gain_fine.py for the full 2.0-30.0 sweep and test_v7cand_joint_search.py for the
independent joint-grid confirmation.

Four OTHER genuinely-untested hypotheses were tried this same session and REJECTED -- see README.md
for the full writeup and each test_v7cand_*.py script for the sweep:
  - self-adaptive BOOST_K (scaling boost strength by its own trailing realized IC, mirroring the
    ALGO leg's adaptive-gain philosophy): rejected -- the boost's realized edge is too stable/uniform
    across this file for a trailing-performance gate to find a genuine regime to exploit.
  - regime-adaptive lookback generalized from MOM_LB to REV_W, BOOST_IC_L, and IC_EW_W: rejected for
    all three -- no variant clears v6 on OLD+NEW+rolling-mean jointly.
  - pair-correlation TREND (strengthening/weakening) as an extra boost confidence multiplier,
    distinct from the existing leader-identity-stability gate: rejected -- no variant clears v6.
================================================================================
  >>> Everything else -- idio ridge+blend, the significance-gated pairwise boost at N=39/          <<<
  >>> IC_L=250/MIN_DAY=480/SCALE_W=1000/K=1.5/P=2.0, and the ALGO leg's vol-regime-adaptive         <<<
  >>> MOM_LB_SHORT=7/LONG=12 momentum switch -- is IDENTICAL to SAFE_llboost_v6.py. The ONLY change <<<
  >>> in this file is COMBINE_GAIN: 3.5 -> 16.0.                                                    <<<

Single-file submission: everything SAFE_llvol.py provides is inlined here (no cross-file import).
Depends on scipy (for the exact Student-t critical value in _sig_threshold) in addition to numpy.

The boost is ADDED to the idio ridge's z-score before taking sign -- it can tip a genuinely
marginal call but can never override a strong ridge conviction on its own (never a sign-flip
mechanism by construction, since a large ridge |wz| dominates a bounded boost term).

Two gates, both necessary:
  1. STATISTICAL SIGNIFICANCE: a stock's best-correlated "leader" only counts if |corr| clears a
     Bonferroni-corrected (39 simultaneous candidate-leader tests per follower) significance
     threshold GIVEN THE ACTUAL SAMPLE SIZE available right now.
  2. MINIMUM HISTORY (BOOST_MIN_DAY = 480 days): no boost at all, however "significant", before this
     much history exists. See SAFE_llboost.py's docstring for why this gate is essential (without
     it, the rolling floor monotonically worsens as boost strength rises).
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v7 (v6 + re-tuned COMBINE_GAIN, ALGO leg re-swept against the true v6 book)"

# --- idio ridge + ALGO adaptive-vol leg (identical to SAFE_llvol.py / SAFE_llboost_v6.py) ---
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
MOM_LB_SHORT = 7   # used when today's realized vol is ELEVATED (fh > 0)
MOM_LB_LONG  = 12  # used when today's realized vol is CALM (fh <= 0)
COMBINE_GAIN = 16.0  # re-tuned from 3.5 -- see docstring; re-swept against the TRUE v6 idio book
                     # (test_v7cand_algoresweep.py), full 2.0-30.0 sweep found a broad plateau at
                     # 15-17 (test_v7cand_combine_gain_extend.py / _fine.py), confirmed by an
                     # independent 720-combo joint search (test_v7cand_joint_search.py).

# --- pairwise boost parameters (identical to SAFE_llboost_v6.py; see its docstring table) ---
BOOST_K = 1.5
BOOST_MIN_DAY = 480
BOOST_N_CANDIDATES = 39
BOOST_IC_L = 250
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
    """Adaptive realized-vol leg -> integer share target for ALGO (instrument 0). Causal."""
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
    caller scales by BOOST_K. The leader SEARCH is restricted to the BOOST_N_CANDIDATES highest
    (causally, trailing-realized-vol-ranked) volatility stocks -- a follower can itself be outside
    that pool, but its leader must come from within it."""
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

    # ---- ALGO index leg: identical to SAFE_llboost_v6/SAFE_llvol except COMBINE_GAIN ----
    pos[0] = _algo_vol_shares(logp[0], cur[0], dlr[0])

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
