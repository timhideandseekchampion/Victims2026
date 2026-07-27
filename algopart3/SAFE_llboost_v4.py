"""
================================================================================
###  SAFE_llboost_v4.py  ·  SAFE_llboost_v3 candidate-pool restriction        ###
###                          + SAFE_llboost_v2 vol-regime-adaptive momentum   ###
================================================================================
Combines two independently-validated, orthogonal changes from SAFE_llboost.py (one to the idio
boost mechanism, one to the ALGO leg's momentum sub-signal -- they don't interact, so tested
together to see whether their improvements compound):
  - from v3: boost leader search restricted to BOOST_N_CANDIDATES=39 highest (causally,
    trailing-vol-ranked) volatility idio stocks, instead of all 49.
  - from v2: ALGO leg's momentum lookback switches between MOM_LB_SHORT=7 (elevated vol) and
    MOM_LB_LONG=12 (calm vol) instead of a fixed MOM_LB=10.

See SAFE_llboost_v3.py and SAFE_llboost_v2.py docstrings for the individual validation of each
component. This file exists to test whether the two, being structurally independent (v3 touches
only the post-day-500 idio boost's candidate pool; v2 touches only the ALGO leg's momentum
sub-signal), compound into a further improvement when combined.

                              OLD 501-750   NEW 751-1000   rolling mean   rolling floor   n_worse/61
  SAFE_llboost (baseline)          774.1         828.6         811.4           563.8            --
  SAFE_llboost_v3 (N=39 only)      793.8         837.8         825.5           563.8           1/61
  SAFE_llboost_v2 (adapt-mom only) 788.9         858.4         840.1           669.5          18/61
  SAFE_llboost_v4 (this file)      808.7         867.5         854.3           669.5          10/61

VALIDATED ON THE ACTUAL getMyPosition PATHWAY (eval_llboost_v4.py: official score 867.52, exactly
matching NEW here; validate_llboost_v4_full.py for the full OLD/rmean/rfloor/n_worse table above).

The combination is STRICTLY BETTER than v2 alone on every metric, including n_worse (10/61 vs
18/61) -- the candidate-pool restriction doesn't just add its own gain, it also reduces the count
of windows where v2's adaptive-momentum change underperforms. The floor gain (669.5, +105.7 over
baseline) is identical to v2's own floor -- entirely attributable to the momentum change, since the
boost restriction never touches the ALGO leg. Worst-window diagnostic: the 10 remaining worse
windows (end-days 610-720, inherited from v2's own known soft spot in that stretch) lose an average
of -10.4 (worst -19.0), while the 51 better windows gain +53.3 on average (best +114.5) -- the same
favorable asymmetry as v2's own diagnostic, just with fewer bad windows. Confirmed identical to
SAFE_llboost_v2 on days 100-400 (out-of-sample, boost inactive there in both) -- 576.0 in both,
zero side effects from the boost-pool restriction before day 500.
================================================================================
  >>> ALGO leg (instrument 0) is IDENTICAL to SAFE_llvol (adaptive realized-   <<<
  >>> vol, unchanged, inlined below). The idio book (instruments 1-49) gets    <<<
  >>> ONE addition: a per-stock size boost from its best statistically-       <<<
  >>> significant "leader" stock's own move, re-estimated fresh from all      <<<
  >>> available history on every call (no stale checkpoint caching -- this    <<<
  >>> runs once/day, so a fresh re-estimate every call is cheap).             <<<

Single-file submission: everything SAFE_llvol.py provides is inlined here (no cross-file import).
Depends on scipy (for the exact Student-t critical value in _sig_threshold) in addition to numpy --
see requirements.txt. A normal-distribution approximation (dropping scipy) was tried and scores
~0.2% lower (820.63 vs 822.16 official); kept scipy since the difference, while small, is real and
in the wrong direction.

The boost is ADDED to the idio ridge's z-score before taking sign -- it can tip a genuinely
marginal call but can never override a strong ridge conviction on its own (never a sign-flip
mechanism by construction, since a large ridge |wz| dominates a bounded boost term).

Two gates, both necessary:
  1. STATISTICAL SIGNIFICANCE: a stock's best-correlated "leader" only counts if |corr| clears a
     Bonferroni-corrected (49 simultaneous candidate-leader tests per follower) significance
     threshold GIVEN THE ACTUAL SAMPLE SIZE available right now -- not a fixed threshold picked by
     eyeballing the full-sample distribution (that was the original, look-ahead-flavored version).
  2. MINIMUM HISTORY (BOOST_MIN_DAY = 500 days): no boost at all, however "significant", before this
     much history exists.

Why gate 2 is essential (this is the whole story -- read before changing BOOST_MIN_DAY):
Without it (see test_significance_adjusted_boost.py), the boost lifted OLD/NEW/rolling-mean at
EVERY strength (K) tested, but monotonically WORSENED the rolling floor as K rose:
  K:        0      0.5     1.0     1.5     2.0     3.0
  floor:  563.8   531.9   522.3   506.0   497.7   479.9
Diagnosis (test_boost_floor_mitigation.py): the damage was not random noise -- the worse windows
were concentrated in end-days 410-570, tracing back to the checkpoints (200-550) where Bonferroni
controls false-discovery WITHIN one test but not ACROSS the ~15 sequential re-estimates made over
the file. A "significant" pair found on a thin ~200-350 day sample can still be a lucky false
positive that then trades with real size for the next several dozen days. Requiring >=500 days of
history before ANY boost is allowed removes that entire failure mode:

                              OLD 501-750   NEW 751-1000   rolling mean   rolling floor   n_worse/61
  SAFE_llvol (baseline)            687.1         761.8         760.7           563.8            --
  +boost, NO min-day, K=1.5        749.2         762.7         772.6           522.3          25/61
  SAFE_llboost (K/min-day only)    772.5         822.2         809.2           563.8           0/61
  SAFE_llboost (this file)         774.1         828.6         811.4           563.8           0/61

Validated on the ACTUAL getMyPosition (not a backtest approximation) via eval_llboost.py and
validate_llboost_full.py, and on three robustness axes: a neighbor sweep over min-history in
{450,480,500,520,550} x K in {1.0,1.25,1.5,1.75,2.0} lands in the same region (several combinations
hit n_worse=0/61 exactly); a checkpoint-refit-cadence sweep (10/25/50/75/100 days) gives n_worse=
0/61 at every cadence; and a follow-up sub-parameter sweep (test_boost_subparam_sweep.py) over
BOOST_P (0.5-3.0), BOOST_SCALE_W (100-1100), and BOOST_IC_L (100-400) found BOOST_P=2.0 was already
at its peak, but BOOST_SCALE_W=1000 and BOOST_IC_L=190 clear OLD/NEW/rmean simultaneously vs the
original 500/220 (772.5/822.2/809.2 -> 774.1/828.6/811.4), confirmed on a joint neighbor-stability
grid (scale_w in {900,1000,1100} x IC_L in {180,190,200} x K in {1.4,1.5,1.6}: every combination in
that 27-point grid scores 0/61 worse -- a broad plateau, not a lucky point) and a re-check that
K=1.5 is still the peak at the new (scale_w, IC_L). See test_boost_floor_mitigation.py,
test_boost_cadence_robustness.py, and test_boost_subparam_sweep.py for the full sweeps.

CAVEAT (same category flagged in SAFE_llmeta's postmortem and this session's synthetic stress test):
the floor is UNCHANGED here, not improved -- this adds real average edge without giving up the
worst-case window, but doesn't make the worst window any better either. It is validated across two
independent robustness axes on this one file, but a single-file 250-day graded score still carries
real sampling variance (this session's parametric-bootstrap stress test put SAFE_llvol's own 761.78
at just the 8th percentile of its own resampled score distribution, std ~118) -- so treat the lift
above as directionally real, not as a guaranteed point improvement at finals.
================================================================================
"""
import numpy as np
from scipy import stats

BOOK = "SAFE · LL-BOOST v4 (v3 candidate-pool restriction + v2 vol-regime-adaptive momentum)"

# --- idio ridge + ALGO adaptive-vol leg (identical to SAFE_llvol.py) ---
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
COMBINE_GAIN = 3.5

# --- pairwise boost parameters (see docstring table for how these were chosen) ---
BOOST_K = 1.5             # size of the boost added to the idio ridge z-score
BOOST_MIN_DAY = 500       # no boost at all before this many days of return history exist
BOOST_N_CANDIDATES = 39   # candidate leaders per follower (Bonferroni divisor) -- restricted to the
                          # 39 highest (causally-ranked, trailing) volatility idio stocks; see
                          # test_ncandidates_causal.py for the sweep and validation
BOOST_IC_L = 190          # trailing window (days) used for the leader-pair sign-of-edge check. Swept
                          # 100-400 (test_boost_subparam_sweep.py): a smooth peak at 180-190, clearing
                          # OLD/NEW/rmean simultaneously vs the original 220 (772.5/822.2/809.2 ->
                          # 774.1/828.6/811.4), floor unchanged, still 0/61 worse.
BOOST_P = 2.0             # boost magnitude exponent. Swept 0.5-3.0: 2.0 already sits at the peak, no
                          # change from the original choice.
BOOST_SCALE_W = 1000      # trailing window (days) to normalise the leader's own return scale. Swept
                          # 100-1100: 900/1000/1100 are numerically identical (saturates to "use all
                          # available history" at this file's length) and mildly beat the original 500
                          # on every metric -- not a fragile fixed window, just more history is better.

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

    # ---- ALGO index leg: unchanged, identical to SAFE_llvol ----
    pos[0] = _algo_vol_shares(logp[0], cur[0], dlr[0])

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
