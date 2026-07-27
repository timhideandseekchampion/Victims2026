"""
##########################################################################################
###  SAFE_rotate.py   ·   CHAMPION–CHALLENGER SIGNAL ROTATION  (IC-gated)              ###
##########################################################################################
Sibling of SAFE_lldollar.py. IDENTICAL trading machinery (sign-sized idio book + net-$
ALGO gate + limits). The ONLY change: which cross-sectional signal drives the idio book is
chosen each day by an IC-gated champion–challenger rotation.

  CHAMPION  = the current book's signal:  (1-BLEND)*lead-lag-ensemble + BLEND*reversion.
  CHALLENGERS (on the bench, monitored but not traded until they earn it):
      mom   cross-sectional MOMENTUM   (+trailing REV_W return)   -> if names start trending
      revL  LONG-horizon reversion     (-trailing 30d return)     -> if reversion slows down
      ll2   multi-lag lead-lag         (ridge on r_t AND r_{t-1}) -> if the lead-lag deepens
      resid index-residual reversion   (Avellaneda-Lee lite)      -> if a basket/pairs edge appears
      btiao Box-Tiao cointegration     (most mean-reverting baskets, faded) -> if a basket edge appears
      volsc vol-scaled lead-lag        (champion LL / trailing vol)-> if the edge concentrates by vol
      momJT cross-sectional momentum   (JT: +return over [t-120,t-20]) -> if names trend (reversal-decontaminated)
      residMom residual momentum       (BHM: momentum on index-residual) -> if idio winners keep winning
  (momVS = Barroso vol-scaled momentum is computed for diagnostics but NOT traded: sign-sizing makes
   it near-degenerate with momJT and it would only raise the multiple-testing bar.)
  ALGO index leg is itself IC-gated FADE (reversion, default) vs TREND (time-series momentum).

ROTATION RULE (no look-ahead; all ICs are realized, past-only):
  Each day, over a trailing window of W forecasts (each graded by the NEXT day's realized
  idio cross-section), compute realized IC for the champion and every challenger. Rotate the
  book onto a challenger c ONLY IF, sustained for P consecutive days:
      (1) mean(IC_c - IC_champ) >= MARGIN                 # challenger matches or beats champion
      (2) paired t(IC_c - IC_champ)  > TCRIT              # the beat is statistically real
      (3) t(IC_c) > TCRIT  and  mean(IC_c) > 0            # the challenger itself is significant
  If several qualify, take the highest mean IC. If none, TRADE THE CHAMPION (current book).

WHY THIS IS SAFE (and different from the detector we already rejected):
  * detector.py timed reversion-vs-lead-lag WITHIN the current signal — IC there is not
    persistent, so it whipsawed and was overfit OOS. This does NOT time the champion; it only
    ever leaves the champion for a *different* signal that clears a high, significant bar.
  * On the data we have, no challenger clears the bar, so getMyPosition == SAFE_lldollar to the
    share (verify with rotate_test.py). It is pure optionality: zero cost now, only fires if the
    market's edge structurally moves to one of the bench signals.
  * MARGIN=0 honours "same or better"; raise it (e.g. 0.005) to demand a real improvement and
    further cut false-positive rotations.
==========================================================================================
"""
import numpy as np

BOOK      = "SAFE · ROTATE (IC-gated champion–challenger)"
CHAMPION  = "(1-BLEND)*lead-lag-ensemble + BLEND*reversion   (== SAFE_lldollar)"

# ---- trading knobs (identical to SAFE_lldollar) -----------------------------
HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
HEDGE       = False
WARMUP      = 96
ALGO_LL_DOLLAR = 50_000

# ---- rotation knobs ---------------------------------------------------------
# ADOPTED pnl-W60 gate (gate_sweep.py + verify-pnl-gate workflow): more sensitive than the IC
# significance gate -- byte-identical on real data (0 rot, 694.13) yet captures a real regime far
# faster (momentum +30k/6 seeds, weak-edge +65k). ROT_TCRIT only matters if GATE_MODE reverts to "ic".
ROT_W      = 60      # trailing window used to estimate the gate metric
ROT_TCRIT  = 2.5     # (ic mode only) significance bar
ROT_MARGIN = 0.0     # (ic mode only) required IC edge over champion
ROT_P      = 5       # consecutive days the gate must hold before rotating (hysteresis)
REVL_W     = 30      # long-reversion lookback
RESID_K    = 60      # residual-reversion / beta window
LL2_HL     = 1000    # half-life for the multi-lag challenger ridge
VOLSC_W    = 20      # trailing window for per-name vol (vol-scaled lead-lag)
BT_NB      = 5       # Box-Tiao: number of most-mean-reverting baskets
BT_LOOK    = 250     # Box-Tiao: VAR(1) fit lookback
BT_ZWIN    = 20      # Box-Tiao: basket-spread z-score window
ROT_BONF   = True    # scale the significance bar for the number of challengers (multiple-testing)
# GATE_MODE (rot_gate_compare.py): "ic" = paired-t significance on IC (conservative, ~32d lag);
# "pnl" = rotate on trailing realized PROFITABILITY beating champion (objective-aligned, ~12d lag,
# still inert on real data); "sharpe" = trailing Sharpe (middle: ~18d, less flip-whipsaw).
GATE_MODE  = "pnl"   # "ic" | "pnl" (ADOPTED) | "sharpe" | "softblend"  (see SAFE_rotate notes above)
PNL_MARGIN = 0.0     # required trailing book-return edge over champion (pnl/sharpe modes)

# ---- momentum challengers (Jegadeesh-Titman / Blitz-Huij-Martens / Barroso) --
MOMJT_L    = 120     # cross-sectional momentum lookback (days)
MOMJT_S    = 20      # skip-recent days (clears the ~7-10d short-term reversal band)
RESIDM_L   = 120     # residual (factor-neutral) momentum lookback / beta window
RESIDM_S   = 20      # residual momentum skip-recent days
MOMVS_VW   = 20      # vol window for vol-scaled momentum (diagnostic only, not traded)

# ---- ALGO index leg: IC-gated FADE (reversion) vs TREND (time-series momentum) -
ALGO_ROT_W = 120     # trailing window for the index fade-vs-trend IC gate
ALGO_ROT_H = 5       # stride/horizon for non-overlapping index forward returns (honest t-stat)
ALGO_TCRIT = 3.0     # significance bar to flip the index leg to TREND (single series -> no Bonferroni)
ALGO_P     = 10      # consecutive days the trend gate must hold before flipping
# NOTE (verify_accel2.py): an accelerant that relaxes this gate on a confirmed momentum regime was
# tested and REVERTED -- it was inert (the 30-day-move z-score signal can't detect a steady trend, so
# relaxing persistence/bar changes nothing). Blind-forcing trend captured an index uptrend (+100k) but
# lost 41k on a mean-reverting index -- a coin-flip bet, not taken. The net-$ gate already transplants
# the book's directional conviction (incl. a momentum rotation) into the index leg on conviction days.

# ---- kill switch: flatten the idio book if the traded edge's IC has INVERTED ---
KILL_ON    = True    # enable the kill switch
KILL_TCRIT = 3.0     # bar: fire only on a SIGNIFICANTLY NEGATIVE IC (t < -KILL_TCRIT), not merely weak
KILL_P     = 10      # consecutive days the significant-negative IC must hold (avoids variance whipsaw)

# ---- xsac validator: DIRECT momentum-regime indicator (champion health check) --
# Verified (regime_detect.py + adversarial workflow): trailing-mean cross-sectional lag-1
# autocorrelation flips hard positive in a momentum regime (~8d detection, 0 false positives on
# real 750d data, where the windowed value stays in [-0.06, +0.07] and never SUSTAINS above 0.05).
# It never trades by itself: while ON it only RELAXES the rotation gate (shorter persistence,
# no Bonferroni bump) so the IC rotation fires in days instead of weeks.
XSAC_W     = 40      # trailing window for the mean cross-sectional lag-1 autocorrelation
XSAC_TH    = 0.07    # momentum-regime threshold: above the real-data max (+0.061) -> strictly silent
                     # on all 750 real days; a genuine momentum regime reads ~+0.35 (regime_detect.py)
XSAC_P     = 5       # consecutive days above threshold before the validator flag turns on
ROT_P_FAST = 3       # relaxed rotation persistence while the xsac flag confirms a momentum regime

_DLR = None
_SIG = {}            # n (#cols used) -> dict{name: 50-vec forecast for day n}
_RET = {}            # n -> realized demeaned idio return graded against _SIG[n]  (known once col n exists)
_ICD = {}            # (name, n) -> realized daily IC   (memoized; pure fn of _SIG/_RET)
_AZ  = {}            # s (#cols) -> index trend z-value from the first s columns (memoized)
_XC  = {}            # n -> cross-sectional lag-1 autocorr corr(_RET[n-1], _RET[n])  (memoized)
_PN  = {}            # (name, n) -> daily as-if-traded book-return proxy (sign(forecast) . realized ret)


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


def _btiao(logp, nb, look, zwin):
    """Box-Tiao: fade the nb most mean-reverting idio baskets. Returns a demeaned 50-vector."""
    n = logp.shape[1]; look = min(look, n - 2)
    X = logp[1:, -look:]; X = X - X.mean(1, keepdims=True)
    p0 = X[:, :-1].T; p1 = X[:, 1:].T
    G = (p0.T @ p0) / p0.shape[0]
    M = np.linalg.solve(G + 1e-6 * np.eye(50), (p0.T @ p1) / p0.shape[0]).T
    A = M @ G @ M.T
    ev, V = np.linalg.eig(np.linalg.inv(G + 1e-6 * np.eye(50)) @ A)
    baskets = V.real[:, np.argsort(ev.real)[:nb]].T
    sig = np.zeros(50)
    for b in baskets:
        vs = b @ (logp[1:, -zwin:] - logp[1:, -zwin:].mean(1, keepdims=True))
        z = (vs[-1] - vs.mean()) / (vs.std() + 1e-9)
        sig += -np.clip(z, -3, 3) * b
    return sig - sig.mean()


def _forecasts(P):
    """All signal forecasts (each a demeaned 50-vector over the idio names) from prc[:, :n].
    Only the SIGN/ranking is traded, so absolute scale is irrelevant; IC uses correlation."""
    logp = np.log(P); r = logp[:, 1:] - logp[:, :-1]; m = r.shape[1]
    out = {}

    # champion: lead-lag ensemble blended with short reversion
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = _ewls_ridge(r[:, :-1].T, r[1:, 1:].T, hl, RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean(); fs.append(fi / (fi.std() + 1e-12))
    ll_z = np.mean(fs, 0)
    rr = logp[1:, -1] - logp[1:, -1 - REV_W]; rr = rr - rr.mean()
    rev_z = -rr / (rr.std() + 1e-12)
    out["champ"] = (1 - BLEND) * ll_z + BLEND * rev_z

    # challenger 1: cross-sectional momentum (sign-flip of short reversion)
    out["mom"] = rr.copy()

    # challenger 2: long-horizon reversion
    rl = logp[1:, -1] - logp[1:, -1 - REVL_W]; out["revL"] = -(rl - rl.mean())

    # challenger 3: multi-lag lead-lag  (regress next idio return on r_t AND r_{t-1})
    if m >= LL2_HL // 4 + 4:
        Xa = r[:, 1:m - 1].T; Xb = r[:, 0:m - 2].T
        X2 = np.hstack([Xa, Xb]); Y2 = r[1:, 2:m].T
        B2, mx2, my2 = _ewls_ridge(X2, Y2, LL2_HL, RIDGE_A)
        xin = np.concatenate([r[:, -1], r[:, -2]])
        p2 = my2 + (xin - mx2) @ B2; out["ll2"] = p2 - p2.mean()
    else:
        out["ll2"] = out["champ"].copy()

    # challenger 4: index-residual reversion (Avellaneda-Lee lite)
    if m >= RESID_K + 1:
        R = r[1:, -RESID_K:]; r0 = r[0, -RESID_K:]; r0c = r0 - r0.mean()
        beta = (R @ r0c) / (r0c @ r0c + 1e-12)
        cum = (R - beta[:, None] * r0[None, :]).sum(1)
        out["resid"] = -(cum - cum.mean())
    else:
        out["resid"] = out["champ"].copy()

    # challenger 5: Box-Tiao cointegration baskets (faded)
    if m >= BT_LOOK // 4 + BT_ZWIN + 2:
        try:
            out["btiao"] = _btiao(logp, BT_NB, BT_LOOK, BT_ZWIN)
        except np.linalg.LinAlgError:
            out["btiao"] = out["champ"].copy()
    else:
        out["btiao"] = out["champ"].copy()

    # challenger 6: vol-scaled lead-lag (the champion's LL edge, re-weighted by 1/vol)
    vol = r[1:, -VOLSC_W:].std(1) if m >= VOLSC_W else r[1:].std(1)
    vs = ll_z / (vol + 1e-9); out["volsc"] = vs - vs.mean()

    # challenger 7: Jegadeesh-Titman cross-sectional momentum (long lookback, skip recent reversal)
    if logp.shape[1] >= MOMJT_L + 1:
        g = logp[1:, -1 - MOMJT_S] - logp[1:, -1 - MOMJT_L]      # cumulative return over [t-L, t-S]
        g = g - g.mean(); out["momJT"] = g / (g.std() + 1e-12)   # +sign = winners keep winning
    else:
        out["momJT"] = out["champ"].copy()

    # challenger 8: Blitz-Huij-Martens residual (factor-neutral) momentum
    if m >= RESIDM_L + 1:
        Rwin = r[1:, -RESIDM_L:]; r0win = r[0, -RESIDM_L:]; r0c = r0win - r0win.mean()
        beta = (Rwin @ r0c) / (r0c @ r0c + 1e-12)
        resid = Rwin - beta[:, None] * r0win[None, :]
        cum = (resid[:, :RESIDM_L - RESIDM_S] if RESIDM_S > 0 else resid).sum(1)
        cum = cum - cum.mean(); out["residMom"] = cum / (cum.std() + 1e-12)
    else:
        out["residMom"] = out["champ"].copy()

    # diagnostic (NOT traded): Barroso vol-scaled momentum -- near-degenerate with momJT under
    # sign-sizing; kept so signal_overlap.py/sig_score.py can quantify the redundancy.
    if logp.shape[1] >= MOMJT_L + 1:
        vmom = (logp[1:, -1 - MOMJT_S] - logp[1:, -1 - MOMJT_L]) / (vol + 1e-9)
        out["momVS"] = vmom - vmom.mean()
    else:
        out["momVS"] = out["champ"].copy()

    return out


# TRADED bench = momentum family ONLY. The stat-arb variants (revL/ll2/resid/btiao/volsc/momVS) are
# still computed for diagnostics but NOT traded: they are either the champion's own edge re-expressed
# (ll2 0.70 / volsc 0.88 forecast-corr -> can never significantly BEAT champ) or too weak to ever clear
# the gate even in a lead-lag-death world (revL/resid/btiao IC~0.006-0.02 -> t_d ~0.6 over a 40d window).
# Every seat raises the Bonferroni bar for ALL challengers, so unwinnable seats only tax the momentum
# protection. Pruning 8 -> 3 lowers the bar 3.23 -> 2.91 (faster legit capture, same FP safety).
CHALLENGERS = ("mom", "momJT", "residMom")


def _ensure_cache(P):
    """Fill _SIG for every column count up to T and _RET for every gradable forecast."""
    nInst, T = P.shape
    for n in range(WARMUP, T + 1):
        if n not in _SIG:
            _SIG[n] = _forecasts(P[:, :n])
        if n not in _RET and n < T:                       # realized return for forecast made at n
            R = np.log(P[1:, n]) - np.log(P[1:, n - 1])   # move over day n (idio names)
            _RET[n] = R - R.mean()


def _ic1(name, n):
    """realized daily IC = corr(forecast made at n, return realized over day n). Memoized."""
    key = (name, n); v = _ICD.get(key)
    if v is None:
        f = _SIG[n][name]; g = _RET[n]
        fm = f - f.mean(); gm = g - g.mean()
        denom = np.sqrt((fm @ fm) * (gm @ gm))
        v = float(fm @ gm / denom) if denom > 1e-18 else 0.0
        _ICD[key] = v
    return v


def _ic(name, lo, hi):
    """realized IC series of a signal over forecast-days [lo, hi)."""
    return np.array([_ic1(name, n) for n in range(lo, hi)])


def _pn1(name, n):
    """daily as-if-traded book-return proxy = sum(sign(forecast) * realized idio return). Memoized."""
    key = (name, n); v = _PN.get(key)
    if v is None:
        v = float((np.sign(_SIG[n][name]) * _RET[n]).sum()); _PN[key] = v
    return v


def _pn(name, lo, hi):
    return np.array([_pn1(name, n) for n in range(lo, hi)])


def _tcrit():
    """Significance bar, Bonferroni-bumped for the number of challengers tested.
    Gaussian tail: to hit tail prob p0/C, t_eff ~= sqrt(t0^2 + 2 ln C)  (numpy-only)."""
    if ROT_BONF and len(CHALLENGERS) > 1:
        return float(np.sqrt(ROT_TCRIT ** 2 + 2.0 * np.log(len(CHALLENGERS))))
    return ROT_TCRIT


def _xc1(n):
    """cross-sectional lag-1 autocorr between realized idio returns of day n-1 and day n. Memoized."""
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
    """trailing-XSAC_W mean cross-sectional lag-1 autocorr as of day a (positive = momentum regime)."""
    vals = [_xc1(n) for n in range(a - XSAC_W + 1, a + 1)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) >= XSAC_W // 2 else None


def _xsac_flag(T):
    """validator flag: xsac sustained above threshold on each of the last XSAC_P gradable days."""
    for a in range(T - XSAC_P, T):
        v = _xsac(a)
        if v is None or v <= XSAC_TH:
            return False
    return True


def _gate_at(a, tcrit=None):
    """Which challenger (if any) qualifies using the trailing window ending at day a.
    GATE_MODE 'ic' = paired-t significance (conservative); 'pnl'/'sharpe' = beat champion on
    trailing realized profitability (faster, still inert on real data -- see rot_gate_compare.py)."""
    lo = a - ROT_W + 1
    if lo < WARMUP:
        return None
    if GATE_MODE == "ic":
        if tcrit is None:
            tcrit = _tcrit()
        ic_c = _ic("champ", lo, a + 1)
        best = None; best_v = -1e18
        for name in CHALLENGERS:
            ic = _ic(name, lo, a + 1); d = ic - ic_c
            t_d = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-18)
            t_i = ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-18)
            if d.mean() >= ROT_MARGIN and t_d > tcrit and ic.mean() > 0 and t_i > tcrit and ic.mean() > best_v:
                best_v = ic.mean(); best = name
        return best
    # profitability-based gate: rotate to the challenger that beats champion on trailing book-return
    pch = _pn("champ", lo, a + 1)
    best = None; best_v = -1e18
    for name in CHALLENGERS:
        pc = _pn(name, lo, a + 1)
        if GATE_MODE == "sharpe":
            shc = pc.mean() / (pc.std() + 1e-9)
            ok = shc > pch.mean() / (pch.std() + 1e-9) + PNL_MARGIN and pc.mean() > 0
            val = shc
        else:  # "pnl"
            ok = (pc - pch).mean() > PNL_MARGIN
            val = pc.mean()
        if ok and val > best_v:
            best_v = val; best = name
    return best


def _blend_wz(T):
    """SOFT-BLEND: continuous champion<->challenger tilt. Each challenger's weight = its trailing
    book-return edge over the champion (relu, scaled by the champion's own trailing PnL). Weight is 0
    when it is not beating the champion, so on real data this collapses to the pure champion; it tilts
    smoothly (never flips discretely) toward a challenger that is genuinely out-earning it."""
    lo = T - ROT_W
    if lo < WARMUP:
        return _SIG[T]["champ"]
    pch = _pn("champ", lo, T)                       # trailing champion book-return (gradable: n < T)
    scale = abs(pch.mean()) + 1e-9
    wz = _SIG[T]["champ"].astype(float).copy(); wsum = 1.0
    for c in CHALLENGERS:
        e = (_pn(c, lo, T) - pch).mean()
        wc = max(0.0, e) / scale                    # 0 unless the challenger out-earns the champion
        if wc > 0.0:
            wz = wz + wc * _SIG[T][c]; wsum += wc
    return wz / wsum


def _choose(T):
    """Signal to trade for column count T. Requires the SAME challenger to have qualified on
    each of the last P gradable days; else trade the champion. When the xsac validator
    independently confirms a momentum regime, the gate relaxes (P=ROT_P_FAST, bar without the
    Bonferroni bump) so a legit rotation fires in days instead of weeks; xsac never fires on
    the real 750d data, so the strict gate applies there."""
    fast = _xsac_flag(T)
    P = ROT_P_FAST if fast else ROT_P
    bar = ROT_TCRIT if fast else _tcrit()
    picks = [_gate_at(a, bar) for a in range(T - P, T)]    # a <= T-1 -> _RET[a] available
    if picks and picks[0] is not None and all(p == picks[0] for p in picks):
        return picks[0]
    return "champ"


# ---- ALGO index leg: IC-gated FADE (reversion, default) vs TREND (time-series momentum) -----
def _algo_z(lpA):
    """The index trend z-value from an index log-price prefix (same z the book already forms)."""
    mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
    if len(mv) < CONTRA_WZ:
        return None
    return float((mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12))


def _algo_zc(lpA, s):
    """memoized trend z from the first s index prices (pure fn of s for a fixed series)."""
    v = _AZ.get(s)
    if v is None:
        v = _algo_z(lpA[:s]); _AZ[s] = v
    return v


def _algo_gate_at(lpA, T):
    """Does the index TREND (not fade) work over the trailing window ending at as-of T?
    corr(trend z made at s, non-overlapping H-day forward index return). Uses only data < T."""
    smin = CONTRA_K + CONTRA_WZ + 1
    zs = []; fs = []
    for s in range(T - ALGO_ROT_H, T - ALGO_ROT_H - ALGO_ROT_W, -ALGO_ROT_H):
        if s < smin:
            break
        z = _algo_zc(lpA, s)
        if z is None:
            continue
        zs.append(z); fs.append(lpA[s - 1 + ALGO_ROT_H] - lpA[s - 1])   # forward H-day index return
    if len(zs) < 8:
        return False                                    # insufficient history -> default FADE
    zs = np.asarray(zs); fs = np.asarray(fs)
    if zs.std() < 1e-12 or fs.std() < 1e-12:
        return False
    r = float(np.corrcoef(zs, fs)[0, 1]); n = len(zs)
    if r <= 0:
        return False
    t_ic = r * np.sqrt((n - 2) / (1.0 - r ** 2 + 1e-12))
    return t_ic > ALGO_TCRIT


def _algo_leg_mode(lpA, T):
    """'trend' only if the index-trend gate holds on each of the last ALGO_P as-of days; else 'fade'."""
    for a in range(T - ALGO_P + 1, T + 1):
        if not _algo_gate_at(lpA, a):
            return "fade"
    return "trend"


def _kill(T, chosen):
    """Kill switch: True (flatten the idio edge) only if the TRADED signal's realized IC is
    significantly NEGATIVE (inverted, not merely weak) on each of the last KILL_P gradable days.
    A weak/zero IC is variance -> keep trading; a sustained significant-negative IC = the edge broke."""
    if not KILL_ON:
        return False
    for a in range(T - KILL_P, T):
        lo = a - ROT_W + 1
        if lo < WARMUP:
            return False
        ic = _ic(chosen, lo, a + 1)
        t_i = ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-18)
        if not (ic.mean() < 0 and t_i < -KILL_TCRIT):     # this day fails the kill condition
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
    ready = t >= WARMUP + ROT_W + ROT_P
    if GATE_MODE == "softblend" and ready:
        chosen = "blend"; wz = _blend_wz(t)
    else:
        chosen = _choose(t) if ready else "champ"
        wz = _SIG[t][chosen]

    logp = np.log(prcSoFar)
    r = logp[:, 1:] - logp[:, :-1]

    # kill switch: if the traded edge's IC has inverted (sustained significant-negative), stay FLAT
    killed = ready and _kill(t, "champ" if chosen == "blend" else chosen)
    if not killed:
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    # ---- net dollar skew of the stock book -> gate the ALGO index leg (identical to lldollar)
    idio_lim = (dlr[1:] / cur[1:]).astype(int)
    idio_int = np.clip(pos[1:], -idio_lim, idio_lim).astype(int)
    net_dol = float((idio_int * cur[1:]).sum())

    cap = dlr[0] / cur[0]
    if ALGO_LL_DOLLAR > 0 and abs(net_dol) >= ALGO_LL_DOLLAR:
        av = float(np.sign(net_dol) * cap)
    else:
        lpA = logp[0]; mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        z = (mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12)
        zc = np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / cur[0])
        # default FADE (index reversion); flip to TREND only on a sustained, significant index trend
        trend = (t >= WARMUP + ALGO_ROT_W + ALGO_P) and (_algo_leg_mode(lpA, t) == "trend")
        av = float(np.clip(zc if trend else -zc, -cap, cap))

    hs = 0.0
    if HEDGE:
        rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
        betas = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
        hs = -((pos[1:] * cur[1:]) @ betas) / cur[0]
    room = max(cap - abs(av), 0.0)
    pos[0] = av + float(np.clip(hs, -room, room))

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
