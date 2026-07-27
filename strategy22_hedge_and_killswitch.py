"""
##########################################################################################
###  SAFE_live.py  ·  SUBMISSION BUILD of SAFE_rotate  (lean runtime, identical trades) ###
##########################################################################################
Self-contained tournament build. POSITION-IDENTICAL to SAFE_rotate.py (verified by
live_test.py: 0-share diff on all real days AND on the injected momentum regime, official
score 694.13 unchanged). Two runtime differences only — neither can change a position:

  1. LEAN FORECASTS: computes only the signals the gates read (champ + mom/momJT/residMom).
     The six diagnostic signals (revL/ll2/resid/btiao/volsc/momVS) are research-only and
     never traded — dropped here. ll2's double-size ridge was the single largest cost.
  2. WINDOWED CACHE: forecasts/ICs are only ever needed over the trailing
     ROT_W + max(KILL_P, ROT_P) days, so the cache fills (and prunes) a sliding window
     instead of all history. First call is seconds, not minutes, at any history length.

Protection stack (all verified in the SAFE_rotate research build):
  * champion = lead-lag ensemble + reversion blend (== SAFE_lldollar, the main edge)
  * momentum challengers mom/momJT/residMom — IC-gated rotation (Balanced: W=40,P=7,t=2.5)
  * xsac validator — direct cross-sectional autocorr; relaxes the gate in a momentum regime
  * kill switch — flattens the idio book if the traded edge's IC INVERTS (sustained t<-3)
  * ALGO leg — net-$ gate; else IC-gated FADE (default) vs TREND (index TSMOM)
==========================================================================================
"""
import numpy as np

BOOK = "SAFE · LIVE aggressive dynamic hedge diagnostics"

# ---- trading knobs (identical to SAFE_lldollar) -----------------------------
HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
WARMUP      = 96
ALGO_LL_DOLLAR = 50_000

# ---- dynamic hedge / ALGO throttling ---------------------------------------------
# These do NOT change the core alpha. They only change how much ALGO capacity is used
# for directional net-dollar betting versus defensive beta hedging when the traded
# signal's realized IC/PnL is statistically deteriorating.
DYN_RISK_ON        = True
DYN_STATS_W        = 20       # trailing realized days used for health check
DYN_IC_SOFT_T      = 0.25      # start reacting when IC t-stat is below -1.5
DYN_IC_HARD_T      = 1.0      # full reaction around IC t-stat below -3.0
DYN_PNL_SOFT_T     = 0.25      # start reacting when traded proxy-PnL t-stat is below -1.5
DYN_PNL_HARD_T     = 1.0      # full reaction around PnL t-stat below -3.0
DYN_DEG_SOFT_T     = 0.25      # start reacting when recent half is worse than old half
DYN_DEG_HARD_T     = 1.0
DYN_MOM_BONUS      = 0.75     # extra caution when xs autocorr says momentum regime
DYN_HEDGE_LOOKBACK = 40      # beta lookback for ALGO hedge
DYN_HEDGE_MAX      = 1.0      # 1.0 = allow full beta hedge when risk score is 1
DYN_ALGO_MIN_SCALE = 0.0      # 0.0 = net-dollar ALGO bet can be fully cut

# ---- rotation knobs (ADOPTED pnl-W60 gate; verify-pnl-gate workflow) ----------
GATE_MODE  = "pnl"   # "ic" | "pnl" (adopted) | "sharpe"
ROT_W      = 60
ROT_TCRIT  = 2.5     # (ic mode only)
ROT_MARGIN = 0.0     # (ic mode only)
PNL_MARGIN = 0.0     # (pnl/sharpe) required trailing book-return edge over champion
ROT_P      = 5
ROT_BONF   = True

# ---- momentum challengers ------------------------------------------------------
MOMJT_L    = 120
MOMJT_S    = 20
RESIDM_L   = 120
RESIDM_S   = 20

# ---- ALGO index leg: IC-gated FADE vs TREND ------------------------------------
ALGO_ROT_W = 120
ALGO_ROT_H = 5
ALGO_TCRIT = 3.0
ALGO_P     = 10          # (accelerant tested + reverted: inert — see SAFE_rotate.py note / verify_accel2.py)

# ---- kill switch -----------------------------------------------------------------
KILL_ON    = True
KILL_TCRIT = 3.0
KILL_P     = 10

# ---- xsac validator ----------------------------------------------------------------
XSAC_W     = 40
XSAC_TH    = 0.07
XSAC_P     = 5
ROT_P_FAST = 3

# windowed cache: everything the gates read lies within this trailing span
#   pnl gate: ROT_P + ROT_W - 1 = 64   _kill: KILL_P + ROT_W - 1 = 69 (deepest)
#   _xsac_flag: XSAC_P + XSAC_W = 45  (RET-only)
LOOKBACK   = ROT_W + max(KILL_P, ROT_P) + 6          # = 76, covers all of the above (deepest 69, margin 7)
PRUNE_PAD  = 10                                      # keep a small margin beyond LOOKBACK

_DLR = None
_SIG = {}            # n -> dict{name: 50-vec forecast for day n}   (sliding window)
_RET = {}            # n -> realized demeaned idio return over day n (sliding window)
_ICD = {}            # (name, n) -> realized daily IC  (floats; kept, they are tiny)
_AZ  = {}            # s -> index trend z from the first s columns
_XC  = {}            # n -> cross-sectional lag-1 autocorr corr(_RET[n-1], _RET[n])
_PN  = {}            # (name, n) -> daily as-if-traded book-return proxy (sign(forecast) . realized ret)
_DEBUG_LAST = {}    # last decision diagnostics, read by diagnostic dashboard


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


def _forecasts(P):
    """Traded signals only (each a demeaned 50-vector). Identical formulas to SAFE_rotate."""
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

    # mom: short cross-sectional momentum (sign-flip of the champion's reversion term)
    out["mom"] = rr.copy()

    # momJT: Jegadeesh-Titman cross-sectional momentum (long lookback, skip recent reversal)
    if logp.shape[1] >= MOMJT_L + 1:
        g = logp[1:, -1 - MOMJT_S] - logp[1:, -1 - MOMJT_L]
        g = g - g.mean(); out["momJT"] = g / (g.std() + 1e-12)
    else:
        out["momJT"] = out["champ"].copy()

    # residMom: Blitz-Huij-Martens residual (factor-neutral) momentum
    if m >= RESIDM_L + 1:
        Rwin = r[1:, -RESIDM_L:]; r0win = r[0, -RESIDM_L:]; r0c = r0win - r0win.mean()
        beta = (Rwin @ r0c) / (r0c @ r0c + 1e-12)
        resid = Rwin - beta[:, None] * r0win[None, :]
        cum = (resid[:, :RESIDM_L - RESIDM_S] if RESIDM_S > 0 else resid).sum(1)
        cum = cum - cum.mean(); out["residMom"] = cum / (cum.std() + 1e-12)
    else:
        out["residMom"] = out["champ"].copy()

    return out


CHALLENGERS = ("mom", "momJT", "residMom")


def _ensure_cache(P):
    """Fill _SIG/_RET over the trailing LOOKBACK window only, then prune stale entries.
    Gate/kill/xsac lookbacks all lie inside the window (see LOOKBACK); ICs for older days
    are already memoized in _ICD when they were inside the window."""
    nInst, T = P.shape
    lo = max(WARMUP, T - LOOKBACK)
    for n in range(lo, T + 1):
        if n not in _SIG:
            _SIG[n] = _forecasts(P[:, :n])
        if n not in _RET and n < T:
            R = np.log(P[1:, n]) - np.log(P[1:, n - 1])
            _RET[n] = R - R.mean()
    cut = lo - PRUNE_PAD
    for d in (_SIG, _RET, _XC):
        for k in [k for k in d if k < cut]:
            del d[k]
    for k in [k for (nm, k) in _PN if k < cut]:               # prune _PN (keyed by (name, day))
        for nm in ("champ",) + CHALLENGERS:
            _PN.pop((nm, k), None)


def _ic1(name, n):
    key = (name, n); v = _ICD.get(key)
    if v is None:
        f = _SIG[n][name]; g = _RET[n]
        fm = f - f.mean(); gm = g - g.mean()
        denom = np.sqrt((fm @ fm) * (gm @ gm))
        v = float(fm @ gm / denom) if denom > 1e-18 else 0.0
        _ICD[key] = v
    return v


def _ic(name, lo, hi):
    return np.array([_ic1(name, n) for n in range(lo, hi)])


def _pn1(name, n):
    key = (name, n); v = _PN.get(key)
    if v is None:
        v = float((np.sign(_SIG[n][name]) * _RET[n]).sum()); _PN[key] = v
    return v


def _pn(name, lo, hi):
    return np.array([_pn1(name, n) for n in range(lo, hi)])


def _tcrit():
    if ROT_BONF and len(CHALLENGERS) > 1:
        return float(np.sqrt(ROT_TCRIT ** 2 + 2.0 * np.log(len(CHALLENGERS))))
    return ROT_TCRIT


def _xc1(n):
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
    vals = [_xc1(n) for n in range(a - XSAC_W + 1, a + 1)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) >= XSAC_W // 2 else None


def _xsac_flag(T):
    for a in range(T - XSAC_P, T):
        v = _xsac(a)
        if v is None or v <= XSAC_TH:
            return False
    return True


def _gate_at(a, tcrit=None):
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
    # profitability-based gate (adopted): beat champion on trailing realized book-return
    pch = _pn("champ", lo, a + 1)
    best = None; best_v = -1e18
    for name in CHALLENGERS:
        pc = _pn(name, lo, a + 1)
        if GATE_MODE == "sharpe":
            ok = pc.mean() / (pc.std() + 1e-9) > pch.mean() / (pch.std() + 1e-9) + PNL_MARGIN and pc.mean() > 0
            val = pc.mean() / (pc.std() + 1e-9)
        else:  # "pnl"
            ok = (pc - pch).mean() > PNL_MARGIN; val = pc.mean()
        if ok and val > best_v:
            best_v = val; best = name
    return best


def _choose(T):
    fast = _xsac_flag(T)
    P = ROT_P_FAST if fast else ROT_P
    bar = ROT_TCRIT if fast else _tcrit()
    picks = [_gate_at(a, bar) for a in range(T - P, T)]
    if picks and picks[0] is not None and all(p == picks[0] for p in picks):
        return picks[0]
    return "champ"


def _algo_z(lpA):
    mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
    if len(mv) < CONTRA_WZ:
        return None
    return float((mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12))


def _algo_zc(lpA, s):
    v = _AZ.get(s)
    if v is None:
        v = _algo_z(lpA[:s]); _AZ[s] = v
    return v


def _algo_gate_at(lpA, T):
    smin = CONTRA_K + CONTRA_WZ + 1
    zs = []; fs = []
    for s in range(T - ALGO_ROT_H, T - ALGO_ROT_H - ALGO_ROT_W, -ALGO_ROT_H):
        if s < smin:
            break
        z = _algo_zc(lpA, s)
        if z is None:
            continue
        zs.append(z); fs.append(lpA[s - 1 + ALGO_ROT_H] - lpA[s - 1])
    if len(zs) < 8:
        return False
    zs = np.asarray(zs); fs = np.asarray(fs)
    if zs.std() < 1e-12 or fs.std() < 1e-12:
        return False
    r = float(np.corrcoef(zs, fs)[0, 1]); n = len(zs)
    if r <= 0:
        return False
    t_ic = r * np.sqrt((n - 2) / (1.0 - r ** 2 + 1e-12))
    return t_ic > ALGO_TCRIT


def _algo_leg_mode(lpA, T):
    for a in range(T - ALGO_P + 1, T + 1):
        if not _algo_gate_at(lpA, a):
            return "fade"
    return "trend"


def _kill(T, chosen):
    if not KILL_ON:
        return False
    for a in range(T - KILL_P, T):
        lo = a - ROT_W + 1
        if lo < WARMUP:
            return False
        ic = _ic(chosen, lo, a + 1)
        t_i = ic.mean() / (ic.std() / np.sqrt(len(ic)) + 1e-18)
        if not (ic.mean() < 0 and t_i < -KILL_TCRIT):
            return False
    return True



def _neg_t_ramp(t_value, soft, hard):
    """0 when t is not meaningfully negative, 1 when t is strongly negative."""
    if not np.isfinite(t_value):
        return 0.0
    return float(np.clip((-t_value - soft) / (hard - soft + 1e-12), 0.0, 1.0))


def _mean_t(x):
    """One-sample t-stat of mean(x) > 0."""
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return 0.0, 0.0
    mu = float(x.mean())
    sd = float(x.std())
    t = mu / (sd / np.sqrt(len(x)) + 1e-18)
    return mu, float(t)


def _two_sample_delta_t(old, new):
    """
    t-stat for deterioration:
        new_mean - old_mean

    Negative means the signal has recently become worse.
    """
    old = np.asarray(old, dtype=float)
    new = np.asarray(new, dtype=float)

    if len(old) < 8 or len(new) < 8:
        return 0.0, 0.0, 0.0

    old_mu = float(old.mean())
    new_mu = float(new.mean())

    se = np.sqrt(
        old.var() / max(len(old), 1)
        + new.var() / max(len(new), 1)
    ) + 1e-18

    delta_t = (new_mu - old_mu) / se

    return old_mu, new_mu, float(delta_t)


def _signal_health(T, chosen):
    """
    Converts trailing realized correctness into two controls:

        badness          0..1  higher means traded signal is becoming unreliable
        hedge_intensity  0..1  higher means use more ALGO beta hedge
        algo_ll_scale    0..1  multiplier on net-dollar ALGO directional bet

    Uses only information that is already realized by time T.
    """
    if not DYN_RISK_ON:
        return {
            "badness": 0.0,
            "hedge_intensity": 0.0,
            "algo_ll_scale": 1.0,
            "ic_mean": 0.0,
            "ic_t": 0.0,
            "pnl_mean": 0.0,
            "pnl_t": 0.0,
            "old_ic_mean": 0.0,
            "new_ic_mean": 0.0,
            "degrade_t": 0.0,
            "momentum_flag": False,
        }

    hi = T
    lo = hi - DYN_STATS_W

    if lo < WARMUP:
        return {
            "badness": 0.0,
            "hedge_intensity": 0.0,
            "algo_ll_scale": 1.0,
            "ic_mean": 0.0,
            "ic_t": 0.0,
            "pnl_mean": 0.0,
            "pnl_t": 0.0,
            "old_ic_mean": 0.0,
            "new_ic_mean": 0.0,
            "degrade_t": 0.0,
            "momentum_flag": False,
        }

    ic = _ic(chosen, lo, hi)
    pn = _pn(chosen, lo, hi)

    ic_mean, ic_t = _mean_t(ic)
    pnl_mean, pnl_t = _mean_t(pn)

    mid = lo + len(ic) // 2
    old_ic = _ic(chosen, lo, mid)
    new_ic = _ic(chosen, mid, hi)
    old_ic_mean, new_ic_mean, degrade_t = _two_sample_delta_t(old_ic, new_ic)

    ic_bad = _neg_t_ramp(ic_t, DYN_IC_SOFT_T, DYN_IC_HARD_T)
    pnl_bad = _neg_t_ramp(pnl_t, DYN_PNL_SOFT_T, DYN_PNL_HARD_T)

    # Only count deterioration if the recent half is actually worse.
    deg_bad = 0.0
    if new_ic_mean < old_ic_mean:
        deg_bad = _neg_t_ramp(degrade_t, DYN_DEG_SOFT_T, DYN_DEG_HARD_T)

    # Positive cross-sectional autocorrelation means price moves are persisting.
    # That is exactly where mean-reversion / lead-lag fading can become dangerous.
    momentum_flag = _xsac_flag(T)
    mom_bad = DYN_MOM_BONUS if momentum_flag and chosen == "champ" else 0.0

    badness = float(np.clip(
        0.45 * ic_bad
        + 0.35 * pnl_bad
        + 0.30 * deg_bad
        + mom_bad,
        0.0,
        1.0,
    ))

    hedge_intensity = float(np.clip(DYN_HEDGE_MAX * badness, 0.0, 1.0))

    # This is the key ALGO throttle:
    # if our traded signal is increasingly wrong, reduce the net-dollar ALGO bet.
    algo_ll_scale = float(np.clip(
        max(DYN_ALGO_MIN_SCALE, 1.0 - badness),
        0.0,
        1.0,
    ))

    return {
        "badness": badness,
        "hedge_intensity": hedge_intensity,
        "algo_ll_scale": algo_ll_scale,
        "ic_mean": ic_mean,
        "ic_t": ic_t,
        "pnl_mean": pnl_mean,
        "pnl_t": pnl_t,
        "old_ic_mean": old_ic_mean,
        "new_ic_mean": new_ic_mean,
        "degrade_t": degrade_t,
        "momentum_flag": momentum_flag,
    }


def _book_beta_hedge(P, idio_int, cur, dlr, hedge_intensity):
    """
    Dynamic ALGO hedge for the 50-stock book.

    Estimate each idio asset's beta to ALGO over a trailing window, then hedge the
    current idio dollar book with ALGO.

        hedge_shares = - sum_i(dollar_i * beta_i) / ALGO_price

    hedge_intensity controls how much of the hedge is applied.
    """
    if hedge_intensity <= 0.0:
        return 0.0

    logp = np.log(P)
    r = logp[:, 1:] - logp[:, :-1]

    L = min(DYN_HEDGE_LOOKBACK, r.shape[1])
    if L < 20:
        return 0.0

    rw = r[:, -L:]

    rA = rw[0] - rw[0].mean()
    denom = float(rA @ rA) + 1e-12

    betas = (
        (rw[1:] - rw[1:].mean(axis=1, keepdims=True)) @ rA
    ) / denom

    idio_dollars = idio_int * cur[1:]
    hedge_shares = -float(idio_dollars @ betas) / cur[0]

    cap = dlr[0] / cur[0]
    hedge_shares = float(np.clip(hedge_intensity * hedge_shares, -cap, cap))

    return hedge_shares


def _algo_fade_or_trend_position(logp, cur, cap, t):
    """
    Original ALGO fade/trend leg from SAFE_live:
        fade by default
        switch to trend if ALGO IC gate says index momentum is significant
    """
    lpA = logp[0]
    mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]

    if len(mv) < CONTRA_WZ:
        return 0.0

    z = (mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12)
    zc = np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / cur[0])

    trend = (
        t >= WARMUP + ALGO_ROT_W + ALGO_P
        and (_algo_leg_mode(lpA, t) == "trend")
    )

    return float(np.clip(zc if trend else -zc, -cap, cap))

def getMyPosition(prcSoFar):
    global _DEBUG_LAST

    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nInst, t = prcSoFar.shape
    dlr = _limits(nInst)
    cur = prcSoFar[:, -1]
    pos = np.zeros(nInst)

    if t < WARMUP:
        _DEBUG_LAST = {
            "day_index": t - 1,
            "ready": False,
            "chosen": "warmup",
            "killed": False,
            "kill_reason": "warmup",
            "net_dol": 0.0,
            "algo_source": "warmup",
            "algo_mode": "warmup",
            "algo_alpha_shares": 0.0,
            "algo_alpha_dollars": 0.0,
            "hedge_alpha_shares": 0.0,
            "hedge_alpha_dollars": 0.0,
            "final_algo_shares": 0.0,
            "final_algo_dollars": 0.0,
            "badness": 0.0,
            "hedge_intensity": 0.0,
            "algo_ll_scale": 1.0,
            "ic_mean": 0.0,
            "ic_t": 0.0,
            "pnl_mean": 0.0,
            "pnl_t": 0.0,
            "old_ic_mean": 0.0,
            "new_ic_mean": 0.0,
            "degrade_t": 0.0,
            "momentum_flag": False,
            "long_count": 0,
            "short_count": 0,
        }
        return pos.astype(int)

    _ensure_cache(prcSoFar)
    ready = t >= WARMUP + ROT_W + ROT_P
    chosen = _choose(t) if ready else "champ"
    wz = _SIG[t][chosen]

    logp = np.log(prcSoFar)

    killed = ready and _kill(t, chosen)
    if not killed:
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    idio_lim = (dlr[1:] / cur[1:]).astype(int)
    idio_int = np.clip(pos[1:], -idio_lim, idio_lim).astype(int)
    net_dol = float((idio_int * cur[1:]).sum())
    long_count = int(np.sum(idio_int > 0))
    short_count = int(np.sum(idio_int < 0))

    cap = dlr[0] / cur[0]

    # ----------------------------------------------------------------------
    # Dynamic risk controls
    # ----------------------------------------------------------------------
    health = _signal_health(t, chosen) if ready else {
        "badness": 0.0,
        "hedge_intensity": 0.0,
        "algo_ll_scale": 1.0,
        "ic_mean": 0.0,
        "ic_t": 0.0,
        "pnl_mean": 0.0,
        "pnl_t": 0.0,
        "old_ic_mean": 0.0,
        "new_ic_mean": 0.0,
        "degrade_t": 0.0,
        "momentum_flag": False,
    }

    algo_ll_scale = float(health["algo_ll_scale"])
    hedge_intensity = float(health["hedge_intensity"])

    # ----------------------------------------------------------------------
    # ALGO alpha leg
    # ----------------------------------------------------------------------
    algo_source = "none"
    algo_mode = "none"
    raw_algo_shares = 0.0

    if ALGO_LL_DOLLAR > 0 and abs(net_dol) >= ALGO_LL_DOLLAR:
        raw_algo_shares = float(np.sign(net_dol) * cap)
        algo_alpha = float(raw_algo_shares * algo_ll_scale)
        algo_source = "net_dollar"
        algo_mode = "net_dollar_scaled" if algo_ll_scale < 0.999 else "net_dollar_full"
    else:
        # Original SAFE ALGO fade/trend leg, expanded here so the dashboard can see the mode.
        lpA = logp[0]
        mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        if len(mv) < CONTRA_WZ:
            algo_alpha = 0.0
            algo_source = "fallback"
            algo_mode = "fallback_not_ready"
        else:
            z = (mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12)
            zc = np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / cur[0])
            trend = (
                t >= WARMUP + ALGO_ROT_W + ALGO_P
                and (_algo_leg_mode(lpA, t) == "trend")
            )
            algo_alpha = float(np.clip(zc if trend else -zc, -cap, cap))
            raw_algo_shares = algo_alpha
            algo_source = "fallback"
            algo_mode = "trend" if trend else "fade"

    # ----------------------------------------------------------------------
    # Dynamic ALGO hedge leg
    # ----------------------------------------------------------------------
    hedge_alpha = _book_beta_hedge(
        P=prcSoFar,
        idio_int=idio_int,
        cur=cur,
        dlr=dlr,
        hedge_intensity=hedge_intensity,
    )

    pos[0] = float(np.clip(algo_alpha + hedge_alpha, -cap, cap))

    lim = (dlr / cur).astype(int)
    final_pos = np.clip(pos, -lim, lim).astype(int)

    hedge_dollars = float(hedge_alpha * cur[0])
    algo_alpha_dollars = float(algo_alpha * cur[0])
    final_algo_dollars = float(final_pos[0] * cur[0])

    reasons = []
    if killed:
        reasons.append("kill switch flattened idio book")
    if hedge_intensity > 1e-6:
        reasons.append("dynamic beta hedge active")
    if algo_source == "net_dollar" and algo_ll_scale < 0.999:
        reasons.append("net-dollar ALGO bet throttled")
    if health.get("momentum_flag", False):
        reasons.append("xs autocorr momentum flag")
    if not reasons:
        reasons.append("normal")

    _DEBUG_LAST = {
        "day_index": t - 1,
        "ready": bool(ready),
        "chosen": chosen,
        "killed": bool(killed),
        "kill_reason": "ic inversion" if killed else "",
        "net_dol": net_dol,
        "algo_source": algo_source,
        "algo_mode": algo_mode,
        "raw_algo_shares": float(raw_algo_shares),
        "raw_algo_dollars": float(raw_algo_shares * cur[0]),
        "algo_alpha_shares": float(algo_alpha),
        "algo_alpha_dollars": algo_alpha_dollars,
        "hedge_alpha_shares": float(hedge_alpha),
        "hedge_alpha_dollars": hedge_dollars,
        "final_algo_shares": float(final_pos[0]),
        "final_algo_dollars": final_algo_dollars,
        "badness": float(health.get("badness", 0.0)),
        "hedge_intensity": hedge_intensity,
        "algo_ll_scale": algo_ll_scale,
        "ic_mean": float(health.get("ic_mean", 0.0)),
        "ic_t": float(health.get("ic_t", 0.0)),
        "pnl_mean": float(health.get("pnl_mean", 0.0)),
        "pnl_t": float(health.get("pnl_t", 0.0)),
        "old_ic_mean": float(health.get("old_ic_mean", 0.0)),
        "new_ic_mean": float(health.get("new_ic_mean", 0.0)),
        "degrade_t": float(health.get("degrade_t", 0.0)),
        "momentum_flag": bool(health.get("momentum_flag", False)),
        "long_count": long_count,
        "short_count": short_count,
        "reason": "; ".join(reasons),
    }

    return final_pos