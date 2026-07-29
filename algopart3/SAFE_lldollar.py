"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$   algopart3/SAFE_lldollar.py   ·   LLDOLLAR + REGIME-SWITCHED FALLBACK (validator)     $$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Part-3 build. Base edge is IDENTICAL to algopart2/SAFE_lldollar.py:
    champion = lead-lag ridge ensemble  +  30% cross-sectional reversion,
    idio book sign-sized, ALGO index leg = net-$ book-skew gate else fade the 30-day move.

ADDED (this build): a REGIME-SWITCH VALIDATOR that can hand the idio book to a FALLBACK signal
when the champion's edge dies, then flatten if it inverts:

  fallback pool
    tsrev     TIME-SERIES own-stock reversion (fade each name vs ITS OWN history, no XS demean)
    mom       short cross-sectional momentum (sign-flip of the champion's reversion leg)
    momJT     Jegadeesh-Titman cross-sectional momentum (long lookback, skip recent reversal)
    residMom  Blitz-Huij-Martens residual (index-neutral) momentum

  validator (_choose / _pick_at)  -- champion-health driven, whipsaw-guarded
    each day: is the champion still working?
      HEALTHY  -> trade champion (default)
      UNHEALTHY (trailing-ROT_W champion IC has gone negative, OR a cross-sectional-momentum
                 regime is flagged by xsac) -> switch to the fallback with the best trailing
                 book-return, provided it is positive AND beats the champion.
    a switch fires only if the SAME fallback wins that test on ROT_P consecutive days.

  kill switch (_kill)
    if the CURRENTLY TRADED signal's trailing IC is significantly negative (t < -KILL_TCRIT)
    for KILL_P consecutive days -> flatten the idio book (regime is hostile to everything).

NOTE ON TSREV: an offline probe (scratchpad/ts_reversion_probe.py) found own-stock reversion is
strictly dominated by the champion's cross-sectional reversion on all real data and fails in
lockstep with it under a momentum regime -- momentum is the real inverted-regime fallback. tsrev
is included here at the user's request; the validator will simply never pick it if it isn't the
best trailing performer, so its presence is harmless.

Caveats (same as the algopart2 windowed builds): monotonic-t calls only; one price panel per
process -- a harness running multiple panels must clear the caches below (or reload the module)
between them, because every cache is keyed by column count, not by series identity.
==========================================================================================
"""
import numpy as np

BOOK      = "SAFE · LL-DOLLAR ($) + regime-switch validator (part3)"
GATE_KIND = "net-$ book skew  |net$| >= $50k  ->  FULL $100k ALGO (MAX)"

# ---- trading knobs (identical to algopart2/SAFE_lldollar) --------------------
HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
WARMUP      = 96
ALGO_LL_DOLLAR = 50_000

# ---- TS (own-stock) reversion fallback ---------------------------------------
TSREV_K     = 7      # reversion horizon (probe-best); z-scored vs each name's own history
TSREV_W     = 60     # own-history window for the z-score

# ---- momentum-family fallbacks -----------------------------------------------
MOMJT_L     = 120
MOMJT_S     = 20
RESIDM_L    = 120
RESIDM_S    = 20

# ---- regime-switch validator -------------------------------------------------
ROT_W       = 60     # trailing window for champion-health IC and fallback book-return
ROT_P       = 5      # a switch must win on this many CONSECUTIVE days (whipsaw guard)
VAL_IC_FLOOR = 0.0   # champion "unhealthy" when its trailing-ROT_W mean IC drops to/below this
XSAC_W      = 40     # cross-sectional lag-1 autocorr window (momentum-regime detector)
XSAC_TH     = 0.07   # sustained autocorr above this => momentum regime (real-data max ~+0.061)

# ---- kill switch -------------------------------------------------------------
KILL_ON     = True
KILL_TCRIT  = 3.0
KILL_P      = 10

FALLBACKS = ("tsrev", "mom", "momJT", "residMom")
ALL_SIG   = ("champ",) + FALLBACKS

# windowed cache: deepest lookback is kill (KILL_P+ROT_W-1=69); +margin
LOOKBACK  = ROT_W + max(KILL_P, ROT_P) + 6      # = 76
PRUNE_PAD = 10

_DLR = None
_SIG = {}     # n -> {name: 50-vec forecast built from prices[:, :n]}
_RET = {}     # n -> demeaned realized idio return over day n
_ICD = {}     # (name, n) -> realized daily IC
_PN  = {}     # (name, n) -> daily as-if-traded book-return proxy sign(forecast).realized
_XC  = {}     # n -> cross-sectional lag-1 autocorr corr(_RET[n-1], _RET[n])


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
    """champion + fallbacks, each a 50-vector over the stocks (index 0 excluded)."""
    logp = np.log(P); r = logp[:, 1:] - logp[:, :-1]; m = r.shape[1]
    L = logp[1:]                                          # stock log prices
    out = {}

    # champion: lead-lag ensemble blended with short cross-sectional reversion (== lldollar)
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = _ewls_ridge(r[:, :-1].T, r[1:, 1:].T, hl, RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean(); fs.append(fi / (fi.std() + 1e-12))
    ll_z = np.mean(fs, 0)
    rr = L[:, -1] - L[:, -1 - REV_W]; rr = rr - rr.mean()
    rev_z = -rr / (rr.std() + 1e-12)
    out["champ"] = (1 - BLEND) * ll_z + BLEND * rev_z

    # mom: short cross-sectional momentum (sign-flip of the champion's reversion term)
    out["mom"] = rr.copy()

    # tsrev: own-stock time-series reversion -- fade each name's TSREV_K-day return, z-scored
    #        against ITS OWN trailing TSREV_W history (no cross-sectional demeaning)
    kr = L[:, TSREV_K:] - L[:, :-TSREV_K]                 # rolling k-day returns per name
    if kr.shape[1] >= TSREV_W + 1:
        cur_g = kr[:, -1]; hist = kr[:, -1 - TSREV_W:-1]
        mu = hist.mean(1); sd = hist.std(1) + 1e-12
        out["tsrev"] = -(cur_g - mu) / sd
    else:
        out["tsrev"] = out["champ"].copy()

    # momJT: Jegadeesh-Titman cross-sectional momentum
    if logp.shape[1] >= MOMJT_L + 1:
        g = L[:, -1 - MOMJT_S] - L[:, -1 - MOMJT_L]
        g = g - g.mean(); out["momJT"] = g / (g.std() + 1e-12)
    else:
        out["momJT"] = out["champ"].copy()

    # residMom: Blitz-Huij-Martens residual (index-neutral) momentum
    if m >= RESIDM_L + 1:
        Rwin = r[1:, -RESIDM_L:]; r0win = r[0, -RESIDM_L:]; r0c = r0win - r0win.mean()
        beta = (Rwin @ r0c) / (r0c @ r0c + 1e-12)
        resid = Rwin - beta[:, None] * r0win[None, :]
        cum = (resid[:, :RESIDM_L - RESIDM_S] if RESIDM_S > 0 else resid).sum(1)
        cum = cum - cum.mean(); out["residMom"] = cum / (cum.std() + 1e-12)
    else:
        out["residMom"] = out["champ"].copy()

    return out


def _ensure_cache(P):
    """Fill _SIG/_RET over the trailing LOOKBACK window, then prune stale entries."""
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
    for k in [k for (nm, k) in _PN if k < cut]:
        for nm in ALL_SIG:
            _PN.pop((nm, k), None)
    for k in [k for (nm, k) in _ICD if k < cut]:
        for nm in ALL_SIG:
            _ICD.pop((nm, k), None)


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


def _tstat(x):
    x = np.asarray(x, float)
    return float(x.mean() / (x.std() / np.sqrt(len(x)) + 1e-18))


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
    """trailing-XSAC_W mean cross-sectional lag-1 autocorr ending at day a (momentum-regime gauge)."""
    vals = [_xc1(n) for n in range(a - XSAC_W + 1, a + 1)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) >= XSAC_W // 2 else None


def _pick_at(a):
    """One day's validator verdict: 'champ' unless the champion is sick AND a fallback is beating it."""
    lo = a - ROT_W + 1
    if lo < WARMUP:
        return "champ"
    ic_c = _ic("champ", lo, a + 1)
    xs = _xsac(a)
    champ_sick = (ic_c.mean() <= VAL_IC_FLOOR) or (xs is not None and xs > XSAC_TH)
    if not champ_sick:
        return "champ"                                    # edge still working -> stay
    pch = _pn("champ", lo, a + 1)
    best = None; best_v = -1e18
    for name in FALLBACKS:
        pc = _pn(name, lo, a + 1)
        if (pc - pch).mean() > 0.0 and pc.mean() > 0.0 and pc.mean() > best_v:
            best_v = pc.mean(); best = name
    return best if best is not None else "champ"


def _choose(T):
    picks = [_pick_at(a) for a in range(T - ROT_P, T)]
    if picks and picks[0] is not None and picks[0] != "champ" and all(p == picks[0] for p in picks):
        return picks[0]
    return "champ"


def _kill(T, chosen):
    if not KILL_ON:
        return False
    for a in range(T - KILL_P, T):
        lo = a - ROT_W + 1
        if lo < WARMUP:
            return False
        ic = _ic(chosen, lo, a + 1)
        if not (ic.mean() < 0 and _tstat(ic) < -KILL_TCRIT):
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
    chosen = _choose(t) if ready else "champ"
    wz = _SIG[t][chosen]

    logp = np.log(prcSoFar)

    killed = ready and _kill(t, chosen)
    if not killed:
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    # ---- net dollar skew of the stock book -> ALGO index leg (== lldollar) ------
    idio_lim = (dlr[1:] / cur[1:]).astype(int)
    idio_int = np.clip(pos[1:], -idio_lim, idio_lim).astype(int)
    net_dol = float((idio_int * cur[1:]).sum())

    cap = dlr[0] / cur[0]
    if ALGO_LL_DOLLAR > 0 and abs(net_dol) >= ALGO_LL_DOLLAR:
        av = float(np.sign(net_dol) * cap)
    else:
        lpA = logp[0]; mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        z = (mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12)
        av = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / cur[0]), -cap, cap))

    pos[0] = av
    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
