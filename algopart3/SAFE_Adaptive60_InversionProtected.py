"""
##########################################################################################
### SAFE adaptive-60 + inversion protection · tournament submission candidate          ###
##########################################################################################
Self-contained tournament build derived from SAFE Live.  It deliberately changes the
lead-lag ensemble weights using only trailing realised data, so it is not position-identical
to SAFE Live.  The adaptive rule and inversion response are both causal.

  1. LEAN FORECASTS: computes only the signals the gates read (champ + mom/momJT/residMom).
     The six diagnostic signals (revL/ll2/resid/btiao/volsc/momVS) are research-only and
     never traded — dropped here. ll2's double-size ridge was the single largest cost.
  2. WINDOWED CACHE: forecasts/ICs are only ever needed over the trailing
     ROT_W + max(KILL_P, ROT_P) days, so the cache fills (and prunes) a sliding window
     instead of all history. First call is seconds, not minutes, at any history length.

Protection stack (all verified in the SAFE_rotate research build):
  * champion = lead-lag ensemble + reversion blend (the main edge)
  * adaptive ensemble = four half-lives weighted by positive trailing 60-day IR;
    equal-weight fallback when none has demonstrated positive realised payoff
  * momentum challengers mom/momJT/residMom — trailing-profit-gated rotation
  * xsac validator — direct cross-sectional autocorr; relaxes the gate in a momentum regime
  * kill switch — flattens the idio book if the traded edge's IC INVERTS (sustained t<-3)
  * inversion response — after a faster, still-persistent failure test, reverses the
    stock book and beta-hedges it with ALGO; completely inactive on visible days 1-750
  * ALGO leg — net-$ gate; else IC-gated FADE (default) vs TREND (index TSMOM)
==========================================================================================
"""
import numpy as np

BOOK = "SAFE · ADAPTIVE60 · INVERSION PROTECTED"

# ---- trading knobs (identical to SAFE_lldollar) -----------------------------
HALF_LIVES  = (250, 500, 1000, 2000)
HL_IR_W     = 60
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
WARMUP      = 96
ALGO_LL_DOLLAR = 50_000

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

# ---- hidden-regime inversion response ------------------------------------------
# If the selected stock signal's realized sign-book payoff is significantly
# negative over 20 days on three consecutive as-of dates, treat that as evidence
# that the relationship has reversed.  These values were chosen from causal
# stress tests, not by optimizing the visible-period score: this branch never
# activates anywhere in days 1-750.
INV_W      = 20
INV_TCRIT  = 1.5
INV_P      = 3
INV_BETA_W = 120

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
        fi = pred - pred.mean()
        signal = fi / (fi.std() + 1e-12)
        fs.append(signal)
        out[f"hl_{hl}"] = signal

    # Causal adaptive weighting.  Only payoffs realised before the current day
    # enter the 60-day information-ratio estimate.  A non-positive estimate gets
    # zero weight; if all four are non-positive, fall back to equal weights.
    T = P.shape[1]
    weights = np.ones(len(HALF_LIVES))
    lo = T - HL_IR_W
    if lo >= WARMUP and all(n in _SIG and n in _RET for n in range(lo, T)):
        estimated = []
        for hl in HALF_LIVES:
            payoff = np.array([
                float(np.sign(_SIG[n][f"hl_{hl}"]) @ _RET[n])
                for n in range(lo, T)
            ])
            ir = payoff.mean() / (payoff.std(ddof=1) + 1e-12)
            estimated.append(max(0.0, float(ir)))
        if sum(estimated) > 1e-12:
            weights = np.asarray(estimated)
    weights = weights / weights.sum()
    ll_z = np.sum(np.asarray(fs) * weights[:, None], axis=0)
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


def _selected_at(a):
    """Signal that would have been selected using information available at a."""
    ready = a >= WARMUP + ROT_W + ROT_P
    return _choose(a) if ready else "champ"


def _inverted_at(a):
    """Causal test for a statistically meaningful negative signal payoff."""
    lo = a - INV_W
    if lo < WARMUP:
        return False
    x = _pn(_selected_at(a), lo, a)
    mean = float(x.mean())
    t_stat = mean / (float(x.std(ddof=1)) / np.sqrt(len(x)) + 1e-18)
    return mean < 0 and t_stat < -INV_TCRIT


def _confirmed_inversion(T):
    """Require the failure condition to persist before changing the book."""
    return all(_inverted_at(a) for a in range(T - INV_P + 1, T + 1))


def _beta_hedge(prices, positions):
    """Size ALGO to offset the stock book's rolling market beta."""
    r = np.diff(np.log(prices), axis=1)
    r = r[:, -min(INV_BETA_W, r.shape[1]):]
    r0 = r[0] - r[0].mean()
    stocks = r[1:] - r[1:].mean(axis=1, keepdims=True)
    betas = stocks @ r0 / (r0 @ r0 + 1e-12)
    cur = prices[:, -1]
    stock_beta_dollars = float((positions[1:] * cur[1:]) @ betas)
    return -stock_beta_dollars / cur[0]


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
        trend = (t >= WARMUP + ALGO_ROT_W + ALGO_P) and (_algo_leg_mode(lpA, t) == "trend")
        av = float(np.clip(zc if trend else -zc, -cap, cap))

    pos[0] = av

    # The ordinary strategy is left untouched until a causal, persistent
    # inversion is observed.  When confirmed, reverse the existing stock book
    # at full size and replace the directional ALGO leg with a beta hedge.
    if ready and _confirmed_inversion(t):
        pos[1:] *= -1.0
        pos[0] = _beta_hedge(prcSoFar, pos)

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
