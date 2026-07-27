"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$   algopart3/SAFE_combined.py   ·   ADAPTIVE lead-lag  +  PROTECTED structural pairs      $$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

One book fusing the three ingredients discussed:

  1. ADAPTIVE  -- the champion is the lead-lag ridge ENSEMBLE (250/500/1000/2000 half-lives),
                  re-fit every day by exponential forgetting. That IS the adaptation: kalman_tvp.py
                  showed a full time-varying (Kalman) B adds nothing here (drift q_hat~0, B is
                  stationary -> a one-factor VAR(1)), so the forgetting-ridge is the right adaptive.
                  Blended with 30% cross-sectional reversion == algopart2/SAFE_lldollar champion.

  2. STRUCTURAL PAIRS, PROTECTED  -- a NEW leg (`pairs`). For each name we adaptively pick its best
                  partner (rolling return-correlation), fit an adaptive hedge ratio (rolling OLS on
                  levels), and fade the spread's z-score. PROTECTION = a per-pair cointegration gate:
                  an augmented-Dickey-Fuller t-stat on the spread; a pair only trades while its spread
                  is significantly mean-reverting (t_DF < PAIR_DF_CRIT), and is dropped the moment it
                  loses that significance. This is exactly "keep testing the relationship; stop
                  trading the pair when it fails" -- implemented per pair, re-checked every day.

  3. PROTECTION (book level)  -- the champion-health VALIDATOR + kill switch from part-3. `pairs`
                  joins the fallback pool {pairs, tsrev, mom, momJT, residMom}: the validator only
                  hands the book to it if it is BOTH the best trailing performer AND beating the
                  champion, while champ is sick. So the pairs leg is safe-by-construction -- it can
                  never dilute the shipped edge; it trades only when it has earned the seat. Kill
                  switch flattens the idio book if the traded signal's IC inverts (sustained t<-3).

Trades identically to algopart2/SAFE_lldollar on healthy real data (validator silent) -- verified
in scratchpad/verify_combined.py. EXPECTATION for the pairs leg (from btiao/reversion-realm probes):
in a one-factor market every idio spread is already stationary, so the DF gate rarely fires and the
surviving pairs carry little edge beyond reversion -- `pairs` is expected to be largely inert, like
tsrev. It is included, protected and combined, so it can help if a regime ever rewards it and cannot
hurt otherwise.

Caveats (same as the other windowed builds): monotonic-t calls only; one price panel per process --
clear the caches below (or reload) between panels, since every cache is keyed by column count.
==========================================================================================
"""
import numpy as np

BOOK = "SAFE · COMBINED (adaptive lead-lag + protected structural pairs + validator)"

# ---- trading knobs (identical to algopart2/SAFE_lldollar champion) -----------
HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
WARMUP      = 96
ALGO_LL_DOLLAR = 50_000

# ---- structural-pairs leg (adaptive hedge ratio + cointegration protection) --
PAIR_WIN     = 120     # rolling window: partner corr, hedge-ratio OLS, spread stats, DF test
PAIR_DF_CRIT = -2.9    # augmented-Dickey-Fuller t on the spread; below this => cointegrated => trade
PAIR_ZCLIP   = 3.0     # clip the spread z-score

# ---- TS (own-stock) reversion fallback ---------------------------------------
TSREV_K     = 7
TSREV_W     = 60

# ---- momentum-family fallbacks -----------------------------------------------
MOMJT_L     = 120
MOMJT_S     = 20
RESIDM_L    = 120
RESIDM_S    = 20

# ---- regime-switch validator -------------------------------------------------
ROT_W        = 60
ROT_P        = 5
VAL_IC_FLOOR = 0.0
XSAC_W       = 40
XSAC_TH      = 0.07

# ---- kill switch -------------------------------------------------------------
KILL_ON      = True
KILL_TCRIT   = 3.0
KILL_P       = 10

FALLBACKS = ("pairs", "tsrev", "mom", "momJT", "residMom")
ALL_SIG   = ("champ",) + FALLBACKS

LOOKBACK  = ROT_W + max(KILL_P, ROT_P) + 6
PRUNE_PAD = 10

_DLR = None
_SIG = {}; _RET = {}; _ICD = {}; _PN = {}; _XC = {}


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


def _pairs_signal(P):
    """Protected adaptive structural-pairs reversion -> demeaned 50-vector (or None if unusable).
    Per name: adaptive best partner + adaptive hedge ratio; fade the spread z ONLY while an ADF
    cointegration test says the spread is significantly mean-reverting (else the pair is dropped)."""
    L = np.log(P)[1:]                                     # (50, T) stock log-price levels
    T = L.shape[1]
    if T < PAIR_WIN + 2:
        return None
    W = L[:, -PAIR_WIN:]                                  # recent levels
    Rw = W[:, 1:] - W[:, :-1]                             # returns over the window
    Rc = Rw - Rw.mean(1, keepdims=True)
    dnm = np.sqrt((Rc ** 2).sum(1))
    C = (Rc @ Rc.T) / (np.outer(dnm, dnm) + 1e-12)        # return-correlation matrix
    np.fill_diagonal(C, -np.inf)
    partner = np.argmax(C, axis=1)                        # adaptive best partner per name

    sig = np.zeros(50); traded = 0
    for i in range(50):
        j = int(partner[i])
        xi, xj = W[i], W[j]
        xjc = xj - xj.mean()
        b = (xjc @ (xi - xi.mean())) / (xjc @ xjc + 1e-12)   # adaptive hedge ratio (levels OLS)
        s = xi - b * xj; s = s - s.mean()                    # cointegrating spread
        # ADF-style gate: regress Δs on s_{t-1}; t-stat of the level coefficient
        ds = s[1:] - s[:-1]; s0 = s[:-1]
        s0c = s0 - s0.mean(); dsc = ds - ds.mean()
        denom = s0c @ s0c + 1e-12
        rho = (s0c @ dsc) / denom
        resid = dsc - rho * s0c
        n = len(ds); se = np.sqrt((resid @ resid) / max(n - 2, 1) / denom)
        t_df = rho / (se + 1e-12)
        if t_df < PAIR_DF_CRIT:                              # significantly mean-reverting -> trade
            z = np.clip((s[-1]) / (s.std() + 1e-12), -PAIR_ZCLIP, PAIR_ZCLIP)
            sig[i] += -z; sig[j] += b * z; traded += 1
    if traded == 0:
        return None
    return sig - sig.mean()


def _forecasts(P):
    """champion + fallbacks (each a 50-vector over the stocks)."""
    logp = np.log(P); r = logp[:, 1:] - logp[:, :-1]; m = r.shape[1]
    L = logp[1:]
    out = {}

    # champion: adaptive lead-lag ensemble + short cross-sectional reversion (== lldollar)
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = _ewls_ridge(r[:, :-1].T, r[1:, 1:].T, hl, RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean(); fs.append(fi / (fi.std() + 1e-12))
    ll_z = np.mean(fs, 0)
    rr = L[:, -1] - L[:, -1 - REV_W]; rr = rr - rr.mean()
    rev_z = -rr / (rr.std() + 1e-12)
    out["champ"] = (1 - BLEND) * ll_z + BLEND * rev_z

    # structural-pairs leg (protected)
    ps = _pairs_signal(P)
    out["pairs"] = ps if ps is not None else out["champ"].copy()

    # mom: short cross-sectional momentum (sign-flip of champ reversion leg)
    out["mom"] = rr.copy()

    # tsrev: own-stock time-series reversion
    kr = L[:, TSREV_K:] - L[:, :-TSREV_K]
    if kr.shape[1] >= TSREV_W + 1:
        cur_g = kr[:, -1]; hist = kr[:, -1 - TSREV_W:-1]
        out["tsrev"] = -(cur_g - hist.mean(1)) / (hist.std(1) + 1e-12)
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
    vals = [_xc1(n) for n in range(a - XSAC_W + 1, a + 1)]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if len(vals) >= XSAC_W // 2 else None


def _pick_at(a):
    lo = a - ROT_W + 1
    if lo < WARMUP:
        return "champ"
    ic_c = _ic("champ", lo, a + 1)
    xs = _xsac(a)
    champ_sick = (ic_c.mean() <= VAL_IC_FLOOR) or (xs is not None and xs > XSAC_TH)
    if not champ_sick:
        return "champ"
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
