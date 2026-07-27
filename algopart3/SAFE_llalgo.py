"""
############################################################################################
###   SAFE_llalgo.py   ·   ALGO GATE = STOCK  #  COUNT SKEW   ·   MAX sizing (full $100k) ###
############################################################################################
  >>> THIS IS THE **COUNT** SIBLING. gate reads a NAME COUNT (|frac|>=0.12 = 28 vs 22).  <<<
  >>> its twin SAFE_lldollar.py reads a $ AMOUNT (net$ book skew >= $50k).                <<<
  Both are the SAME edge & SAME score (696 graded / 6.06 Sharpe); only the gate UNIT differs.

Copy of SAFE.py. ONLY the ALGO (index, instrument 0) leg is changed: instead of
fading ALGO's 30-day move (reversion), it is driven PRIMARILY by the lead-lag
signal, expressed as the cross-sectional long/short imbalance of the idio book
  frac = mean(sign(wz))            # +frac => more names flagged long
which carried a stable +0.08..+0.14 IC on ALGO's next-day return across windows,
vs the reversion fade's ~0.02 on the graded leg (algo_leadlag_probe.py).

  ALGO_LL_W : weight on the lead-lag leg (1.0 = pure lead-lag, 0.0 = old reversion)
  FRAC_SCALE: std of `frac` used to turn it into a z-score (empirical ~0.09)

The idio leg (instruments 1..49) is IDENTICAL to SAFE.py.
================================================================================
"""
import numpy as np

HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
HEDGE       = False
WARMUP      = 96

ALGO_LL_W   = 1.0       # <-- weight on lead-lag on days the gate is ON (1.0 = pure lead-lag)
ALGO_LL_GATE = 0.12     # <-- VALIDATED: only use lead-lag when |frac| >= 0.12 (>=28/50 names lean
                        #     one way); else reversion default. Rolling 250-day windows: mean 655->687,
                        #     floor 534->544, 700+ windows 9->14, beats baseline 32/38 (validate_gate.py)
FRAC_SCALE  = 0.09      # empirical std of mean(sign(wz)); turns frac into a ~z-score

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

    pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    # ---- ALGO index leg: lead-lag (tilt) prioritised over reversion -------------
    cap = dlr[0] / cur[0]
    notional = CONTRA_DOL / cur[0]

    # (a) lead-lag leg: cross-sectional long/short imbalance -> ALGO direction
    frac = float(np.mean(np.sign(wz)))                    # +ve => net long tilt
    zt = np.clip(frac / FRAC_SCALE, -3.0, 3.0)
    ll_av = zt / 3.0 * notional                           # +frac => long ALGO (IC +ve)

    # (b) reversion leg: fade ALGO's 30-day move (the old signal), kept as secondary
    lpA = logp[0]; mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
    z = (mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12)
    rev_av = -np.clip(z, -3, 3) / 3.0 * notional

    # gate: use lead-lag only on high-skew days; otherwise keep the reversion default
    if ALGO_LL_GATE > 0.0 and abs(frac) < ALGO_LL_GATE:
        blended = rev_av
    else:
        blended = ALGO_LL_W * ll_av + (1.0 - ALGO_LL_W) * rev_av
    av = float(np.clip(blended, -cap, cap))

    hs = 0.0
    if HEDGE:
        rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
        betas = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
        hs = -((pos[1:] * cur[1:]) @ betas) / cur[0]
    room = max(cap - abs(av), 0.0)
    pos[0] = av + float(np.clip(hs, -room, room))

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
