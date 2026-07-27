"""
================================================================================
###   SAFE_llmatch.py   ·   ALGO leg = LEAD-LAG, VOLUME-MATCHED (no gate)     ###
================================================================================
  >>> Sibling of SAFE_llalgo / SAFE_lldollar. SAME idio book (49 names).       <<<
  >>> The ALGO index leg is driven PURELY by lead-lag and SIZED to conviction. <<<

The shipped LLALGO/LLDOLLAR books gate the index leg: on high-skew days they slam
the FULL $100k in the book-skew direction, and on the ~72% of low-skew days they
fall back to fading ALGO's 30-day move (reversion). On the 751-1000 draw it was the
reversion-fallback days that bled the leg (-$29k) and cratered the score (694 -> 452).

This book removes the gate and the reversion fallback entirely. Every day it puts the
index in the SAME NET-$ DIRECTION AND (scaled) SIZE as the idio book's own tilt:

    net_dol = sum(idio_position_i * price_i)      # the book's signed $ market bet
    ALGO_$  = clip( MATCH_K * net_dol , -$100k, +$100k )

i.e. "match the volume the lead-lag signal is already taking in the stocks and put
that into the index." Exposure scales with conviction, so low-signal days cost little
and no single reversion day can dominate.

  MATCH_K = 1.0  — the natural 1:1 match: the index takes exactly the book's own net
  predicted-$ tilt (no arbitrary amplification). It is a size DIAL, not a threshold;
  robustness across 61 rolling 250-day draws (compute_diagnostics.py):
     k=0.0 -> index leg OFF (pure idio book)   rolling mean 651 / floor 493   OLD 585 / NEW 586
     k=1.0 -> THIS BOOK (1:1 match)            rolling mean 657 / floor 482   OLD 564 / NEW 600
     k=1.5 -> mild amplification               rolling mean 655 / floor 471   OLD 539 / NEW 608
     k=2.0 -> new-draw-tilted                  rolling mean 649 / floor 447   OLD 512 / NEW 618
  k=1.0 has the best rolling MEAN and a high FLOOR — the robust choice. Raising k trades
  old-window score for new-window score and erodes the floor; size k on the rolling FLOOR,
  not on any single window's score.

The idio leg (instruments 1..49) is byte-identical to SAFE.py.
================================================================================
"""
import numpy as np

BOOK      = "SAFE · LL-MATCH (volume-matched lead-lag index leg)"
GATE_KIND = "none — index $ = MATCH_K * net-$ book skew, clipped to the $100k cap"

HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
HEDGE       = False
WARMUP      = 96

MATCH_K     = 1.0       # <-- size dial: index $ = MATCH_K * net-$ of the idio book (0 = leg off)
                        #     1.0 = 1:1 match to the book's predicted tilt (robust); 1.5 = mild boost

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

    # ---- idio leg: lead-lag ridge ensemble + cross-sectional reversion blend (== SAFE.py)
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

    # ---- ALGO index leg: match the book's net-$ tilt, scaled by MATCH_K --------
    cap = dlr[0] / cur[0]
    idio_lim = (dlr[1:] / cur[1:]).astype(int)
    idio_int = np.clip(pos[1:], -idio_lim, idio_lim).astype(int)
    net_dol = float((idio_int * cur[1:]).sum())          # signed $ the lead-lag book is holding
    av = float(np.clip(MATCH_K * net_dol / cur[0], -cap, cap))

    hs = 0.0
    if HEDGE:
        rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
        betas = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
        hs = -((pos[1:] * cur[1:]) @ betas) / cur[0]
    room = max(cap - abs(av), 0.0)
    pos[0] = av + float(np.clip(hs, -room, room))

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
