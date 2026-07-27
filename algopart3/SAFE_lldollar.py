"""
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
$$$   SAFE_lldollar.py   ·   ALGO GATE = NET  $  BOOK SKEW   ·   MAX sizing (full $100k)   $$$
$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
  >>> THIS IS THE **DOLLAR** SIBLING. gate reads a $ amount (net$ book skew).  <<<
  >>> its twin SAFE_llalgo.py reads a NAME COUNT (|frac|>=0.12, 28 vs 22).     <<<
  Both are the SAME edge & SAME score (696 graded / 6.06 Sharpe); only the gate UNIT differs.

Sibling of SAFE_llalgo.py. Identical idio leg; the ALGO (index) leg's long/short SWITCH is
expressed as the net DOLLAR skew of the stock book instead of a name count.

  net$ = sum(idio_position_i * price_i)      # the signed $ exposure sitting in the 50 stocks
  when |net$| >= ALGO_LL_DOLLAR  ->  the book carries a real market bet; transplant it into ALGO:
      go FULL $100k long/short in the direction of net$
  otherwise  ->  fall back to fading ALGO's 30-day move (reversion default)

WHY $50k (not $60k): net$ and the frac name-count are the SAME signal (corr 1.00); a 6-name skew
(28 vs 22) is ~$60k, but its MEDIAN is $59,885 -- so a hard $60k cut randomly drops ~half the genuine
6-name days to reversion (-13 on the graded leg). A $50k cut cleanly separates the 4-name (~$40k) from
the 6-name (~$60k) skew, reproducing the validated |frac|>=0.12 gate EXACTLY (graded 696, Sharpe 6.06;
rolling mean 687, beats baseline 32/38). See ll_dollar_gate.py.
==========================================================================================
"""
import numpy as np

# ---- book identity (so you never confuse this with SAFE_llalgo) --------------
BOOK      = "SAFE · LL-DOLLAR ($)"
GATE_KIND = "net-$ book skew  |net$| >= $50k  ->  FULL $100k ALGO (MAX)"
SIBLING   = "SAFE_llalgo.py — same edge gated by NAME COUNT |frac|>=0.12"

HALF_LIVES  = (250, 500, 1000, 2000)
RIDGE_A     = 0.1
BLEND       = 0.3
REV_W       = 10
CONTRA_DOL  = 1_000_000
CONTRA_K    = 30
CONTRA_WZ   = 60
HEDGE       = False
WARMUP      = 96

ALGO_LL_DOLLAR = 50_000  # <-- net $-book skew that flips the ALGO leg to lead-lag (reproduces 28v22)
                         #     0 disables the gate (always lead-lag); large disables it (always reversion)
                         # SIZING = MAX: on a trigger the leg goes the FULL $100k (binary switch, not
                         # matched to the skew) — the max-score choice; matching net$ trades ~8pts of
                         # score for +0.15 Sharpe (ll_dollar_gate.py / sizing test), deliberately NOT used.

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

    # ---- net dollar skew of the stock book (the market bet sitting in the 50 names) ----
    idio_lim = (dlr[1:] / cur[1:]).astype(int)
    idio_int = np.clip(pos[1:], -idio_lim, idio_lim).astype(int)
    net_dol = float((idio_int * cur[1:]).sum())

    # ---- ALGO index leg: gate on |net$|; transplant the book skew into the market ------
    cap = dlr[0] / cur[0]
    if ALGO_LL_DOLLAR > 0 and abs(net_dol) >= ALGO_LL_DOLLAR:
        av = float(np.sign(net_dol) * cap)                # full $100k in the book-skew direction
    else:
        lpA = logp[0]; mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        z = (mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12)
        av = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (CONTRA_DOL / cur[0]), -cap, cap))

    hs = 0.0
    if HEDGE:
        rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
        betas = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
        hs = -((pos[1:] * cur[1:]) @ betas) / cur[0]
    room = max(cap - abs(av), 0.0)
    pos[0] = av + float(np.clip(hs, -room, room))

    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
