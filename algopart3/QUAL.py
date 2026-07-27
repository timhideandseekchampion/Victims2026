"""
=========================  QUAL.py — THE FINAL / CATCH-UP BOOK  =========================
USE THIS FOR THE FINAL (days 1500-2000), or the 1000-1500 qualifier ONLY if you need to
maximise expected score to catch a podium — NOT to merely clear the top-10 bar (use SAFE for
that; it has the higher floor). Higher true MEAN than SAFE with the ensemble's low variance;
better floor + lower variance than SWING. This is the highest-EV robust book for a LONG
(250d/500d) grading window, where score converges to the config's true mean.
  config: HL-ensemble(250/500/1000/2000), RIDGE_A=0.1, BLEND=0.20, CONTRA_DOL=1M, HEDGE=False.

WHY 0.20 (validated this session, see DECISION.md):
  Over 500-day windows (the 1000-1500 / 1500-2000 grade length) score ~= true mean PnL because
  day-to-day variance washes out. Ranked by mean on 500d windows: b.20=618 > SWING(hl1000,b.15)=613
  > SAFE(b.30)=600, and b.20 also has the best floor of that group (517). BLEND=0.20 sits on the
  same flat plateau as SAFE(0.30)/SWING(0.15) — not a fragile peak. The ONE caveat: on a
  lead-lag-WEAK draw (like the 500-750 leg) the heavier-blend SAFE(b.30) wins (613 vs 549), which
  is why SAFE remains the pick when the objective is floor/survival rather than mean.

Everything else is identical to SAFE.py. Self-contained (numpy only).
========================================================================================
"""
import numpy as np

# ------------------------------------------------------------------ knobs
HALF_LIVES  = (250, 500, 1000, 2000)  # ENSEMBLE of memories -> lower estimation variance, higher floor
RIDGE_A     = 0.1       # L2 on the 51->50 coefficient matrix
BLEND       = 0.20      # reversion weight: 0.20 maximises across-(long)-window MEAN (vs SAFE's 0.30 floor pick)
REV_W       = 10        # reversion lookback (days)
CONTRA_DOL  = 1_000_000 # ALGO fade notional (pins the $100k cap; validated: 1M > 500k on mean AND floor AND the leg)
CONTRA_K    = 30        # ALGO move lookback we fade
CONTRA_WZ   = 60        # window to z-score that move
HEDGE       = False     # OFF: fade pins the $100k cap so the hedge gets ~$0 room (see DECISION.md note)
WARMUP      = 96        # need enough history for the ridge

_DLR = None             # per-instrument dollar limits, lazily sized to nInst


def _limits(nInst):
    global _DLR
    if _DLR is None or len(_DLR) != nInst:
        _DLR = np.full(nInst, 10_000.0); _DLR[0] = 100_000.0
    return _DLR


def _ewls_ridge(X, Y, hl, a):
    """Forgetting-weighted ridge: predict Y (t+1 idio returns) from X (t returns, all names)."""
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
    r = logp[:, 1:] - logp[:, :-1]                        # (nInst, t-1) log returns

    # ---- lead-lag ridge forecast: ENSEMBLE across half-lives (variance reduction) -
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = _ewls_ridge(r[:, :-1].T, r[1:, 1:].T, hl, RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))                # z-score each memory then average
    wz = np.mean(fs, 0)

    # ---- blend in cross-sectional reversion (orthogonal diversifier) -------------
    if BLEND > 0:
        rr = logp[1:, -1] - logp[1:, -1 - REV_W]
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        wz = (1 - BLEND) * wz + BLEND * rv

    # ---- idio leg: trade ALL names at full $10k, sign-sized (breadth) ------------
    pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])

    # ---- ALGO index leg: fade the 30-day move, pinned to the $100k cap -----------
    cap = dlr[0] / cur[0]
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

    # integer shares, clipped to dollar limits (grader re-clips anyway)
    lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)
