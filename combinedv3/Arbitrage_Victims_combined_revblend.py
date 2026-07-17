"""[PRESERVED VARIANT — not the primary submission] Reversion-blend combinedv3.

This is the earlier combinedv3 draft, kept on request. It adds a cross-sectional
short-reversion signal (SIGNAL 2 below) into the ridge forecast at 20% weight. On this
one 500-day sample it scores marginally higher (766 vs 763 @250), but per algo26v1's
FINDINGS.md that +3 is BELOW the ±110/day fresh-window noise floor, plain reversal is
~3x weaker than the lead-lag edge, and signal-blending was tested-dead in hunt #3 — so
the primary submission (Arbitrage_Victims_combined.py) deliberately excludes it. Use this
only as an aggressive A/B variant, never as the final entry without real OOS confirmation.

--- original docstring ---
Algothon 2026 — COMBINED v3 submission (Arbitrage Victims).

A three-signal ensemble, self-contained (numpy only). Backtests to eval.py
Score ~766 (last-250 days) / ~614 (full days 60-500), Sharpe ~7.1 / ~5.8.

Each day the book is built from three orthogonal edges, then risk-managed:

  SIGNAL 1 — peer lead-lag (idio, core). Predict every tradeable name's next-day
             return from today's full 51-name cross-section with a forgetting-
             weighted ridge (EWLS, half-life 2000d ~= expanding window on a proven-
             stationary DGP, light L2). Demean it => market-neutral, z-score it.
  SIGNAL 2 — cross-sectional short reversion (idio). z-score of the negative
             trailing REV_W-day return, demeaned across names. Blended into signal 1
             at weight BLEND. This is the *combination* that lifts v3 over v4: it
             raises BOTH scored windows AND the worst rolling-fold floor
             (514.9 vs 478.6) — a robustness gain, not window-fitting.
  SIGNAL 3 — ALGO index contrarian (market). The index (inst 0) mean-reverts over
             multi-day horizons; fade its recent CONTRA_K-day move, sized off its
             special $100k / 0.2bp capacity. Takes first claim on the $100k cap.

  RISK      — conviction gate (only trade names whose |signal| clears CONV_Z * daily
             cross-sectional std; the traded count floats ~32-47/day) + a residual
             beta-hedge with ALGO applied LAST into whatever ALGO cap room is left.

Knobs are the winners of a two-window (250 & 440) + 5-fold rolling-window sweep;
see README.md in this directory for the full comparison table.
"""
import numpy as np

HALF_LIFE = 2000     # EWLS forgetting half-life; ~expanding window on the stationary DGP
ALPHA = 0.1          # light ridge shrinkage on the 51x50 fit
LIMIT = 10_000       # per-asset dollar position limit
ALGO_LIMIT = 100_000 # ALGO (index) dollar position limit — special 10x cap
CONV_Z = 0.2         # conviction gate: trade a name only if |signal| >= CONV_Z * x-sectional std
BLEND = 0.2          # weight on the cross-sectional-reversion signal vs the ridge forecast
REV_W = 10           # trailing window (days) for the reversion signal
HEDGE = True         # residual-beta neutralize with ALGO (applied last, into leftover cap room)
CONTRA_DOLLARS = 200_000  # ALGO contrarian notional (Score-saturating plateau floor)
CONTRA_K = 30        # lookback (days) for the ALGO move we fade
CONTRA_WZ = 60       # window to z-score that move

_cache = {"fit_t": None, "model": None}


def _ewls_ridge_fit(X, Y):
    """Exponentially-weighted ridge, weighted-demean form. Returns (B, mx, my)."""
    n, p = X.shape
    lam = 0.5 ** (1.0 / HALF_LIFE)
    w = lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw
    my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc)
    XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + ALPHA) * np.eye(p), XtWY)
    return B, mx, my


def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    pos = np.zeros(nInst)
    if t < 60:                                   # warm-up before fitting
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]                 # daily log returns (nInst, t-1)
    if _cache["fit_t"] != t:                     # refit keyed to exactly this t (no lookahead)
        X = ret[:, :-1].T                        # today's cross-section (all 51)
        Y = ret[1:, 1:].T                        # next-day return of the 50 tradeable assets
        _cache["model"] = _ewls_ridge_fit(X, Y)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]

    # --- SIGNAL 1: peer lead-lag ridge forecast, market-neutral, z-scored ---
    pred = my + (ret[:, -1] - mx) @ B            # next-day forecast (50,)
    w = pred - pred.mean()
    wz = w / (np.std(w) + 1e-12)

    # --- SIGNAL 2: cross-sectional short-horizon reversion, blended in ---
    r = ret[1:, -REV_W:].sum(1)                  # trailing REV_W-day return per name
    r = r - r.mean()                             # cross-sectional demean -> idiosyncratic
    revz = -r / (np.std(r) + 1e-12)
    sig = (1 - BLEND) * wz + BLEND * revz

    sized = np.sign(sig) * (LIMIT / prcSoFar[1:, -1])   # MAX sizing on the 50 assets
    if CONV_Z > 0:                               # conviction gate (floating name count)
        keep = np.abs(sig) >= CONV_Z * (np.std(sig) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized

    # --- SIGNAL 3: ALGO index contrarian (reversion gets first claim on the $100k cap) ---
    cap_sh = ALGO_LIMIT / prcSoFar[0, -1]
    rev_sh = 0.0
    if CONTRA_DOLLARS > 0 and t > CONTRA_K + CONTRA_WZ + 2:
        lpA = np.log(prcSoFar[0])
        move = lpA[CONTRA_K:] - lpA[:-CONTRA_K]              # rolling K-day ALGO returns
        z = (move[-1] - move[-CONTRA_WZ:].mean()) / (move[-CONTRA_WZ:].std() + 1e-12)
        rev_sh = -float(np.clip(z, -3, 3)) * CONTRA_DOLLARS / prcSoFar[0, -1]
    rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))

    # --- RISK: residual beta-hedge with ALGO, applied LAST into leftover cap room ---
    hedge_sh = 0.0
    if HEDGE:
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
    room = max(cap_sh - abs(rev_sh), 0.0)
    pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
    return pos.astype(int)
