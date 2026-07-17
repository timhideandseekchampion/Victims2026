"""Speculative / borderline STANDALONE overlays — quarantined from the validated core.

These are the "a little correlation and a p-stat" findings from the research hunts,
kept OUT of the main strategy (validate_oos.py build_getpos) so the validated book +
reversion + hedge stay clean. They are wired into the OOS harness only through config
flags (drift_tilt / index_spread / coint_pairs) and adjudicated by the pre-registered
ship rule when real out-of-sample data lands. NONE is in the live submission.

In-sample verdicts (days 1-500, all correctly NEGATIVE — see docstrings): they exist to
be TESTED forward, not because they help today. drift_tilt in particular is a loaded
drift-detector reading ~0 because idiosyncratic drift is exactly 0 in-sample; it only
turns +EV if forward drift appears in 501-750/751-1000.
"""
import numpy as np


def drift_tilt_forecast(w, ret, tilt, win):
    """Tilt the sizing forecast toward each asset's cross-sec-DEMEANED trailing drift.

    Idiosyncratic drift-continuation, market-neutral by construction (NOT market beta).
    In-sample EV ~ 0 (idio drift = 0, so this only burns commission); +EV iff forward
    drift is nonzero. Returns a modified copy of the forecast vector `w` (tradeable assets).
    """
    if tilt == 0:
        return w
    dr = ret[1:, -win:].mean(1)                 # per-asset trailing mean log-return (drift proxy)
    dr = dr - dr.mean()                         # cross-sec demean -> idiosyncratic only
    drz = dr / (dr.std() + 1e-12)
    return w + tilt * drz * (np.std(w) + 1e-12)  # add in w's own units so tilt=1 ~ equal weight


def index_spread_positions(pos, prc, dollars):
    """Long ALGO index / short the equal-weight constituent basket, IN PLACE on `pos`.

    In-sample the spread is a real signal (t=3.47, both halves, Sharpe 2.39) BUT its
    short-basket leg competes with the book for the $10k caps => Score-NEGATIVE. OOS
    re-tests whether more data overturns that (mechanism is structural, so unlikely).
    """
    if dollars <= 0:
        return
    nStk = pos.shape[0] - 1
    pos[0] += dollars / prc[0, -1]
    pos[1:] -= (dollars / nStk) / prc[1:, -1]


def coint_pairs_positions(pos, lp, prc, cache, t, dollars):
    """Engle-Granger cointegration pairs overlay, IN PLACE on `pos`.

    Pairs (re)selected from PAST data only every 25 days (leakage-safe), each spread
    faded toward its rolling mean. In-sample standalone Sharpe 2.58 but phase-sensitive
    (+-13 Score from reselection timing) => not a robust Score add. `cache` is the
    caller's per-run dict (holds selected pairs + last-selection day).
    """
    if dollars <= 0 or t < 255:
        return
    from statsmodels.tsa.stattools import coint
    LB, CORRMIN, PMAX, KMAX, ZW, ZC, RESEL = 250, 0.6, 0.02, 15, 60, 2.0, 25
    if cache.get("pairs") is None or (t - cache.get("pair_t", -10**9)) >= RESEL:
        nn = prc.shape[0] - 1
        loo = max(0, t - LB)
        cand = []
        for i in range(nn):
            for j in range(i + 1, nn):
                a, b = lp[i + 1, loo:t], lp[j + 1, loo:t]
                if abs(np.corrcoef(a, b)[0, 1]) < CORRMIN:
                    continue
                try:
                    _, pv, _ = coint(a, b)
                except Exception:
                    continue
                if pv < PMAX:
                    cand.append((i + 1, j + 1, pv))
        cand.sort(key=lambda x: x[2])
        cache["pairs"] = [(i, j) for i, j, _ in cand[:KMAX]]
        cache["pair_t"] = t
    cur = prc[:, -1]
    for i, j in cache["pairs"]:
        beta = np.polyfit(lp[j, -ZW:], lp[i, -ZW:], 1)[0]
        spr = lp[i] - beta * lp[j]
        z = (spr[-1] - spr[-ZW:].mean()) / (spr[-ZW:].std() + 1e-12)
        u = -np.clip(z, -ZC, ZC) / ZC
        pos[i] += u * dollars / cur[i]
        pos[j] -= u * beta * dollars / cur[j]
