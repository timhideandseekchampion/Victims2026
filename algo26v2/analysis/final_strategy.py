"""Algothon 2026 - combined mean-reversion book.

getMyPosition(prcSoFar) stacks the edges that survived out-of-sample testing,
all computed from PAST data only (no look-ahead) so it generalises to the
grading window:

  1. Cointegration pairs  - rolling Engle-Granger selection (p<0.01) on a
     correlation-prefiltered candidate set, traded with OU hysteresis bands
     (enter 1.0 sigma, exit 0.5 sigma). Primary edge.
  2. ALGO 5-day mean-reversion - fade ALGO's own 5-day move (instrument 0 has
     the $100k limit & 0.2bp commission, so it is cheap to size).
  3. Correlation-vs-ALGO residual reversion - factor stat-arb, ALGO-hedged.
  4. Lead-lag cross-prediction - each name from its best lag-1 cross-corr peer.
  5. Cross-sectional 10-day reversal - weekly-horizon reversal (1-day is noise).

Only numpy + statsmodels.coint are used (both in the grading sandbox).
"""
import numpy as np
from statsmodels.tsa.stattools import coint

nInst = 51
ALGO = 0

# ---- tunable weights ----
# Lean book chosen for ROBUSTNESS across sub-windows (module 25), not peak
# in-sample score: pairs + ALGO-timing + corr-vs-ALGO residual. Lead-lag and
# XS-reversal are dropped - they lifted the full-sample score but were fragile
# out-of-window and churned 3x the commission.
W_PAIRS, W_ALGO, W_CORR, W_LEAD, W_XS = 1.5, 2.0, 0.5, 0.0, 0.0

# ---- pair engine params ----
PAIR_P = 0.01
PAIR_SELWIN = 250
PAIR_RESELECT = 25
MAX_PAIRS = 16
PAIR_LB = 90
PAIR_ENTRY = 1.0
PAIR_EXIT = 0.5
PAIR_DOLLARS = 8000
CORR_PREFILTER = 0.4

# ---- overlay params ----
ALGO_H = 5
ALGO_DOLLARS = 40000
CORR_LB = 90
CORR_ENTRY = 0.9
CORR_DOLLARS = 3500
LEAD_LB = 60
LEAD_DOLLARS = 2500
XS_H = 10
XS_DOLLARS = 3000

# ---- persistent state ----
_pairs = {"day": -10**9, "list": []}
_pair_state = {}


def _mn_book(sig, cur, dollars):
    sig = sig - sig.mean()
    s = np.abs(sig).sum()
    return (sig / s) * dollars * nInst / cur if s > 1e-12 else np.zeros(len(cur))


def _select_pairs(logp):
    n, t = logp.shape
    win = logp[:, -min(t, PAIR_SELWIN):]
    rr = np.diff(win, axis=1)
    C = np.corrcoef(rr)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(C[i, j]) > CORR_PREFILTER:
                try:
                    p = coint(win[i], win[j])[1]
                    if p < PAIR_P:
                        beta = np.polyfit(win[j], win[i], 1)[0]
                        out.append((i, j, p, beta))
                except Exception:
                    pass
    out.sort(key=lambda x: x[2])
    return out[:MAX_PAIRS]


def _pairs_ou(prc, logp, cur):
    n, t = prc.shape
    pos = np.zeros(n)
    if t < PAIR_LB + 2:
        return pos
    if t - _pairs["day"] >= PAIR_RESELECT or not _pairs["list"]:
        _pairs["list"] = _select_pairs(logp)
        _pairs["day"] = t
    for i, j, _, beta in _pairs["list"]:
        spread = prc[i, :] - beta * prc[j, :]
        w = spread[-PAIR_LB:]
        z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        st = _pair_state.get((i, j), 0)
        if st == 0 and abs(z) > PAIR_ENTRY:
            st = -int(np.sign(z))
        elif st != 0 and abs(z) < PAIR_EXIT:
            st = 0
        _pair_state[(i, j)] = st
        if st:
            pos[i] += st * PAIR_DOLLARS / cur[i]
            pos[j] += -st * beta * PAIR_DOLLARS / cur[j]
    return pos


def _algo_timing(prc, cur):
    t = prc.shape[1]
    if t < ALGO_H + 1:
        return np.zeros(nInst)
    r = np.log(prc[ALGO, -1] / prc[ALGO, -1 - ALGO_H])
    pos = np.zeros(nInst)
    pos[ALGO] = -np.sign(r) * ALGO_DOLLARS / cur[ALGO]
    return pos


def _corr_algo(prc, logp, cur):
    n, t = prc.shape
    pos = np.zeros(n)
    if t < CORR_LB + 2:
        return pos
    la = logp[ALGO, -CORR_LB:]
    leg = 0.0
    for i in range(1, n):
        beta = np.polyfit(la, logp[i, -CORR_LB:], 1)[0]
        resid = logp[i, :] - beta * logp[ALGO, :]
        w = resid[-CORR_LB:]
        z = (resid[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > CORR_ENTRY:
            sh = -np.sign(z) * CORR_DOLLARS / cur[i]
            pos[i] += sh
            leg += -sh * beta * cur[i] / cur[ALGO]
    pos[ALGO] += leg
    return pos


def _leadlag(logp, cur):
    n, t = logp.shape
    if t < LEAD_LB + 3:
        return np.zeros(n)
    R = np.diff(logp[:, -LEAD_LB:], axis=1)
    last = R[:, -1]
    A = R[:, 1:] - R[:, 1:].mean(1, keepdims=True)
    B = R[:, :-1] - R[:, :-1].mean(1, keepdims=True)
    xc = (A @ B.T) / (np.sqrt((A ** 2).sum(1)[:, None] * (B ** 2).sum(1)[None, :]) + 1e-12)
    np.fill_diagonal(xc, 0)
    sig = np.array([xc[i, np.argmax(np.abs(xc[i]))] * last[np.argmax(np.abs(xc[i]))]
                    for i in range(n)])
    return _mn_book(sig, cur, LEAD_DOLLARS)


def _xs_reversal(prc, cur):
    if prc.shape[1] < XS_H + 1:
        return np.zeros(nInst)
    r = np.log(prc[:, -1] / prc[:, -1 - XS_H])
    return _mn_book(-r, cur, XS_DOLLARS)


def getMyPosition(prcSoFar):
    prc = np.asarray(prcSoFar, dtype=float)
    n, t = prc.shape
    if t < 2:
        return np.zeros(n, dtype=int)
    logp = np.log(prc)
    cur = prc[:, -1]
    pos = (W_PAIRS * _pairs_ou(prc, logp, cur)
           + W_ALGO * _algo_timing(prc, cur)
           + W_CORR * _corr_algo(prc, logp, cur)
           + W_LEAD * _leadlag(logp, cur)
           + W_XS * _xs_reversal(prc, cur))
    return pos.astype(int)
