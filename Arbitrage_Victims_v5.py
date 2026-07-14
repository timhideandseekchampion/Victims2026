"""Algothon 2026 submission (v5) — v4 with CONTRA_WZ 60->20 and the hedge OFF.

*** READ THIS BEFORE SHIPPING v5 ***
These two changes were found by sliding the X-ray on the LAST-250 window (days 251-500).
They maximize the Score on THAT window (763 -> 795) but FAIL out-of-sample validation:
  - WZ=20 scores -94 vs v4 on the held-out FIRST-190 days (days 60-250);
  - full-window (60-500) v5 = 573 vs v4's 593 (WORSE by 20);
  - paired PnL vs v4 sign-flips across halves (half1 -86/day, half2 +49/day) -> not robust;
  - hedge=off leaves ~$31k/day of unhedged market beta = uncompensated directional variance.
By PBO (=90% on this candidate set) and the k=0.75 / both-halves gate, this is the classic
overfitting signature: a bigger number on the tuned window, worse everywhere else. It is tuned
to days 251-500, but the live board scores 501-750 and the prize scores 1001-1500 — DIFFERENT
windows — so the +32 may not even appear live.

v4 (Arbitrage_Victims_v4.py) remains the robustness-recommended submission. Keep v5 only as an
aggressive/leaderboard variant, or to A/B-test on the live board (a genuine OOS check). Do not make
it the final 1001-1500 entry unless real out-of-sample data confirms WZ=20 + no-hedge beat v4.
"""
import numpy as np

HALF_LIFE = 2000
ALPHA = 0.1
LIMIT = 10_000
ALGO_LIMIT = 100_000
HEDGE = False        # v5: hedge OFF (was True). Leaves residual market beta unhedged (~$31k/day).
CONTRA_DOLLARS = 200_000
CONTRA_K = 30
CONTRA_WZ = 34       # v5: z-window 60 -> 20 (shorter/more reactive). Overfits recent moves; -94 OOS on held-out data.
CONV_Z = 0.1

_cache = {"model": None, "fit_t": None}


def _ewls_ridge_fit(X, Y):
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
    if t < 60:
        return pos
    lp = np.log(prcSoFar)
    ret = lp[:, 1:] - lp[:, :-1]
    if _cache["fit_t"] != t:
        X = ret[:, :-1].T
        Y = ret[1:, 1:].T
        _cache["model"] = _ewls_ridge_fit(X, Y)
        _cache["fit_t"] = t
    B, mx, my = _cache["model"]
    pred = my + (ret[:, -1] - mx) @ B
    w = pred - pred.mean()
    sized = np.sign(w) * (LIMIT / prcSoFar[1:, -1])
    if CONV_Z > 0:
        keep = np.abs(w) >= CONV_Z * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
    pos[1:] = sized
    cap_sh = ALGO_LIMIT / prcSoFar[0, -1]
    rev_sh = 0.0
    if CONTRA_DOLLARS > 0 and t > CONTRA_K + CONTRA_WZ + 2:
        lpA = np.log(prcSoFar[0])
        move = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
        z = (move[-1] - move[-CONTRA_WZ:].mean()) / (move[-CONTRA_WZ:].std() + 1e-12)
        rev_sh = -float(np.clip(z, -3, 3)) * CONTRA_DOLLARS / prcSoFar[0, -1]
    rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))
    hedge_sh = 0.0
    if HEDGE:
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
    room = max(cap_sh - abs(rev_sh), 0.0)
    pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
    return pos.astype(int)
