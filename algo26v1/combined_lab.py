#!/usr/bin/env python
"""Lab: parameterized COMBINED strategy factory + sweep, scored on the exact eval.py loop.

Idio leg  = ensemble of (a) EWLS ridge peer-lead-lag forecast and (b) cross-sectional
            short-horizon reversion z-score, blended by BLEND, MAX-sized with a conviction gate.
ALGO leg  = contrarian K-day reversion overlay (first claim on $100k cap).
Hedge     = residual-beta neutralize with ALGO into leftover cap room.
"""
import numpy as np, pandas as pd

prcAll = pd.read_csv("./prices.txt", sep=r"\s+", header=0, index_col=None).values.T
nInst, nt = prcAll.shape
commRate = np.full(nInst, 0.0001); commRate[0] = 0.00002
dlrPosLimit = np.full(nInst, 10_000); dlrPosLimit[0] = 100_000


def score(mu, sigma, param=1.0):
    if mu <= 0 or sigma < 1e-10:
        return mu
    sr = np.sqrt(250) * mu / sigma
    return mu * sr**2 / (sr**2 + param**2)


def _ewls_ridge_fit(X, Y, half_life, alpha):
    n, p = X.shape
    lam = 0.5 ** (1.0 / half_life)
    w = lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw
    my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc)
    XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + alpha) * np.eye(p), XtWY)
    return B, mx, my


def make_getpos(half_life=2000, alpha=0.1, conv_z=0.2, blend=0.0, rev_w=5,
                contra_dollars=200_000, contra_k=30, contra_wz=60, hedge=True,
                limit=10_000, algo_limit=100_000):
    cache = {"fit_t": None, "model": None}

    def getMyPosition(prcSoFar):
        ni, t = prcSoFar.shape
        pos = np.zeros(ni)
        if t < 60:
            return pos
        lp = np.log(prcSoFar)
        ret = lp[:, 1:] - lp[:, :-1]
        if cache["fit_t"] != t:
            X = ret[:, :-1].T
            Y = ret[1:, 1:].T
            cache["model"] = _ewls_ridge_fit(X, Y, half_life, alpha)
            cache["fit_t"] = t
        B, mx, my = cache["model"]
        pred = my + (ret[:, -1] - mx) @ B
        w = pred - pred.mean()
        wz = w / (np.std(w) + 1e-12)
        # cross-sectional short reversion z (idio) as an ensemble partner
        if blend > 0:
            r = ret[1:, -rev_w:].sum(1)          # trailing rev_w-day return per name
            r = r - r.mean()
            revz = -r / (np.std(r) + 1e-12)
            sig = (1 - blend) * wz + blend * revz
        else:
            sig = wz
        sized = np.sign(sig) * (limit / prcSoFar[1:, -1])
        if conv_z > 0:
            keep = np.abs(sig) >= conv_z * (np.std(sig) + 1e-12)
            sized = np.where(keep, sized, 0.0)
        pos[1:] = sized
        cap_sh = algo_limit / prcSoFar[0, -1]
        rev_sh = 0.0
        if contra_dollars > 0 and t > contra_k + contra_wz + 2:
            lpA = np.log(prcSoFar[0])
            move = lpA[contra_k:] - lpA[:-contra_k]
            z = (move[-1] - move[-contra_wz:].mean()) / (move[-contra_wz:].std() + 1e-12)
            rev_sh = -float(np.clip(z, -3, 3)) * contra_dollars / prcSoFar[0, -1]
        rev_sh = float(np.clip(rev_sh, -cap_sh, cap_sh))
        hedge_sh = 0.0
        if hedge:
            rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
            betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
            net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
            hedge_sh = -net_beta / prcSoFar[0, -1]
        room = max(cap_sh - abs(rev_sh), 0.0)
        pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
        return pos.astype(int)

    return getMyPosition


def calcPL(getPosition, numTestDays):
    cash = 0; curPos = np.zeros(nInst); totDV = 0; value = 0; comm = 0
    pll = []; startDay = nt - numTestDays
    for t in range(startDay, nt + 1):
        prc = prcAll[:, :t]; cur = prc[:, -1]
        if t < nt:
            npos = getPosition(prc)
            lim = (dlrPosLimit / cur).astype(int)
            npos = np.clip(npos, -lim, lim).astype(int)
        else:
            npos = np.array(curPos)
        d = npos - curPos
        cash -= cur.dot(d) + comm
        dv = cur * np.abs(d); totDV += np.sum(dv); comm = np.sum(dv * commRate)
        curPos = np.array(npos)
        pl = cash + curPos.dot(cur) - value
        value = cash + curPos.dot(cur)
        if t > startDay:
            pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sh = np.sqrt(250) * mu / sd if sd > 0 else 0.0
    return mu, sd, sh, score(mu, sd), pll


def evaluate(windows=(250, 440), folds=None, **knobs):
    out = {}
    for w in windows:
        _, _, sh, sc, _ = calcPL(make_getpos(**knobs), w)
        out[w] = (sc, sh)
    return out


if __name__ == "__main__":
    print("Baseline v4 config (blend=0):")
    print("  ", evaluate())
