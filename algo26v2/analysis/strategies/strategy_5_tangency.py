"""Algothon 2026 strategy: max-Sharpe tangency blend of all edges
Self-contained getMyPosition. numpy only. Mean-reversion book:
pairs+ALGO+corr+lead weighted by the max-Sharpe (tangency) solution.
"""
import numpy as np

nInst = 51
ALGO = 0
PAIRS = [(1, 20), (13, 45), (7, 40), (10, 46), (8, 27), (25, 37), (31, 43), (18, 35), (23, 35), (27, 35), (30, 42), (8, 35), (28, 35), (26, 33), (9, 20), (18, 42), (19, 35), (36, 41), (18, 33), (12, 33), (12, 35), (35, 42), (15, 37), (14, 50), (36, 50), (28, 49), (6, 48), (33, 42), (14, 20), (14, 36), (41, 50), (33, 35), (35, 50), (1, 14), (11, 18), (35, 41), (11, 35), (42, 47), (18, 26), (35, 36), (30, 33), (11, 42), (26, 42), (12, 26), (14, 49), (35, 49), (28, 50), (22, 44), (6, 42), (18, 32), (19, 33), (32, 42), (42, 43), (12, 18), (18, 27), (28, 42), (5, 21), (33, 40), (11, 33), (36, 49), (5, 10), (14, 35), (26, 35), (32, 33), (9, 14), (24, 49), (33, 47), (6, 35), (14, 41), (18, 19), (12, 42), (5, 46), (41, 49), (6, 33), (14, 34), (4, 44), (28, 33), (24, 36), (35, 48), (33, 48), (7, 33), (23, 36), (1, 50), (31, 42), (21, 33), (26, 32), (13, 39), (32, 35), (6, 30), (29, 35), (8, 41), (42, 48), (17, 19), (18, 48)]
CFG = {'plb': 90, 'pentry': 1.25, 'pexit': 0.3, 'pdollars': 10000, 'adollars': 100000, 'clb': 90, 'centry': 1.0, 'cdollars': 6000, 'llb': 60, 'ldollars': 4000, 'xh': 10, 'xdollars': 5000, 'mflb': 60, 'mfk': 3, 'mfdollars': 5000, 'w_pairs': 7, 'w_algo': 6.7, 'w_corr': 1.0, 'w_lead': 6.4, 'w_xs': 0, 'w_mf': 0}

_pstate = {}


def _mn(sig, cur, dollars):
    sig = sig - sig.mean(); s = np.abs(sig).sum()
    return (sig / s) * dollars * nInst / cur if s > 1e-12 else np.zeros(len(cur))


def getMyPosition(prcSoFar):
    prc = np.asarray(prcSoFar, float); n, t = prc.shape
    if t < 3: return np.zeros(n, dtype=int)
    logp = np.log(prc); cur = prc[:, -1]; c = CFG
    pos = np.zeros(n)

    # --- cointegration pairs (fixed identities, rolling beta, OU hysteresis) ---
    if c["w_pairs"] and t > c["plb"] + 2:
        for i, j in PAIRS:
            beta = np.polyfit(prc[j, -c["plb"]:], prc[i, -c["plb"]:], 1)[0]
            spread = prc[i, :] - beta * prc[j, :]
            w = spread[-c["plb"]:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
            st = _pstate.get((i, j), 0)
            if st == 0 and abs(z) > c["pentry"]: st = -int(np.sign(z))
            elif st != 0 and abs(z) < c["pexit"]: st = 0
            _pstate[(i, j)] = st
            if st:
                pos[i] += c["w_pairs"] * st * c["pdollars"] / cur[i]
                pos[j] += -c["w_pairs"] * st * beta * c["pdollars"] / cur[j]

    # --- ALGO own 5-day mean-reversion ---
    if c["w_algo"] and t > 6:
        r = np.log(prc[ALGO, -1] / prc[ALGO, -6])
        pos[ALGO] += -c["w_algo"] * np.sign(r) * c["adollars"] / cur[ALGO]

    # --- correlation-vs-ALGO residual reversion (ALGO-hedged) ---
    if c["w_corr"] and t > c["clb"] + 2:
        la = logp[ALGO, -c["clb"]:]; leg = 0.0
        for i in range(1, n):
            b = np.polyfit(la, logp[i, -c["clb"]:], 1)[0]
            res = logp[i, :] - b * logp[ALGO, :]
            w = res[-c["clb"]:]; z = (res[-1] - w.mean()) / (w.std() + 1e-9)
            if abs(z) > c["centry"]:
                sh = -c["w_corr"] * np.sign(z) * c["cdollars"] / cur[i]
                pos[i] += sh; leg += -sh * b * cur[i] / cur[ALGO]
        pos[ALGO] += leg

    # --- lead-lag cross-prediction ---
    if c["w_lead"] and t > c["llb"] + 3:
        R = np.diff(logp[:, -c["llb"]:], axis=1); last = R[:, -1]
        A = R[:, 1:] - R[:, 1:].mean(1, keepdims=True)
        B = R[:, :-1] - R[:, :-1].mean(1, keepdims=True)
        xc = (A @ B.T) / (np.sqrt((A**2).sum(1)[:, None] * (B**2).sum(1)[None, :]) + 1e-12)
        np.fill_diagonal(xc, 0)
        sig = np.array([xc[i, np.argmax(np.abs(xc[i]))] * last[np.argmax(np.abs(xc[i]))] for i in range(n)])
        pos += c["w_lead"] * _mn(sig, cur, c["ldollars"])

    # --- cross-sectional 10-day reversal ---
    if c["w_xs"] and t > c["xh"] + 1:
        r = np.log(prc[:, -1] / prc[:, -1 - c["xh"]])
        pos += c["w_xs"] * _mn(-r, cur, c["xdollars"])

    # --- PCA multi-factor residual reversion ---
    if c["w_mf"] and t > c["mflb"] + 2:
        Rm = np.diff(logp[:, -c["mflb"]:], axis=1).T; Rc = Rm - Rm.mean(0)
        _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
        comp = Vt[:c["mfk"]]; lastc = Rc[-1]
        resid = lastc - comp.T @ (comp @ lastc)
        pos += c["w_mf"] * _mn(-resid, cur, c["mfdollars"])

    return pos.astype(int)
