"""Parameterized mean-reversion engine used to search for high-scoring configs.

All components read only past prices. A config dict selects components, weights,
and (critically) dollar sizing - the main lever on score, since
score = mean_PnL * SR^2/(SR^2+1) ~= mean_PnL at high Sharpe, and per-name
exposure is capped by the $ position limits.
"""
import numpy as np
from statsmodels.tsa.stattools import coint

nInst = 51
ALGO = 0


def _mn(sig, cur, dollars):
    sig = sig - sig.mean(); s = np.abs(sig).sum()
    return (sig / s) * dollars * nInst / cur if s > 1e-12 else np.zeros(len(cur))


class Engine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pairs = {"day": -10**9, "list": []}
        self.pstate = {}

    # ---- pair selection (rolling, correlation-prefiltered) ----
    def _select(self, logp):
        c = self.cfg; n, t = logp.shape
        win = logp[:, -min(t, c["sel_win"]):]
        rr = np.diff(win, axis=1); C = np.corrcoef(rr)
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(C[i, j]) > c["prefilter"]:
                    try:
                        p = coint(win[i], win[j])[1]
                        if p < c["pmax"]:
                            beta = np.polyfit(win[j], win[i], 1)[0]
                            out.append((i, j, p, beta))
                    except Exception:
                        pass
        out.sort(key=lambda x: x[2])
        return out[:c["max_pairs"]]

    def _pairs(self, prc, logp, cur):
        c = self.cfg; n, t = prc.shape; pos = np.zeros(n)
        if t < c["pair_lb"] + 2 or c.get("w_pairs", 0) == 0:
            return pos
        if c.get("fixed_pairs"):
            # fixed pair identities (from research); beta re-estimated rolling
            lst = []
            for i, j in c["fixed_pairs"]:
                beta = np.polyfit(prc[j, -c["pair_lb"]:], prc[i, -c["pair_lb"]:], 1)[0]
                lst.append((i, j, 0.0, beta))
            pair_list = lst
        else:
            if t - self.pairs["day"] >= c["reselect"] or not self.pairs["list"]:
                self.pairs["list"] = self._select(logp); self.pairs["day"] = t
            pair_list = self.pairs["list"]
        for i, j, _, beta in pair_list:
            spread = prc[i, :] - beta * prc[j, :]
            w = spread[-c["pair_lb"]:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
            st = self.pstate.get((i, j), 0)
            if st == 0 and abs(z) > c["pair_entry"]:
                st = -int(np.sign(z))
            elif st != 0 and abs(z) < c["pair_exit"]:
                st = 0
            self.pstate[(i, j)] = st
            if st:
                pos[i] += st * c["pair_dollars"] / cur[i]
                pos[j] += -st * beta * c["pair_dollars"] / cur[j]
        return pos

    def _algo(self, prc, cur):
        c = self.cfg; t = prc.shape[1]
        if c.get("w_algo", 0) == 0 or t < c["algo_h"] + 1: return np.zeros(nInst)
        r = np.log(prc[ALGO, -1] / prc[ALGO, -1 - c["algo_h"]])
        pos = np.zeros(nInst); pos[ALGO] = -np.sign(r) * c["algo_dollars"] / cur[ALGO]
        return pos

    def _corr(self, prc, logp, cur):
        c = self.cfg; n, t = prc.shape; pos = np.zeros(n)
        if c.get("w_corr", 0) == 0 or t < c["corr_lb"] + 2: return pos
        la = logp[ALGO, -c["corr_lb"]:]; leg = 0.0
        for i in range(1, n):
            beta = np.polyfit(la, logp[i, -c["corr_lb"]:], 1)[0]
            resid = logp[i, :] - beta * logp[ALGO, :]
            w = resid[-c["corr_lb"]:]; z = (resid[-1] - w.mean()) / (w.std() + 1e-9)
            if abs(z) > c["corr_entry"]:
                sh = -np.sign(z) * c["corr_dollars"] / cur[i]; pos[i] += sh
                leg += -sh * beta * cur[i] / cur[ALGO]
        pos[ALGO] += leg
        return pos

    def _multifactor(self, logp, cur):
        c = self.cfg; n, t = logp.shape
        if c.get("w_mf", 0) == 0 or t < c["mf_lb"] + 2: return np.zeros(n)
        R = np.diff(logp[:, -c["mf_lb"]:], axis=1).T
        Rc = R - R.mean(0)
        U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
        comp = Vt[:c["mf_k"]]; last = Rc[-1]
        resid = last - comp.T @ (comp @ last)
        return _mn(-resid, cur, c["mf_dollars"])

    def _lead(self, logp, cur):
        c = self.cfg; n, t = logp.shape
        if c.get("w_lead", 0) == 0 or t < c["lead_lb"] + 3: return np.zeros(n)
        R = np.diff(logp[:, -c["lead_lb"]:], axis=1); last = R[:, -1]
        A = R[:, 1:] - R[:, 1:].mean(1, keepdims=True)
        B = R[:, :-1] - R[:, :-1].mean(1, keepdims=True)
        xc = (A @ B.T) / (np.sqrt((A**2).sum(1)[:, None] * (B**2).sum(1)[None, :]) + 1e-12)
        np.fill_diagonal(xc, 0)
        sig = np.array([xc[i, np.argmax(np.abs(xc[i]))] * last[np.argmax(np.abs(xc[i]))] for i in range(n)])
        return _mn(sig, cur, c["lead_dollars"])

    def _xs(self, prc, cur):
        c = self.cfg
        if c.get("w_xs", 0) == 0 or prc.shape[1] < c["xs_h"] + 1: return np.zeros(nInst)
        r = np.log(prc[:, -1] / prc[:, -1 - c["xs_h"]])
        return _mn(-r, cur, c["xs_dollars"])

    def position(self, prc):
        c = self.cfg; n, t = prc.shape
        if t < 2: return np.zeros(n, dtype=int)
        logp = np.log(prc); cur = prc[:, -1]
        pos = (c.get("w_pairs", 0) * self._pairs(prc, logp, cur)
               + c.get("w_algo", 0) * self._algo(prc, cur)
               + c.get("w_corr", 0) * self._corr(prc, logp, cur)
               + c.get("w_mf", 0) * self._multifactor(logp, cur)
               + c.get("w_lead", 0) * self._lead(logp, cur)
               + c.get("w_xs", 0) * self._xs(prc, cur))
        return pos.astype(int)


DEFAULTS = dict(
    sel_win=250, prefilter=0.4, pmax=0.05, max_pairs=16, reselect=25,
    pair_lb=90, pair_entry=1.0, pair_exit=0.5, pair_dollars=8000,
    algo_h=5, algo_dollars=40000, corr_lb=90, corr_entry=0.9, corr_dollars=3500,
    mf_lb=60, mf_k=3, mf_dollars=3000, lead_lb=60, lead_dollars=2500,
    xs_h=10, xs_dollars=3000,
    w_pairs=1.0, w_algo=0.0, w_corr=0.0, w_mf=0.0, w_lead=0.0, w_xs=0.0,
)


def cfg(**kw):
    d = dict(DEFAULTS); d.update(kw); return d
