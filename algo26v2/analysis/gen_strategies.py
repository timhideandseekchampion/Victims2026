"""Generate 8 self-contained getMyPosition strategy files + batch-validate them
through the exact eval.py logic. Fixed-pair mode => numpy only, fast."""
import os, numpy as np, pandas as pd
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv").sort_values("coint_p")
def pb(p): s = cd[cd.coint_p < p]; return [(IDX[a], IDX[b]) for a, b in zip(s.a, s.b)]
P01, P02 = pb(0.01), pb(0.02)

TEMPLATE = '''"""Algothon 2026 strategy: {desc}
Self-contained getMyPosition. numpy only. Mean-reversion book:
{desc_long}
"""
import numpy as np

nInst = 51
ALGO = 0
PAIRS = {pairs}
CFG = {cfg}

_pstate = {{}}


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
'''

BASE = dict(plb=90, pentry=1.25, pexit=0.3, pdollars=10000, adollars=100000,
            clb=90, centry=1.0, cdollars=6000, llb=60, ldollars=4000,
            xh=10, xdollars=5000, mflb=60, mfk=3, mfdollars=5000,
            w_pairs=0, w_algo=0, w_corr=0, w_lead=0, w_xs=0, w_mf=0)

def C(**kw): d = dict(BASE); d.update(kw); return d

STRATS = [
 ("strategy_1_pairs",       "pure cointegration pairs (OU bands)",
  "43 strong pairs, OU hysteresis enter 1.25/exit 0.3, sized to limits.",
  P02, C(w_pairs=5, pdollars=10000)),
 ("strategy_2_pairs_algo",  "pairs + ALGO 5-day reversion",
  "pairs core + ALGO own mean-reversion at the $100k limit.",
  P02, C(w_pairs=5, w_algo=1, pdollars=10000, adollars=100000)),
 ("strategy_3_pairs_algo_lead", "pairs + ALGO + lead-lag  (top scorer)",
  "adds cross-instrument lead-lag; the highest full-sample score.",
  P02, C(w_pairs=10, w_algo=8, w_lead=4, pdollars=10000, adollars=100000, ldollars=4000)),
 ("strategy_4_broad",       "pairs + ALGO + factor-residual + XS reversal",
  "broad reversion book across pairs, factor residual and weekly reversal.",
  P02, C(w_pairs=8, w_algo=6, w_corr=3, w_xs=3, pdollars=10000, adollars=100000, cdollars=6000, xdollars=5000)),
 ("strategy_5_tangency",    "max-Sharpe tangency blend of all edges",
  "pairs+ALGO+corr+lead weighted by the max-Sharpe (tangency) solution.",
  P02, C(w_pairs=7, w_algo=6.7, w_corr=1.0, w_lead=6.4, pdollars=10000, adollars=100000, cdollars=6000, ldollars=4000)),
 ("strategy_6_highSharpe",  "high-Sharpe core (p<0.01 pairs) + ALGO",
  "only the cleanest p<0.01 pairs (highest Sharpe) + ALGO, sized big.",
  P01, C(w_pairs=8, w_algo=1, pdollars=10000, adollars=100000)),
 ("strategy_7_kitchen_sink","all six edges combined",
  "every positive edge: pairs, ALGO, corr, lead, XS, PCA-residual.",
  P02, C(w_pairs=8, w_algo=8, w_corr=3, w_lead=3, w_xs=3, w_mf=1,
         pdollars=10000, adollars=100000, cdollars=6000, ldollars=4000, xdollars=5000, mfdollars=4000)),
 ("strategy_8_aggressive",  "aggressive max-deployment (pairs+ALGO+lead, scaled)",
  "top blend scaled hard to saturate every position limit.",
  P02, C(w_pairs=20, w_algo=15, w_lead=8, pdollars=10000, adollars=100000, ldollars=5000)),
]

outdir = os.path.join(os.path.dirname(__file__), "strategies")
os.makedirs(outdir, exist_ok=True)
for name, desc, dl, pairs, cfg in STRATS:
    src = TEMPLATE.format(desc=desc, desc_long=dl, pairs=repr(pairs), cfg=repr(cfg))
    open(os.path.join(outdir, name + ".py"), "w").write(src)
print(f"generated {len(STRATS)} strategies in {outdir}")

# ---- batch validate through eval logic ----
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
import importlib.util
def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(outdir, name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def score_strat(m, start, end):
    m._pstate.clear()
    cash=0.0; cp=np.zeros(N); tv=0.0; val=0.0; cm=0.0; pll=[]
    for t in range(start, end+1):
        h = P[:, :t]; cur = h[:, -1]
        if t < end:
            lim=(dlrPosLimit/cur).astype(int); pos=np.clip(m.getMyPosition(h), -lim, lim).astype(int)
        else: pos=np.array(cp)
        d=pos-cp; cash-=cur.dot(d)+cm; dv=cur*np.abs(d); cm=np.sum(dv*commRate); tv+=dv.sum()
        cp=np.array(pos); pv=cp.dot(cur); pll.append(cash+pv-val) if t>start else None; val=cash+pv
    pll=np.array(pll); mu,sd=pll.mean(),pll.std(); sr=np.sqrt(250)*mu/sd if sd>0 else 0
    return sr, (mu*(sr**2/(sr**2+1)) if mu>0 and sd>1e-10 else mu), mu

print(f"\n{'strategy':<32}{'Sharpe':>7}{'Score':>7}{'mean$':>7}{'first125':>9}{'last125':>9}")
for name, desc, dl, pairs, cfg in STRATS:
    m = load(name)
    sr, sc, mu = score_strat(m, T-250, T)
    _, s1, _ = score_strat(m, T-250, T-125)
    _, s2, _ = score_strat(m, T-125, T)
    print(f"{name:<32}{sr:>7.2f}{sc:>7.0f}{mu:>7.0f}{s1:>9.0f}{s2:>9.0f}")
