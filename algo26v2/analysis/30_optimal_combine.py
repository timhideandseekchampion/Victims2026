"""Combine ALL somewhat-significant signals optimally.

Method: each edge is a market-neutral sub-book. Backtest each standalone to get
its daily-PnL series, then find the max-Sharpe (tangency) weights
w ~ Sigma^-1 mu (w>=0). Because the edges are weakly correlated, the blended
Sharpe exceeds any single edge; and since score ~ mean*SR^2/(SR^2+1) scales
linearly with size at fixed Sharpe, we then scale the blended book to saturate
the $ position limits. Backtested with exact eval.py logic.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from strat_engine import Engine, cfg, nInst
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv").sort_values("coint_p")
def pb(p): s = cd[cd.coint_p < p]; return [(IDX[a], IDX[b]) for a, b in zip(s.a, s.b)]

# each edge = an Engine with a single active component (base sizing)
EDGE_CFGS = {
    "pairs":   cfg(w_pairs=1, fixed_pairs=pb(0.02), pair_dollars=10000, pair_entry=1.25, pair_exit=0.3),
    "algo":    cfg(w_algo=1, algo_dollars=100000, algo_h=5),
    "corr":    cfg(w_corr=1, corr_dollars=6000, corr_entry=1.0),
    "lead":    cfg(w_lead=1, lead_dollars=4000),
    "xs":      cfg(w_xs=1, xs_dollars=5000, xs_h=10),
    "mf":      cfg(w_mf=1, mf_dollars=5000, mf_k=3),
}
EDGES = list(EDGE_CFGS)


def edge_positions(name, hist):
    return Engine(EDGE_CFGS[name]).position(hist)  # fresh engine ok except pairs state


def run_books(start, end):
    """Return dict name -> daily PnL series (clipped, net comm), and cache positions."""
    engs = {n: Engine(EDGE_CFGS[n]) for n in EDGES}
    books = {n: [] for n in EDGES}          # per-day target share vectors
    for t in range(start, end):
        hist = P[:, :t]
        for n in EDGES:
            books[n].append(engs[n].position(hist))
    return books


def pnl_of(books_seq, weights, start, end):
    """Backtest the weighted-sum book through eval logic; return pnl series."""
    cash=0.0; cp=np.zeros(N); val=0.0; cm=0.0; pll=[]
    for k, t in enumerate(range(start, end+1)):
        cur = P[:, t-1]
        if t < end:
            raw = np.zeros(N)
            for n in EDGES: raw += weights[n]*books_seq[n][k]
            lim=(dlrPosLimit/cur).astype(int); np_=np.clip(raw,-lim,lim).astype(int)
        else: np_=np.array(cp)
        d=np_-cp; cash-=cur.dot(d)+cm; dv=cur*np.abs(d); cm=np.sum(dv*commRate)
        cp=np.array(np_); pv=cp.dot(cur); pll.append(cash+pv-val) if t>start else None; val=cash+pv
    return np.array(pll)


def stats(pll):
    mu,sd=pll.mean(),pll.std(); sr=np.sqrt(250)*mu/sd if sd>0 else 0
    return sr, (mu*(sr**2/(sr**2+1)) if mu>0 and sd>1e-10 else mu), mu

FULL=(T-250,T)
section("30A. STANDALONE EDGE PnL (base sizing) + tangency weights")
books = run_books(*FULL)
# standalone pnl per edge (unit weight, others 0)
pnls = {}
for n in EDGES:
    w = {m: (1.0 if m==n else 0.0) for m in EDGES}
    pnls[n] = pnl_of(books, w, *FULL)
    sr,sc,mu = stats(pnls[n])
    print(f"  {n:<7}: Sharpe {sr:5.2f}  mean ${mu:7.2f}  score {sc:7.1f}")

M = np.array([pnls[n] for n in EDGES])
mu = M.mean(1); Sig = np.cov(M)
w_tan = np.linalg.solve(Sig + 1e-6*np.eye(len(EDGES)), mu)
w_tan = np.clip(w_tan, 0, None)
w_tan = w_tan/ w_tan.max()   # normalise so largest weight = 1
wt = {n: float(w_tan[i]) for i,n in enumerate(EDGES)}
print("\nTangency (max-Sharpe) weights (relative):")
for n in EDGES: print(f"  {n:<7}: {wt[n]:.3f}")

section("30B. BLENDED BOOK, SCALED TO LIMITS (score vs scale multiplier)")
print(f"{'scale':>7}{'Sharpe':>8}{'Score':>8}{'mean$':>8}")
best=(None,-1)
for scale in (1,2,4,7,10,15,20):
    w = {n: wt[n]*scale for n in EDGES}
    sr,sc,mu_=stats(pnl_of(books, w, *FULL))
    if sc>best[1]: best=(scale,sc,sr,mu_)
    print(f"{scale:>7}{sr:>8.2f}{sc:>8.0f}{mu_:>8.0f}")
print(f"\nBest scale {best[0]}: Sharpe {best[2]:.2f} score {best[1]:.0f} mean ${best[3]:.0f}")

section("30C. REFINE — grid around tangency (drop weak edges, boost pairs/algo)")
variants = {
  "tangency*best":    {n: wt[n]*best[0] for n in EDGES},
  "pairs+algo heavy":  dict(pairs=8, algo=8, corr=2, lead=2, xs=2, mf=0),
  "all equal x8":      dict(pairs=8, algo=8, corr=8, lead=8, xs=8, mf=4),
  "pairs+algo+lead":   dict(pairs=10, algo=8, corr=0, lead=4, xs=0, mf=0),
  "pairs+algo+corr+xs":dict(pairs=10, algo=8, corr=3, lead=2, xs=3, mf=0),
}
print(f"{'variant':<22}{'Sharpe':>8}{'Score':>8}{'mean$':>8}{'H1':>7}{'H2':>7}")
booksH1 = run_books(T-250, T-125); booksH2 = run_books(T-125, T)
bestv=(None,-1,None)
for name,w in variants.items():
    sr,sc,mu_=stats(pnl_of(books, w, *FULL))
    _,s1,_=stats(pnl_of(booksH1, w, T-250, T-125))
    _,s2,_=stats(pnl_of(booksH2, w, T-125, T))
    if sc>bestv[1]: bestv=(name,sc,w)
    print(f"{name:<22}{sr:>8.2f}{sc:>8.0f}{mu_:>8.0f}{s1:>7.0f}{s2:>7.0f}")
print(f"\nBEST: {bestv[0]} -> full score {bestv[1]:.0f}")
import json; json.dump({"weights":bestv[2],"edge_cfgs":{n:EDGE_CFGS[n] for n in EDGES}},
                       open(f"{RESULTS}/best_combo.json","w"), default=str)
