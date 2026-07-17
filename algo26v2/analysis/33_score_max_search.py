"""Score-maximizing random search over ALL six signals (weak ones included).

Objective is SCORE = mean*SR^2/(SR^2+1), not Sharpe. Weak signals that add
mean-PnL on idle capacity can raise score even while lowering Sharpe. We
randomly sample weights + dollar sizes for all components and keep the top
scorers, reporting recency split (first/last 125d) too.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from strat_engine import Engine, cfg
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section, RESULTS)

rng = np.random.RandomState(7)
P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
cr = np.full(N, COMM_DEFAULT); cr[0] = COMM_INST0
lim0 = np.full(N, POSLIM_DEFAULT); lim0[0] = POSLIM_INST0
cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv").sort_values("coint_p")
def pb(p): s = cd[cd.coint_p < p]; return [(IDX[a], IDX[b]) for a, b in zip(s.a, s.b)]
PSETS = {0.005: pb(0.005), 0.01: pb(0.01), 0.02: pb(0.02)}


def bt(c, start, end):
    eng = Engine(c); cash=0.0; cp=np.zeros(N); val=0.0; cm=0.0; pll=[]
    for t in range(start, end+1):
        h = P[:, :t]; cur = h[:, -1]
        if t < end:
            lim=(lim0/cur).astype(int); pos=np.clip(eng.position(h), -lim, lim).astype(int)
        else: pos=np.array(cp)
        d=pos-cp; cash-=cur.dot(d)+cm; dv=cur*np.abs(d); cm=np.sum(dv*cr)
        cp=np.array(pos); pv=cp.dot(cur); pll.append(cash+pv-val) if t>start else None; val=cash+pv
    pll=np.array(pll); mu,sd=pll.mean(),pll.std(); sr=np.sqrt(250)*mu/sd if sd>0 else 0
    return sr, (mu*(sr**2/(sr**2+1)) if mu>0 and sd>1e-10 else mu), mu


def sample():
    thr = rng.choice([0.005, 0.01, 0.02])
    return cfg(
        fixed_pairs=PSETS[thr],
        pair_dollars=int(rng.choice([10000, 25000, 40000, 60000])),
        pair_entry=float(rng.choice([0.75, 1.0, 1.25])),
        pair_exit=float(rng.choice([0.3, 0.5])),
        algo_dollars=100000, algo_h=int(rng.choice([3, 5, 7])),
        corr_dollars=int(rng.choice([4000, 6000, 9000])), corr_entry=float(rng.choice([0.8, 1.0, 1.2])),
        lead_dollars=int(rng.choice([3000, 5000, 8000])),
        xs_dollars=int(rng.choice([3000, 6000, 9000])), xs_h=int(rng.choice([5, 10, 20])),
        mf_dollars=int(rng.choice([3000, 6000])), mf_k=int(rng.choice([3, 5])),
        w_pairs=float(rng.choice([0, 1, 2, 4, 8])) or 1.0,
        w_algo=float(rng.choice([0, 1, 2, 4, 8])),
        w_corr=float(rng.choice([0, 1, 2, 3])),
        w_lead=float(rng.choice([0, 1, 2, 4])),
        w_xs=float(rng.choice([0, 1, 2, 3])),
        w_mf=float(rng.choice([0, 1, 2])),
    )


FULL = (T-250, T)
section("33. SCORE-MAX RANDOM SEARCH (live) — mixing weak signals for score")
results = []
NTRY = 120
best_live = -1
print(f"{'it':>4}{'Score':>7}{'Sharpe':>7}{'mean$':>7}  running-best", flush=True)
for it in range(NTRY):
    c = sample()
    try:
        sr, sc, mu = bt(c, *FULL)
        results.append((sc, sr, mu, c))
        if sc > best_live:
            best_live = sc
            print(f"{it:>4}{sc:>7.0f}{sr:>7.2f}{mu:>7.0f}  <-- new best", flush=True)
    except Exception:
        pass
results.sort(key=lambda x: -x[0])
print(f"Evaluated {len(results)} configs. Top 12 by score:\n")
print(f"{'rank':>4}{'Score':>7}{'Sharpe':>7}{'mean$':>7}  {'weights (P/A/C/L/X/M)':<26}{'pairs/$':>12}{'H1':>6}{'H2':>6}")
for r, (sc, sr, mu, c) in enumerate(results[:12], 1):
    _, s1, _ = bt(c, T-250, T-125); _, s2, _ = bt(c, T-125, T)
    wts = f"{c['w_pairs']:.0f}/{c['w_algo']:.0f}/{c['w_corr']:.0f}/{c['w_lead']:.0f}/{c['w_xs']:.0f}/{c['w_mf']:.0f}"
    npr = len(c['fixed_pairs'])
    print(f"{r:>4}{sc:>7.0f}{sr:>7.2f}{mu:>7.0f}  {wts:<26}{npr:>4}/${c['pair_dollars']//1000}k{s1:>7.0f}{s2:>6.0f}")

best = results[0][3]
import json; json.dump({k: v for k, v in best.items() if k != 'fixed_pairs'} | {'n_pairs': len(best['fixed_pairs'])},
                       open(f"{RESULTS}/best_scoremax.json", "w"), default=str)
print(f"\nBest score {results[0][0]:.0f} (Sharpe {results[0][1]:.2f}, mean ${results[0][2]:.0f})")
