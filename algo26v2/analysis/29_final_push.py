"""Final push: lock best entry/exit (1.25/0.3), saturate sizing, tune ALGO/corr."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from strat_engine import Engine, cfg
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv").sort_values("coint_p")
def pb(p): s = cd[cd.coint_p < p]; return [(IDX[a], IDX[b]) for a, b in zip(s.a, s.b)]


def bt(c, start, end):
    eng = Engine(c); cash=0.0; cp=np.zeros(N); tv=0.0; val=0.0; cm=0.0; pll=[]
    for t in range(start, end+1):
        h = P[:, :t]; cur = h[:, -1]
        if t < end:
            lim = (dlrPosLimit/cur).astype(int); np_ = np.clip(eng.position(h), -lim, lim).astype(int)
        else: np_ = np.array(cp)
        d = np_-cp; cash -= cur.dot(d)+cm; dv=cur*np.abs(d); cm=np.sum(dv*commRate); tv+=dv.sum()
        cp = np.array(np_); pv=cp.dot(cur); pll.append(cash+pv-val) if t>start else None; val=cash+pv
    pll=np.array(pll); mu,sd=pll.mean(),pll.std(); sr=np.sqrt(250)*mu/sd if sd>0 else 0
    return sr, (mu*(sr**2/(sr**2+1)) if mu>0 and sd>1e-10 else mu), mu

FULL=(T-250,T); H1=(T-250,T-125); H2=(T-125,T)
section("29. FINAL PUSH (entry1.25/exit0.3) — sizing x components (Full / H1 / H2 score)")
print(f"{'config':<32}{'Sharpe':>7}{'Score':>7}{'mean$':>7}{'H1':>6}{'H2':>6}")
E = dict(pair_entry=1.25, pair_exit=0.3)
grid = [
 ("p<.02 43+ $50k",      cfg(w_pairs=1, fixed_pairs=pb(0.02), pair_dollars=50000, **E)),
 ("p<.02 $50k +ALGO",    cfg(w_pairs=1, w_algo=1, fixed_pairs=pb(0.02), pair_dollars=50000, algo_dollars=100000, **E)),
 ("p<.02 $80k +ALGO",    cfg(w_pairs=1, w_algo=1, fixed_pairs=pb(0.02), pair_dollars=80000, algo_dollars=100000, **E)),
 ("p<.01 $80k +ALGO",    cfg(w_pairs=1, w_algo=1, fixed_pairs=pb(0.01), pair_dollars=80000, algo_dollars=100000, **E)),
 ("p<.02 $80k+ALGO+corr6",cfg(w_pairs=1, w_algo=1, w_corr=1, fixed_pairs=pb(0.02), pair_dollars=80000,
                             algo_dollars=100000, corr_dollars=6000, corr_entry=1.2, **E)),
 ("p<.015 $80k +ALGO",   cfg(w_pairs=1, w_algo=1, fixed_pairs=pb(0.015), pair_dollars=80000, algo_dollars=100000, **E)),
 ("p<.02 $120k +ALGO",   cfg(w_pairs=1, w_algo=1, fixed_pairs=pb(0.02), pair_dollars=120000, algo_dollars=100000, **E)),
]
best=(None,-1,None)
for name,c in grid:
    sF,scF,muF=bt(c,*FULL); _,sc1,_=bt(c,*H1); _,sc2,_=bt(c,*H2)
    if scF>best[1]: best=(name,scF,c)
    print(f"{name:<32}{sF:>7.2f}{scF:>7.0f}{muF:>7.0f}{sc1:>6.0f}{sc2:>6.0f}")
print(f"\nBest: {best[0]} -> full score {best[1]:.0f}")
import json; json.dump(best[0], open(f"{RESULTS}/best_hs.json","w"))
