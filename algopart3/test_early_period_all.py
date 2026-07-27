"""Test EVERY strategy in the repo over days 100-400 -- entirely before BOOST_MIN_DAY=500 (so the
boost is inactive throughout for LLBOOST/LLBOOST_V2), and outside every tuning/validation window
used this session (rolling windows started at day 400/150; OLD/NEW are 500-750/750-1000).
"""
import numpy as np, pandas as pd, time
import SAFE, SWING, QUAL, SAFE_llalgo, SAFE_lldollar, SAFE_llmatch, SAFE_llmeta
import SAFE_llvol, SAFE_llvol_vo, SAFE_llboost, SAFE_llboost_v2

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


def build_pos(mod, first_day, last_day):
    POS = np.zeros((nInst, nt))
    for k in range(first_day, last_day):
        POS[:, k] = mod.getMyPosition(P[:, :k + 1])
    return POS


S, E = 100, 400
print(f"=== days {S}-{E}, ALL strategies (exact eval-style accounting) ===\n")

STRATS = [("SAFE (base, reversion ALGO)", SAFE),
          ("SWING", SWING),
          ("QUAL", QUAL),
          ("LLALGO (lead-lag, name-count)", SAFE_llalgo),
          ("LLDOLLAR (lead-lag, net-$)", SAFE_lldollar),
          ("LLMATCH (volume-matched)", SAFE_llmatch),
          ("LLMETA (documented overfit dead-end)", SAFE_llmeta),
          ("LLVOL (adaptive vol+mom)", SAFE_llvol),
          ("LLVOL_VO (vol-only)", SAFE_llvol_vo),
          ("LLBOOST (boost inactive here)", SAFE_llboost),
          ("LLBOOST_V2 (boost inactive, adaptive mom)", SAFE_llboost_v2)]

results = []
for name, mod in STRATS:
    t0 = time.time()
    POS = build_pos(mod, S - 1, E)
    w = window(POS, S, E)
    results.append((name, w))
    print(f"{name:<44} mu={w['mu']:>8.1f}  sd={w['sd']:>8.1f}  score={w['score']:>8.1f}  "
          f"({time.time()-t0:.0f}s)")

print(f"\n=== ranked by score (days {S}-{E}) ===")
for name, w in sorted(results, key=lambda x: -x[1]['score']):
    print(f"  {name:<44} score={w['score']:>8.1f}")
