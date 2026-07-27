"""Check performance over days 100-400 -- entirely before BOOST_MIN_DAY=500, so any difference
between SAFE_llboost and SAFE_llboost_v2 here comes purely from the ALGO leg's adaptive-vs-fixed
momentum lookback, in a period outside every tuning/validation window used all session (rolling
windows started at end_day=400; OLD/NEW are 500-750/750-1000). A genuinely out-of-sample check.
"""
import numpy as np, pandas as pd, time
import SAFE_llvol, SAFE_llboost, SAFE_llboost_v2

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
print(f"=== days {S}-{E}, entirely before BOOST_MIN_DAY=500 (boost inactive throughout) ===\n")

for name, mod in [("LLVOL (no boost, fixed mom)", SAFE_llvol),
                   ("LLBOOST (boost inactive here, fixed mom)", SAFE_llboost),
                   ("LLBOOST_V2 (boost inactive here, adaptive mom)", SAFE_llboost_v2)]:
    t0 = time.time()
    POS = build_pos(mod, S - 1, E)
    w = window(POS, S, E)
    print(f"{name:<48} mu={w['mu']:>8.1f}  sd={w['sd']:>8.1f}  score={w['score']:>8.1f}  "
          f"({time.time()-t0:.0f}s)")
