"""Final validation: run the ACTUAL SAFE_llboost_v6.getMyPosition (production module) through the
full standard report -- OLD, NEW, rolling mean/floor over all 61 windows, n_worse vs the currently
shipped SAFE_llboost baseline -- not a backtest approximation. Mirrors validate_llboost_full.py.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost as SHIPPED
import SAFE_llboost_v6 as CANDIDATE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250


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
    return float(score(tot.mean(), tot.std()))


def build_pos(mod, first_day):
    POS = np.zeros((nInst, nt))
    for k in range(first_day, nt):
        POS[:, k] = mod.getMyPosition(P[:, :k + 1])
    return POS


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
FIRST_DAY = 148  # covers every rolling window (earliest need: end_day=400 -> S=150 -> POS index 149)

print("building shipped SAFE_llboost positions (baseline) ...")
t0 = time.time()
POS_base = build_pos(SHIPPED, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")

print("building SAFE_llboost_v6 positions (candidate) ...")
t0 = time.time()
POS_cand = build_pos(CANDIDATE, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E) for E in end_days])
    line = f"{nm:<20}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


base_scs = report("SAFE_llboost (base)", POS_base)
report("SAFE_llboost_v6 (new)", POS_cand, base_scs)

print("\nsanity: NEW here should match eval_llboost_v2.py's official score exactly (868.87)")
