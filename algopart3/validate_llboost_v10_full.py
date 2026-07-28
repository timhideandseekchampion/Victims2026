"""Final validation: run the ACTUAL SAFE_llboost_v10.getMyPosition (production module) through the
full standard report -- OLD, NEW, rolling mean/floor over all 61 windows, n_worse vs both the
currently shipped SAFE_llboost baseline AND SAFE_llboost_v9 (the immediate predecessor v10 is meant
to beat). Mirrors validate_llboost_v9_full.py exactly. Not a backtest approximation -- calls
getMyPosition sequentially in increasing-day order. v10's only change (_rank_stability_signal) is a
pure function of the price history with no cross-call state, so there's no cold-start class of bug
to check here either.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost as SHIPPED
import SAFE_llboost_v9 as V9
import SAFE_llboost_v10 as CANDIDATE

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
FIRST_DAY = 148

print("building shipped SAFE_llboost positions (baseline) ...")
t0 = time.time()
POS_base = build_pos(SHIPPED, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")

print("building SAFE_llboost_v9 positions (v10's immediate predecessor) ...")
t0 = time.time()
POS_v9 = build_pos(V9, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")

print("building SAFE_llboost_v10 positions (candidate, real getMyPosition) ...")
t0 = time.time()
POS_v10 = build_pos(CANDIDATE, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E) for E in end_days])
    line = f"{nm:<24}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs, wo, wn


base_scs, _, _ = report("SAFE_llboost (base)", POS_base)
v9_scs, v9_wo, v9_wn = report("SAFE_llboost_v9", POS_v9, base_scs)
v10_scs, v10_wo, v10_wn = report("SAFE_llboost_v10 (NEW)", POS_v10, base_scs)
print()
print("v10 vs v9 directly (v10's actual improvement target):")
nworse_v9 = int((v10_scs < v9_scs).sum())
passed = (v10_wo > v9_wo) and (v10_wn > v9_wn) and (v10_scs.mean() > v9_scs.mean())
print(f"  OLD:    v9={v9_wo:.1f}   v10={v10_wo:.1f}   delta={v10_wo-v9_wo:+.1f}")
print(f"  NEW:    v9={v9_wn:.1f}   v10={v10_wn:.1f}   delta={v10_wn-v9_wn:+.1f}")
print(f"  rmean:  v9={v9_scs.mean():.1f}   v10={v10_scs.mean():.1f}   delta={v10_scs.mean()-v9_scs.mean():+.1f}")
print(f"  rfloor: v9={v9_scs.min():.1f}   v10={v10_scs.min():.1f}   delta={v10_scs.min()-v9_scs.min():+.1f}")
print(f"  n_worse vs v9: {nworse_v9}/{len(v10_scs)}")
print(f"  passes OLD+NEW+rmean jointly vs v9: {passed}")

print("\nsanity vs the backtest-equivalent sweep (test_v16cand_rank_stability.py predicted "
      "OLD=871.0 NEW=912.6 rmean=909.8 rfloor=709.7 n_worse=0/61 vs v9):")
print(f"  real getMyPosition gives: OLD={v10_wo:.1f} NEW={v10_wn:.1f} rmean={v10_scs.mean():.1f} "
      f"rfloor={v10_scs.min():.1f} n_worse={nworse_v9}/61")
