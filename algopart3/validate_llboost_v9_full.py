"""Final validation: run the ACTUAL SAFE_llboost_v9.getMyPosition (production module) through the
full standard report -- OLD, NEW, rolling mean/floor over all 61 windows, n_worse vs both the
currently shipped SAFE_llboost baseline AND SAFE_llboost_v8 (the immediate predecessor v9 is meant to
beat). Mirrors validate_llboost_v8_full.py exactly. Not a backtest approximation -- calls
getMyPosition sequentially in increasing-day order, same convention as eval_llboost_v9.py. Unlike v8,
v9's only change (_beta_adjusted_target) is a pure function of the price history with no cross-call
state, so there's no cold-start class of bug to check here -- but the harness convention is kept
identical for consistency with every other validate_*_full.py in this repo.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost as SHIPPED
import SAFE_llboost_v8 as V8
import SAFE_llboost_v9 as CANDIDATE

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

print("building SAFE_llboost_v8 positions (v9's immediate predecessor) ...")
t0 = time.time()
POS_v8 = build_pos(V8, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")

print("building SAFE_llboost_v9 positions (candidate, real getMyPosition) ...")
t0 = time.time()
POS_v9 = build_pos(CANDIDATE, FIRST_DAY)
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
v8_scs, v8_wo, v8_wn = report("SAFE_llboost_v8", POS_v8, base_scs)
v9_scs, v9_wo, v9_wn = report("SAFE_llboost_v9 (NEW)", POS_v9, base_scs)
print()
print("v9 vs v8 directly (v9's actual improvement target):")
nworse_v8 = int((v9_scs < v8_scs).sum())
passed = (v9_wo > v8_wo) and (v9_wn > v8_wn) and (v9_scs.mean() > v8_scs.mean())
print(f"  OLD:    v8={v8_wo:.1f}   v9={v9_wo:.1f}   delta={v9_wo-v8_wo:+.1f}")
print(f"  NEW:    v8={v8_wn:.1f}   v9={v9_wn:.1f}   delta={v9_wn-v8_wn:+.1f}")
print(f"  rmean:  v8={v8_scs.mean():.1f}   v9={v9_scs.mean():.1f}   delta={v9_scs.mean()-v8_scs.mean():+.1f}")
print(f"  rfloor: v8={v8_scs.min():.1f}   v9={v9_scs.min():.1f}   delta={v9_scs.min()-v8_scs.min():+.1f}")
print(f"  n_worse vs v8: {nworse_v8}/{len(v9_scs)}")
print(f"  passes OLD+NEW+rmean jointly vs v8: {passed}")

print("\nsanity vs the backtest-equivalent sweep (test_v10cand_beta_demean.py predicted "
      "OLD=848.8 NEW=893.3 rmean=894.1 rfloor=708.6 n_worse=16/61 vs v8):")
print(f"  real getMyPosition gives: OLD={v9_wo:.1f} NEW={v9_wn:.1f} rmean={v9_scs.mean():.1f} "
      f"rfloor={v9_scs.min():.1f} n_worse={nworse_v8}/61")
