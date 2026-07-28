"""Final validation: run the ACTUAL SAFE_llboost_v8.getMyPosition (production module, module-level
HOLD-state included) through the full standard report -- OLD, NEW, rolling mean/floor over all 61
windows, n_worse vs both the currently shipped SAFE_llboost baseline AND SAFE_llboost_v7 (the
immediate predecessor v8 is meant to beat). Mirrors validate_llboost_v7_full.py exactly. Not a
backtest approximation -- calls getMyPosition sequentially in increasing-day order, same convention
as eval_llboost_v8.py and every prior validate_*_full.py in this repo (v8's HOLD deadband relies on
this exact calling convention for its module-level state, same as _limits' _DLR cache).
"""
import numpy as np, pandas as pd, time
import SAFE_llboost as SHIPPED
import SAFE_llboost_v7 as V7
import SAFE_llboost_v8 as CANDIDATE

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

print("building SAFE_llboost_v7 positions (v8's immediate predecessor) ...")
t0 = time.time()
POS_v7 = build_pos(V7, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s")

print("building SAFE_llboost_v8 positions (candidate, real getMyPosition incl. HOLD state) ...")
t0 = time.time()
POS_v8 = build_pos(CANDIDATE, FIRST_DAY)
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
v7_scs, v7_wo, v7_wn = report("SAFE_llboost_v7", POS_v7, base_scs)
v8_scs, v8_wo, v8_wn = report("SAFE_llboost_v8 (NEW)", POS_v8, base_scs)
print()
print("v8 vs v7 directly (v8's actual improvement target):")
nworse_v7 = int((v8_scs < v7_scs).sum())
passed = (v8_wo > v7_wo) and (v8_wn > v7_wn) and (v8_scs.mean() > v7_scs.mean())
print(f"  OLD:    v7={v7_wo:.1f}   v8={v8_wo:.1f}   delta={v8_wo-v7_wo:+.1f}")
print(f"  NEW:    v7={v7_wn:.1f}   v8={v8_wn:.1f}   delta={v8_wn-v7_wn:+.1f}")
print(f"  rmean:  v7={v7_scs.mean():.1f}   v8={v8_scs.mean():.1f}   delta={v8_scs.mean()-v7_scs.mean():+.1f}")
print(f"  rfloor: v7={v7_scs.min():.1f}   v8={v8_scs.min():.1f}   delta={v8_scs.min()-v7_scs.min():+.1f}")
print(f"  n_worse vs v7: {nworse_v7}/{len(v8_scs)}")
print(f"  passes OLD+NEW+rmean jointly vs v7: {passed}")

print("\nsanity vs the backtest-equivalent sweep (test_v7_algo_deadband_v2.py predicted "
      "OLD=847.4 NEW=888.9 rmean=886.2 rfloor=674.4 n_worse=0/61 vs v7):")
print(f"  real getMyPosition gives: OLD={v8_wo:.1f} NEW={v8_wn:.1f} rmean={v8_scs.mean():.1f} "
      f"rfloor={v8_scs.min():.1f} n_worse={nworse_v7}/61")
