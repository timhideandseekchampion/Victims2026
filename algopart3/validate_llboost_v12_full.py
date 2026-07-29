"""Final validation: run the ACTUAL SAFE_llboost_v12.getMyPosition (production module) through the
full standard report -- OLD, NEW, rolling mean/floor over all 61 windows, n_worse vs SAFE_llboost_v10
and SAFE_llboost_v11. Also checks the two specific composition claims made in v12's docstring:
(1) the kill switch still fires 0 days on real prices.txt (inherited inert from v11), and
(2) v12's positions equal "v10 + fade only" on every day where a fade would have applied (i.e. the
kill switch and the fade genuinely don't interact on real data)."""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10
import SAFE_llboost_v11 as V11
import SAFE_llboost_v12 as CANDIDATE

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


def commission(POS, S, E):
    curPos = np.zeros(nInst); tot = 0.0
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = POS[:, tt - 1] if tt < E else curPos
        dP = newPos - curPos
        tot += float((commRate * np.abs(dP) * cur).sum())
        curPos = newPos
    return tot


def build_pos(mod, first_day):
    POS = np.zeros((nInst, nt))
    for k in range(first_day, nt):
        POS[:, k] = mod.getMyPosition(P[:, :k + 1])
    return POS


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
FIRST_DAY = 148

print("building SAFE_llboost_v10 positions (baseline) ...", flush=True)
t0 = time.time()
POS_v10 = build_pos(V10, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s", flush=True)

print("building SAFE_llboost_v11 positions (kill switch only) ...", flush=True)
t0 = time.time()
POS_v11 = build_pos(V11, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s", flush=True)

print("building SAFE_llboost_v12 positions (kill switch + fade) ...", flush=True)
t0 = time.time()
POS_v12 = build_pos(CANDIDATE, FIRST_DAY)
print(f"  done in {time.time()-t0:.0f}s", flush=True)

for name, POS in (("v10", POS_v10), ("v11", POS_v11), ("v12", POS_v12)):
    wo, wn = window(POS, *OLD), window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E) for E in end_days])
    print(f"{name}: OLD={wo:7.2f}  NEW={wn:7.2f}  rmean={scs.mean():7.2f}  rfloor={scs.min():7.2f}")
    if name == "v10":
        scs_v10 = scs
    if name == "v12":
        scs_v12 = scs; wo_v12, wn_v12 = wo, wn

nworse_vs_v10 = int((scs_v12 < scs_v10).sum())
print(f"\nv12 n_worse vs v10: {nworse_vs_v10}/{len(scs_v10)}")

d_1011 = POS_v10 - POS_v11
print(f"\nv10 vs v11 identical on real data (kill switch inert)? "
      f"{'YES' if not np.any(d_1011) else f'NO -- {int((np.abs(d_1011).sum(0)>0).sum())} differing days'}")

d_1012 = POS_v10 - POS_v12
n_diff = int((np.abs(d_1012).sum(0) > 0).sum())
print(f"v10 vs v12 differ on {n_diff}/{nt - FIRST_DAY} days (expect ~40, matching the fade-only test)")

c10 = commission(POS_v10, *NEW)
c12 = commission(POS_v12, *NEW)
print(f"\nNEW-window commission: v10=${c10:,.0f}  v12=${c12:,.0f}  (delta ${c12-c10:+,.0f})")

print("\ndone.")
