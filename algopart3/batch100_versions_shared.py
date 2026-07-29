"""
batch100_versions_shared.py -- shared expensive precompute for I91/I92/I93/I94 (all comparisons across
SAFE_llboost.py (original), _v7, _v8, _v9, _v10). NOT a test_*.py file itself (imported by
test_batch100_I9x.py scripts), matching the house convention of batch100_shared.py /
precompute_batch100.py.

Builds full (nInst, nt) position arrays for each version by DIRECTLY calling that version's own
getMyPosition(prcSoFar) sequentially with growing history -- i.e. an actual walk-forward simulation,
not a reconstruction from shared internals (these 5 modules have genuinely different idio/ALGO logic
across versions, so there is no single shared "expensive part" to factor out the way there is within
the v10-only batches). This is the correct, most-faithful way to get each version's real positions,
and doubles as the sanity check (must reproduce each version's own documented OLD/NEW/rmean numbers).
Cached to .npz since each getMyPosition call recomputes its ridge ensemble from scratch (O(nInst) but
still nontrivial over ~900 calls x 5 versions).

Reference (README.md), OLD=(500,750) NEW=(750,1000):
  SAFE_llboost (original):  774.1 / 828.6 / 811.4
  SAFE_llboost_v7:          830.3 / 888.5 / 876.8
  SAFE_llboost_v8:          847.4 / 888.9 / 886.2
  SAFE_llboost_v9:          848.8 / 893.3 / 894.1
  SAFE_llboost_v10:         871.0 / 912.6 / 909.8
"""
import os, time
import numpy as np, pandas as pd
import SAFE_llboost as ORIG
import SAFE_llboost_v7 as V7
import SAFE_llboost_v8 as V8
import SAFE_llboost_v9 as V9
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250

REF = {
    "orig": (774.1, 828.6, 811.4),
    "v7":   (830.3, 888.5, 876.8),
    "v8":   (847.4, 888.9, 886.2),
    "v9":   (848.8, 893.3, 894.1),
    "v10":  (871.0, 912.6, 909.8),
}
MODULES = {"orig": ORIG, "v7": V7, "v8": V8, "v9": V9, "v10": V10}


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


def daily_pnl(POS, S, E):
    """Same as wscore's inner loop, but returns the raw per-day PnL array instead of collapsing to
    a score -- needed for leave-one-day-out analysis."""
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    return np.array(tot)


def wscore_commmult(POS, S, E, mult):
    """Same as wscore but with commRate scaled by `mult` (for the commission-sensitivity diagnostic)."""
    cr = commRate * mult
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = cr * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

CACHE_PATH = "batch100_versions_cache.npz"
_ok = False
if os.path.exists(CACHE_PATH):
    try:
        _c = np.load(CACHE_PATH)
        if int(_c["nt"]) == nt and int(_c["nInst"]) == nInst:
            POS = {k: _c[f"POS_{k}"] for k in MODULES}
            _ok = True
            print(f"=== batch100_versions_shared: loaded cached position arrays from {CACHE_PATH} ===",
                  flush=True)
    except Exception as e:
        print(f"  cache load failed ({e}), recomputing", flush=True)

if not _ok:
    print("=== batch100_versions_shared: direct walk-forward simulation of getMyPosition for "
          "orig/v7/v8/v9/v10 (each version's OWN logic, sequential calls with growing history) ===",
          flush=True)
    POS = {}
    for name, mod in MODULES.items():
        t0 = time.time()
        posarr = np.zeros((nInst, nt))
        for t in range(1, nt):
            prcSoFar = P_[:, :t]
            p = mod.getMyPosition(prcSoFar)
            lim = (dlr / prcSoFar[:, -1]).astype(int)
            posarr[:, t - 1] = np.clip(np.asarray(p, dtype=float), -lim, lim)
        POS[name] = posarr
        print(f"  {name} done ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(CACHE_PATH, nt=nt, nInst=nInst, **{f"POS_{k}": v for k, v in POS.items()})
    print(f"  cached to {CACHE_PATH}", flush=True)

print("\n=== sanity check: each version's direct simulation must reproduce its own documented numbers ===")
SANITY = {}
for name in MODULES:
    wo, wn = wscore(POS[name], *OLD), wscore(POS[name], *NEW)
    rm = scs_curve(POS[name]).mean()
    ref_o, ref_n, ref_m = REF[name]
    ok = abs(wo - ref_o) < 1.0 and abs(wn - ref_n) < 1.0 and abs(rm - ref_m) < 1.0
    SANITY[name] = ok
    print(f"  {name:<6} OLD={wo:7.1f} (ref {ref_o})  NEW={wn:7.1f} (ref {ref_n})  "
          f"rmean={rm:7.1f} (ref {ref_m})  {'OK' if ok else '*** MISMATCH ***'}")
