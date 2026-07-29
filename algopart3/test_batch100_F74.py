"""
test_batch100_F74.py

F74: Read SAFE_combined.py (algopart2/algopart3 sibling file) for its existing cointegration-gated
pairs mechanism (ADF-gated, rolling-OLS hedge ratio) and test adapting/layering a similar signal
ADDITIVELY onto v10 rather than as a standalone book.

MECHANISM: reuse SAFE_combined._pairs_signal(P) VERBATIM (per-name adaptive best partner via rolling
return-correlation, adaptive hedge ratio via rolling-window OLS on levels, ADF-style cointegration gate
-- a pair only contributes while its spread is significantly mean-reverting). Layer this signal
ADDITIVELY onto the FULL v10 wz (after boost, after rank-stability -- v10 "as a book", additive on
top), z-scored and blended with a small weight, analogous to how the rank-stability signal itself is
blended in. Sweep the blend weight (one free parameter) over {0.01, 0.02, 0.05, 0.10}.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10
import SAFE_combined as CMB

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP = V10.WARMUP


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


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

CACHE = np.load("batch100_cache.npz")
algo_pos = CACHE["algo_pos"]; WZ_V10 = CACHE["WZ_V10"]
days = CACHE["days"].tolist()


def build_pos_from_wz(WZfull):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZfull[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("=== sanity check: reproduce SAFE_llboost_v10 exactly from cache ===")
POS_base = build_pos_from_wz(WZ_V10)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} "
      f"rfloor={base_scs.min():.1f}  (v10 docstring: 871.0/912.6/909.8/709.7)")
sanity_ok = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if sanity_ok else "  *** WARNING: mismatch ***")

print("\n=== F74: cointegration-gated structural-pairs signal (SAFE_combined._pairs_signal, verbatim) "
      "layered additively onto v10 ===")

t0 = time.time()
PAIRS_RAW = np.full((nIdio, nt), np.nan)
n_traded_days = 0
for t in days:
    if t < CMB.PAIR_WIN + 2:
        continue
    sig = CMB._pairs_signal(P_[:, :t + 1])
    if sig is not None:
        PAIRS_RAW[:, t] = sig
        n_traded_days += 1
print(f"  pairs signal computed ({time.time()-t0:.0f}s); {n_traded_days}/{len(days)} days had >=1 "
      f"cointegrated pair firing")


def build_wz_pairs(weight):
    WZ = np.zeros((nIdio, nt))
    for t in days:
        wz = WZ_V10[:, t].copy()
        s = PAIRS_RAW[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            if sstd > 1e-12:
                s_z = (s - s.mean()) / sstd
                wz = (1 - weight) * wz + weight * s_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return WZ


results = []
for weight in [0.01, 0.02, 0.05, 0.10]:
    WZf = build_wz_pairs(weight)
    POS = build_pos_from_wz(WZf)
    scs = scs_curve(POS)
    wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    results.append(dict(weight=weight, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed))
    tag = "  <== PASS" if passed else ""
    print(f"  weight={weight:<6}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} blend weights beat v10 on OLD+NEW+rmean jointly.")
