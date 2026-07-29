"""
test_batch100_G79.py

G79: Test reordering the pipeline: apply the rank-stability blend BEFORE the pairwise boost instead
of after (order-dependency check) against v10.

Shipped v10 order: wz = (1-BLEND)*ridge + BLEND*REV  ->  wz += BOOST_K*boost  ->  blend rank-stability
into wz (RS_WEIGHT formula uses np.abs(wz).mean() AFTER the boost has already been added).
This tests the reordering: wz = (1-BLEND)*ridge + BLEND*REV  ->  blend rank-stability in (RS_WEIGHT
formula now uses np.abs(wz).mean() BEFORE the boost)  ->  wz += BOOST_K*boost.
Everything else (constants, mechanisms) is identical -- pure order-dependency check, no free
parameter to sweep.
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RS_WEIGHT = V10.RS_WEIGHT
BLEND = V10.BLEND


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
WZ_PRE = CACHE["WZ_PRE"]; BOOST = CACHE["BOOST"]
algo_pos = CACHE["algo_pos"]; RS_RAW = CACHE["RS_RAW"]; WZ_V10 = CACHE["WZ_V10"]
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

print("\n=== G79: reorder pipeline -- rank-stability blend BEFORE pairwise boost ===")


def build_wz_reordered():
    WZ = np.zeros((nIdio, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        s = RS_RAW[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        WZ[:, t] = wz
    return WZ


WZf = build_wz_reordered()
POS = build_pos_from_wz(WZf)
scs = scs_curve(POS)
wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
nworse = int((scs < base_scs).sum())
tag = "  <== PASS" if passed else ""
print(f"  reordered (RS before boost)   OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
      f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")
print(f"  shipped order (boost before RS): OLD={base_wo:.1f}  NEW={base_wn:.1f}  "
      f"rmean={base_scs.mean():.1f}  rfloor={base_scs.min():.1f}  n_worse=0/61")
print(f"\n{'Order matters: reordering ' + ('helps' if passed else 'does NOT help') + '.'}")
