"""
test_batch100_G80.py

G80: Test blending TWO simultaneous rank-stability configs (e.g. short8/long15 AND the shipped
short8/long22) together instead of just one.

MECHANISM: compute a SECOND rank-stability raw signal at short8/long15 (using
V10._rank_stability_signal verbatim, via temporary monkeypatch of its module-level RS_SHORT_W/
RS_LONG_W constants -- reusing the exact shipped mechanism rather than reimplementing it) alongside
the shipped raw signal already cached at short8/long22. Average the two raw signals on days both are
available (fall back to whichever is available on early days when long22 needs more warmup than
long15), z-score the combination, and blend it in with the SAME blend-into-wz mechanism the shipped
signal uses. Sweep the blend WEIGHT (one free parameter) over {0.010, 0.015, 0.020, 0.025} -- 0.015
is the shipped single-signal weight.
"""
import numpy as np, pandas as pd, time
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
algo_pos = CACHE["algo_pos"]; RS_RAW_8_22 = CACHE["RS_RAW"]; WZ_V10 = CACHE["WZ_V10"]
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

print("\n=== G80: blend TWO simultaneous rank-stability configs (short8/long15 AND short8/long22) ===")

orig_short, orig_long = V10.RS_SHORT_W, V10.RS_LONG_W
V10.RS_SHORT_W, V10.RS_LONG_W = 8, 15
RS_RAW_8_15 = np.full((nIdio, nt), np.nan)
for t in days:
    sig = V10._rank_stability_signal(logp[:, :t + 1])
    if sig is not None:
        RS_RAW_8_15[:, t] = sig
V10.RS_SHORT_W, V10.RS_LONG_W = orig_short, orig_long  # restore shipped config
print(f"  second config (short8/long15) computed; restored shipped constants "
      f"(RS_SHORT_W={V10.RS_SHORT_W}, RS_LONG_W={V10.RS_LONG_W})")


def build_wz_dualrs(weight):
    WZ = np.zeros((nIdio, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s1 = RS_RAW_8_22[:, t]; s2 = RS_RAW_8_15[:, t]
        avail = [s for s in (s1, s2) if np.isfinite(s).all()]
        if avail:
            s = np.mean(avail, axis=0)
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - weight) * wz + weight * s_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return WZ


results = []
for weight in [0.010, 0.015, 0.020, 0.025]:
    WZf = build_wz_dualrs(weight)
    POS = build_pos_from_wz(WZf)
    scs = scs_curve(POS)
    wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    results.append(dict(weight=weight, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed))
    tag = "  <== PASS" if passed else ""
    print(f"  weight={weight:<7}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} blend weights beat v10 on OLD+NEW+rmean jointly.")
