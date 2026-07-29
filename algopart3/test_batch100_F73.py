"""
test_batch100_F73.py

F73: Test per-name estimated AR(1) mean-reversion speed as a conviction multiplier on the existing
reversal leg.

MECHANISM: the shipped REV_W=10 cross-sectional reversal leg (BLEND=0.3) applies the SAME weight to
every idio name uniformly. This tests whether scaling each name's reversal contribution by its own
trailing AR(1) coefficient (causal, lag-1 autocorrelation of its daily returns over a trailing window
W) -- i.e. giving more conviction to names whose returns are ACTUALLY mean-reverting (phi<0) and less
to names that are trending/noisy (phi>=0) -- beats the uniform blend. Per-day, conviction_i =
max(0,-phi_i) renormalized to mean 1 across names (so the average blend intensity is unchanged,
only reallocated cross-sectionally): wz = (1-BLEND)*ridge_z + BLEND*REV*conviction.
Sweep trailing window W (one free parameter) over {60, 120, 250}.
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
WZ_RIDGE = CACHE["WZ_RIDGE"]; REV = CACHE["REV"]; BOOST = CACHE["BOOST"]
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

print("\n=== F73: per-name AR(1) mean-reversion speed as conviction multiplier on REV leg ===")


def ar1_conviction(W):
    AR = np.zeros((nIdio, nt))
    for t in days:
        if t < W + 2:
            continue
        seg = rs[:, t - W:t]  # causal: rs[:, :t] is what's available forming day-t's signal
        x = seg[:, :-1]; y = seg[:, 1:]
        xc = x - x.mean(1, keepdims=True); yc = y - y.mean(1, keepdims=True)
        num = (xc * yc).sum(1)
        den = np.sqrt((xc ** 2).sum(1) * (yc ** 2).sum(1)) + 1e-12
        AR[:, t] = num / den
    return AR


def build_wz_ar1(W):
    AR = ar1_conviction(W)
    WZ = np.zeros((nIdio, nt))
    for t in days:
        conv = np.maximum(0.0, -AR[:, t])
        cm = conv.mean()
        conv_norm = conv / cm if cm > 1e-8 else np.ones(nIdio)
        wz = (1 - BLEND) * WZ_RIDGE[:, t] + BLEND * REV[:, t] * conv_norm
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s = RS_RAW[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return WZ


results = []
for W in [60, 120, 250]:
    t0 = time.time()
    WZf = build_wz_ar1(W)
    POS = build_pos_from_wz(WZf)
    scs = scs_curve(POS)
    wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    results.append(dict(W=W, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed))
    tag = "  <== PASS" if passed else ""
    print(f"  W={W:<5}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}   [{time.time()-t0:.0f}s]")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} AR1-window configs beat v10 on OLD+NEW+rmean jointly.")
