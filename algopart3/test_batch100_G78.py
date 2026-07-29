"""
test_batch100_G78.py

G78: Test a LEARNED (not equal-weight) blend of the 4 ridge half-life forecasts, weights fit via each
half-life's own trailing realized hit-rate.

MECHANISM: the shipped ensemble is wz_ridge = mean(fs_hl for hl in HALF_LIVES) -- equal weight. This
tests replacing the average with weights proportional to each half-life's own trailing pooled (across
all 50 idio names) sign-hit-rate over a causal trailing window: weight_hl(t) ~ max(hitrate_hl(t)-0.5,
0), renormalized to sum to 1 (falling back to equal weight if data is insufficient or all half-lives
show <=50% trailing hit-rate that day). Sweep the trailing window W (one free parameter) over
{60, 120, 250}. Everything downstream (BLEND w/ REV, boost, rank-stability) is unchanged, exactly as
shipped, applied to the reweighted ridge output.
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
HALF_LIVES = V10.HALF_LIVES
n_hl = len(HALF_LIVES)


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
FS = CACHE["FS"]; REV = CACHE["REV"]; BOOST = CACHE["BOOST"]
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

print("\n=== G78: learned blend of the 4 ridge half-life forecasts, weights = trailing hit-rate ===")

# pooled (cross all 50 idio names) sign-hit-rate of each half-life's raw forecast on each day
CORR = np.full((n_hl, nt), np.nan)
for hi in range(n_hl):
    for t in days:
        if t >= rs.shape[1]:
            continue
        pred = FS[hi, :, t]; act = rs[:, t]
        valid = ~np.isnan(pred) & (pred != 0) & (act != 0)
        if valid.sum() == 0:
            continue
        CORR[hi, t] = np.mean(np.sign(pred[valid]) == np.sign(act[valid]))


def trailing_hitrate(corr_row, W):
    HR = np.full(nt, np.nan)
    valid = ~np.isnan(corr_row)
    c = np.where(valid, corr_row, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(c)))
    cnt = np.concatenate(([0], np.cumsum(valid.astype(int))))
    for t in days:
        lo = max(0, t - W)
        n = cnt[t] - cnt[lo]
        if n < 10:
            continue
        HR[t] = (csum[t] - csum[lo]) / n
    return HR


def build_wz_learned(W):
    HRs = np.array([trailing_hitrate(CORR[hi], W) for hi in range(n_hl)])
    WZr = np.zeros((nIdio, nt))
    for t in days:
        w = HRs[:, t]
        if np.isnan(w).any():
            wts = np.full(n_hl, 1.0 / n_hl)
        else:
            adj = np.maximum(w - 0.5, 0.0)
            wts = adj / adj.sum() if adj.sum() > 1e-8 else np.full(n_hl, 1.0 / n_hl)
        WZr[:, t] = sum(wts[hi] * FS[hi, :, t] for hi in range(n_hl))
    return WZr


def build_wz_full_from_ridge(WZr):
    WZ = np.zeros((nIdio, nt))
    for t in days:
        wz = (1 - BLEND) * WZr[:, t] + BLEND * REV[:, t]
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
    WZr = build_wz_learned(W)
    WZf = build_wz_full_from_ridge(WZr)
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
print(f"\n{len(passing)}/{len(results)} trailing-window configs beat v10 on OLD+NEW+rmean jointly.")
