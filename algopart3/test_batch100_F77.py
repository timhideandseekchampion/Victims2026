"""
test_batch100_F77.py

F77: Test a rolling average-pairwise-correlation-level (correlation regime instability) market-wide
overlay signal against v10.

MECHANISM: compute the rolling average pairwise correlation of the 50 idio names' daily returns
(window W_CORR=60, causal), then its own z-scored deviation from its trailing longer-window mean
(W_INST=250) -- "instability" = how unusual today's correlation-regime level is relative to its
recent history. Since position sizing here is sign(wz)*fixed-dollar (a uniform sign-preserving
rescale of wz is a no-op on positions), the only way a market-wide overlay like this can matter is by
scaling the DOLLAR SIZE of the idio book directly (a legitimate, common risk-overlay mechanism) --
so it is applied as an exposure-scaling multiplier on dlr[1:], not as a wz rescale. Direction is not
assumed a priori (does more correlation-regime instability call for MORE or LESS idio exposure?), so
GAIN is swept over both signs: {-0.3, -0.15, -0.05, 0.05, 0.15, 0.3}, with scale clipped to [0.3, 1.7].
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

print("\n=== F77: rolling average-pairwise-correlation regime-instability overlay (exposure scaling) ===")

W_CORR = 60
W_INST = 250


def rolling_avg_corr(W):
    C = np.full(nt, np.nan)
    iu = np.triu_indices(nIdio, k=1)
    for t in days:
        if t < W + 2:
            continue
        seg = rs[:, t - W:t]  # causal window ending at day t-1 (available at day t)
        seg_c = seg - seg.mean(1, keepdims=True)
        std = seg_c.std(1) + 1e-12
        segz = seg_c / std[:, None]
        corrm = (segz @ segz.T) / W
        C[t] = corrm[iu].mean()
    return C


t0 = time.time()
AVCORR = rolling_avg_corr(W_CORR)
print(f"  rolling avg-pairwise-corr computed ({time.time()-t0:.0f}s); level range "
      f"[{np.nanmin(AVCORR):.3f}, {np.nanmax(AVCORR):.3f}], mean={np.nanmean(AVCORR):.3f}")


def instability_z(series, W):
    Z = np.full(nt, np.nan)
    for t in days:
        if t < W + 2 or np.isnan(series[t]):
            continue
        hist = series[max(0, t - W):t]
        hist = hist[~np.isnan(hist)]
        if len(hist) < 30:
            continue
        mu, sd = hist.mean(), hist.std() + 1e-12
        Z[t] = (series[t] - mu) / sd
    return Z


INSTAB = instability_z(AVCORR, W_INST)


def build_pos_corr_overlay(gain):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]
        z = INSTAB[t]
        scale = 1.0 if np.isnan(z) else float(np.clip(1 - gain * np.clip(z, -2, 2), 0.3, 1.7))
        lim = (dlr[1:] / cur[1:]).astype(int)
        target = np.sign(wz) * (dlr[1:] * scale / cur[1:])
        POS[1:, t] = np.clip(target, -lim, lim)
    POS[0, :] = algo_pos
    return POS


results = []
for gain in [-0.3, -0.15, -0.05, 0.05, 0.15, 0.3]:
    POS = build_pos_corr_overlay(gain)
    scs = scs_curve(POS)
    wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    results.append(dict(gain=gain, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed))
    tag = "  <== PASS" if passed else ""
    print(f"  gain={gain:<7}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} gain values beat v10 on OLD+NEW+rmean jointly.")
