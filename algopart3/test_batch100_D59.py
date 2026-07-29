"""
test_batch100_D59.py

D59: vol-normalize the ridge's INPUT features (standardize each day's cross-sectional returns by each
instrument's own trailing realized vol, causal, via V10._roll_std reused verbatim) while keeping the
TARGET (beta-adjusted next-day return) unchanged. Rationale: the shipped ridge feeds raw log-returns
as features into a shared closed-form solve across all names; if per-instrument vol is heteroskedastic
over time, raw-return features give high-vol names disproportionate leverage on some days and none on
others. Dividing each instrument's return by its own trailing realized vol puts every feature on a
common (roughly unit-vol) scale before the ridge sees it.

Everything else (BLEND reversion, pairwise boost, rank-stability blend, ALGO leg) is reused VERBATIM
from V10 via batch100_shared's cached precompute (REV, BOOST, rs_blend, algo_pos) -- unaffected by
this idea, which only touches the ridge's INPUT features.
"""
import time
import numpy as np
import SAFE_llboost_v10 as V10
import batch100_shared as S

nInst, nt, nIdio = S.nInst, S.nt, S.nIdio
r = S.r
RIDGE_A, HALF_LIVES = S.RIDGE_A, S.HALF_LIVES
BOOST_MIN_DAY, BOOST_K = S.BOOST_MIN_DAY, S.BOOST_K


def build_pos_from_ridge(WZ_RIDGE):
    WZ = np.full((nIdio, nt), np.nan)
    for t in S.days:
        wz = (1 - V10.BLEND) * WZ_RIDGE[:, t] + V10.BLEND * S.REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * S.BOOST[:, t]
        wz = S.rs_blend(wz, t)
        WZ[:, t] = wz
    return S.build_pos_from_wz(WZ)


def build_wz_ridge(Xfeat):
    """Xfeat: (nInst, nt-1) feature array used INSTEAD of raw r for both training columns and the
    query row. Target Y always comes from raw r (unchanged) via V10._beta_adjusted_target."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in S.days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = Xfeat[:, :t - 1].T
        xq = Xfeat[:, t - 1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ


def evaluate(nm, WZ_RIDGE, base_wo, base_wn, base_scs):
    Pz = build_pos_from_ridge(WZ_RIDGE); scs = S.scs_curve(Pz)
    wo = S.wscore(Pz, *S.OLD); wn = S.wscore(Pz, *S.NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    tag = "  <== PASS" if passed else ""
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=bool(passed))


print("=== sanity check: raw-return features (mechanism OFF) must reproduce SAFE_llboost_v10 exactly ===")
t0 = time.time()
WZ_BASE = build_wz_ridge(r)
POS_base = build_pos_from_ridge(WZ_BASE)
base_scs = S.scs_curve(POS_base)
base_wo, base_wn = S.wscore(POS_base, *S.OLD), S.wscore(POS_base, *S.NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


print("\n=== SWEEP: vol-normalize input features, VN_W (trailing realized-vol window) in {10,20,40} ===")
results = []
for VN_W in (10, 20, 40):
    VOL = np.full((nInst, nt - 1), np.nan)
    for i in range(nInst):
        VOL[i, VN_W - 1:] = V10._roll_std(r[i], VN_W)
    Rn = r / (VOL + 1e-8)
    bad = ~np.isfinite(Rn)
    Rn[bad] = r[bad]          # fallback to raw return before enough vol history exists
    WZ_VN = build_wz_ridge(Rn)
    c = evaluate(f"volnorm VN_W={VN_W}", WZ_VN, base_wo, base_wn, base_scs)
    results.append(c)

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} VN_W configs beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"best by rmean: {best['name']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61  "
      f"passed={best['passed']}")

print(f"\nRESULT D59 (best={best['name']}): passed={best['passed']}  OLD={best['wo']:.1f} "
      f"NEW={best['wn']:.1f} rmean={best['rm']:.1f} rfloor={best['rf']:.1f} n_worse={best['nworse']}/61")
print(f"SANITY_CHECK_PASSED={SANITY_OK}")
