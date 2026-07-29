"""
test_batch100_D60.py

D60: replace the exponentially-weighted (EW) kernel in the idio ridge with a boxcar (flat trailing
window) kernel, for the SAME nominal half-life values (250, 500, 1000, 2000) reinterpreted as flat
window lengths (capped at whatever history is actually available -- causal). Distinct from the
shipped EW decay (weight = 0.5**(k/hl)); this gives every day inside the window equal weight and zero
weight outside it.

Everything else (BLEND reversion, pairwise boost, rank-stability blend, ALGO leg) is reused VERBATIM
from V10 via batch100_shared's cached precompute (REV, BOOST, rs_blend, algo_pos) -- unaffected by
this idea, which only touches the ridge's TIME-WEIGHTING kernel.
"""
import time
import numpy as np
import SAFE_llboost_v10 as V10
import batch100_shared as S

nInst, nt, nIdio = S.nInst, S.nt, S.nIdio
r = S.r
RIDGE_A, HALF_LIVES = S.RIDGE_A, S.HALF_LIVES
BOOST_MIN_DAY, BOOST_K = S.BOOST_MIN_DAY, S.BOOST_K

print("=== MANDATORY sanity check setup: reuses batch100_shared's cached V10 baseline (REV, BOOST, "
      "rs_blend, algo_pos) verbatim -- only the ridge kernel below is new ===")
print(f"  target: OLD=871.0  NEW=912.6  rmean=909.8  rfloor=709.7")


def build_pos_from_ridge(WZ_RIDGE):
    WZ = np.full((nIdio, nt), np.nan)
    for t in S.days:
        wz = (1 - V10.BLEND) * WZ_RIDGE[:, t] + V10.BLEND * S.REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * S.BOOST[:, t]
        wz = S.rs_blend(wz, t)
        WZ[:, t] = wz
    return S.build_pos_from_wz(WZ)


def _ewls_ridge_boxcar(X, Y, W, a):
    """Same closed-form ridge solve as V10._ewls_ridge, but with a FLAT (boxcar) weight over the
    trailing min(W, n) samples instead of an exponential-decay kernel."""
    n, p = X.shape
    w = np.zeros(n)
    w[max(0, n - W):] = 1.0
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


def build_wz_ridge(kernel):
    WZ = np.full((nIdio, nt), np.nan)
    for t in S.days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            if kernel == 'ew':
                B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            else:
                B, mx, my = _ewls_ridge_boxcar(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ


print("\n=== sanity check: EW kernel (mechanism OFF) must reproduce SAFE_llboost_v10 exactly ===")
t0 = time.time()
WZ_BASE = build_wz_ridge('ew')
POS_base = build_pos_from_ridge(WZ_BASE)
base_scs = S.scs_curve(POS_base)
base_wo, base_wn = S.wscore(POS_base, *S.OLD), S.wscore(POS_base, *S.NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


def evaluate(nm, WZ_RIDGE):
    Pz = build_pos_from_ridge(WZ_RIDGE); scs = S.scs_curve(Pz)
    wo = S.wscore(Pz, *S.OLD); wn = S.wscore(Pz, *S.NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    tag = "  <== PASS" if passed else ""
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=bool(passed))


print("\n=== CANDIDATE: boxcar kernel, same nominal window lengths as HALF_LIVES (250,500,1000,2000) ===")
t0 = time.time()
WZ_BOX = build_wz_ridge('boxcar')
c60 = evaluate("boxcar (all 4 windows)", WZ_BOX)
print(f"  [{time.time()-t0:.0f}s]")

print(f"\nRESULT D60: passed={c60['passed']}  OLD={c60['wo']:.1f} NEW={c60['wn']:.1f} "
      f"rmean={c60['rm']:.1f} rfloor={c60['rf']:.1f} n_worse={c60['nworse']}/61")
print(f"SANITY_CHECK_PASSED={SANITY_OK}")
