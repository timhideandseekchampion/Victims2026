"""
test_batch100_D64_hiershrink.py

D64: hierarchical / multilevel ridge -- shrink each name's OWN coefficient vector (one of the 50
columns of B returned by the shared closed-form ridge solve) toward the POPULATION-AVERAGE coefficient
vector (mean across all 50 idio names), rather than only toward zero (which is all the existing ridge
penalty `a` does). This is a simple empirical-Bayes / multilevel-model style shrinkage applied post-hoc
to each half-life's fitted B: B_j <- (1-LAMBDA_H)*B_j + LAMBDA_H*mean_j(B_j).

Distinct from the already-tried "per-half-life differential shrinkage" (test_v11cand_predictor_shrink,
which shrinks the ridge_a strength differently ACROSS half-lives) -- this shrinks ACROSS NAMES within
one half-life's fit, toward a common population coefficient vector, at a fixed half-life.

Everything else (BLEND reversal, pairwise boost, rank-stability blend, ALGO leg) is reused verbatim
from SAFE_llboost_v10 via batch100_d6x_shared.py (shared precompute across the whole D61-D64 batch).
"""
import numpy as np, time
import SAFE_llboost_v10 as V10
import batch100_d6x_shared as SH

r, days, nIdio, nt = SH.r, SH.days, SH.nIdio, SH.nt
HALF_LIVES, RIDGE_A = SH.HALF_LIVES, SH.RIDGE_A

print(f"\nSANITY_CHECK_PASSED (shared baseline) = {SH.SANITY_OK}")


def build_wz_ridge(lam_h):
    """lam_h=0.0 must reproduce v10 exactly (no shrinkage toward pop mean). lam_h>0: each half-life's
    B column (per idio name) is pulled toward the mean coefficient vector across all 50 names."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            if lam_h > 0:
                Bbar = B.mean(axis=1, keepdims=True)
                B = (1 - lam_h) * B + lam_h * Bbar
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ


print("\n=== sanity check: LAMBDA_H=0 (mechanism OFF), re-derived via this script's own build_wz_ridge, "
      "must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ0 = build_wz_ridge(0.0)
c0 = SH.evaluate("LAMBDA_H=0 (=v10)", WZ0)
print(f"  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(c0["wo"] - 871.0) < 0.5 and abs(c0["wn"] - 912.6) < 0.5 and SH.SANITY_OK
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


print("\n=== SWEEP: LAMBDA_H (population-mean shrinkage weight on each name's coefficient vector) ===")
results = [c0]
for lam_h in (0.1, 0.2, 0.3, 0.5):
    t0 = time.time()
    WZ_H = build_wz_ridge(lam_h)
    c = SH.evaluate(f"LAMBDA_H={lam_h}", WZ_H, SH.base_wo, SH.base_wn, SH.base_scs)
    results.append(c)
    print(f"  [{time.time()-t0:.0f}s]")

cand_results = results[1:]
passing = [c for c in cand_results if c["passed"]]
print(f"\n{len(passing)}/{len(cand_results)} LAMBDA_H configs beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(cand_results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
