"""
test_batch100_D61_resid_ar1.py

D61: two-stage ridge. Stage 1 is the shipped ridge (V10._ewls_ridge), unchanged. Stage 2 models the
AUTOCORRELATION of stage-1's own in-sample fit residuals (a Cochrane-Orcutt-style AR(1) residual
correction) and adds a correction term to the out-of-sample forecast:
    pred_corrected = pred + PHI_SCALE * PHI * E_last
where E_last is the most recent in-sample residual (per name) and PHI is a pooled (across names and
time, for tractability -- same "aggregate, stated honestly" simplification test_v12cand_huber.py used
for its per-day robustness weight) lag-1 autocorrelation of the stage-1 residuals, computed fresh each
call from that half-life's in-sample fit.

PHI_SCALE=0 reproduces v10 exactly (pure stage-1, sanity check). PHI_SCALE=1 applies the full AR(1)
correction; PHI_SCALE=0.5 a half-strength (shrunk) version, in case the pooled/aggregate PHI is too
noisy applied at full strength.

Everything else (BLEND reversal, pairwise boost, rank-stability blend, ALGO leg) is reused verbatim
from SAFE_llboost_v10 via batch100_d6x_shared.py (shared precompute across the whole D61-D64 batch).
"""
import numpy as np, time
import SAFE_llboost_v10 as V10
import batch100_d6x_shared as SH

r, days, nIdio, nt = SH.r, SH.days, SH.nIdio, SH.nt
HALF_LIVES, RIDGE_A = SH.HALF_LIVES, SH.RIDGE_A

print(f"\nSANITY_CHECK_PASSED (shared baseline) = {SH.SANITY_OK}")


def build_wz_ridge(phi_scale):
    WZ = np.full((nIdio, nt), np.nan)
    phi_track = []
    for t in days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            if phi_scale != 0.0:
                Xc = X - mx; Yc = Y - my
                E = Yc - Xc @ B
                if E.shape[0] >= 30:
                    e0 = E[:-1, :].ravel(); e1 = E[1:, :].ravel()
                    s0, s1 = e0.std(), e1.std()
                    if s0 > 1e-12 and s1 > 1e-12:
                        phi = float(np.corrcoef(e0, e1)[0, 1])
                        pred = pred + phi_scale * phi * E[-1, :]
                        phi_track.append(phi)
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ, phi_track


print("\n=== sanity check: PHI_SCALE=0 (mechanism OFF), re-derived via this script's own build_wz_ridge, "
      "must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ0, _ = build_wz_ridge(0.0)
c0 = SH.evaluate("PHI_SCALE=0 (=v10)", WZ0)
print(f"  (shared-cache baseline was OLD={SH.base_wo:.1f} NEW={SH.base_wn:.1f} rmean={SH.base_scs.mean():.1f}) "
      f"[{time.time()-t0:.0f}s]")
SANITY_OK = abs(c0["wo"] - 871.0) < 0.5 and abs(c0["wn"] - 912.6) < 0.5 and SH.SANITY_OK
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")

print("\n=== DIAGNOSTIC: pooled lag-1 autocorrelation of stage-1 ridge residuals (hl=500, sampled every "
      "60 days) ===")
sample_phis = []
for t in days[::60]:
    rr_ = r[:, :t]
    Y = V10._beta_adjusted_target(rr_)
    X = rr_[:, :-1].T
    B, mx, my = V10._ewls_ridge(X, Y, 500, RIDGE_A)
    Xc = X - mx; Yc = Y - my
    E = Yc - Xc @ B
    if E.shape[0] >= 30:
        e0 = E[:-1, :].ravel(); e1 = E[1:, :].ravel()
        if e0.std() > 1e-12 and e1.std() > 1e-12:
            sample_phis.append(float(np.corrcoef(e0, e1)[0, 1]))
sample_phis = np.array(sample_phis)
print(f"  mean={sample_phis.mean():.4f}  std={sample_phis.std():.4f}  min={sample_phis.min():.4f}  "
      f"max={sample_phis.max():.4f}  n={len(sample_phis)}")

print("\n=== CANDIDATE: PHI_SCALE in {0.5, 1.0} (pooled AR(1) residual correction) ===")
results = [c0]
for phi_scale in (0.5, 1.0):
    t0 = time.time()
    WZ_C, _ = build_wz_ridge(phi_scale)
    c = SH.evaluate(f"PHI_SCALE={phi_scale}", WZ_C, SH.base_wo, SH.base_wn, SH.base_scs)
    results.append(c)
    print(f"  [{time.time()-t0:.0f}s]")

cand_results = results[1:]
passing = [c for c in cand_results if c["passed"]]
print(f"\n{len(passing)}/{len(cand_results)} PHI_SCALE configs beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(cand_results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
