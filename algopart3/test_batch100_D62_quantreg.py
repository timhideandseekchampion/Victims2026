"""
test_batch100_D62_quantreg.py

D62: quantile regression at a NON-MEDIAN quantile (0.60 and 0.40) as the ridge target, distinct from
the already-rejected median (0.50) quantile regression (test_q20_item03_quantreg.py, which replaced
the shipped ridge's MSE loss with a median-pinball loss and found no improvement). The idea here is
different: at q=0.60 the fitted line tracks the UPPER-middle of the conditional return distribution
(more sensitive to a positive skew / right-tail lean in a name's short-term return distribution) and
at q=0.40 the lower-middle (left-tail lean) -- i.e. deliberately asymmetric, not a "more robust median"
variant.

NEW MODEL CLASS (a per-name QuantileRegressor fit, not the shared ridge closed form) -- per repo policy
a quick single/few-config precheck is appropriate rather than an exhaustive grid. Periodic refit (every
REFIT_FREQ days, trailing TRAIN_W window) for tractability, same convention test_q20_item03_quantreg.py
used, but retargeted at V10's beta-adjusted target and plugged into the full V10 pipeline (BLEND
reversal, pairwise boost, rank-stability blend, ALGO leg -- all reused verbatim via
batch100_d6x_shared.py) instead of the old pre-v9 ridge target.
"""
import numpy as np, time
from sklearn.linear_model import QuantileRegressor
import SAFE_llboost_v10 as V10
import batch100_d6x_shared as SH

r, days, nIdio, nInst, nt = SH.r, SH.days, SH.nIdio, SH.nInst, SH.nt

print(f"\nSANITY_CHECK_PASSED (shared baseline) = {SH.SANITY_OK}")
print("\n=== sanity check for THIS script: shared batch100_d6x_shared.WZ_BASE already verified to "
      "reproduce SAFE_llboost_v10 (see module import line above) -- reused directly, not rebuilt ===")

REFIT_FREQ = 20
TRAIN_W = 500
ALPHA_L1 = 0.001


def build_wz_quantreg(quantile):
    """Periodic refit (every REFIT_FREQ days), trailing TRAIN_W window, single QuantileRegressor per
    idio name (no half-life ensemble -- new model class, single-config precheck per repo policy)."""
    WZ = np.full((nIdio, nt), np.nan)
    coefs = None; intercepts = None; last_refit = -10_000
    for t in days:
        if t - last_refit >= REFIT_FREQ or coefs is None:
            rr_ = r[:, :t]
            Y = V10._beta_adjusted_target(rr_)          # (n_train, nIdio)
            a = max(0, Y.shape[0] - TRAIN_W)
            Xtr = rr_[:, :-1].T[a:]                       # align with Y's rows
            Ytr = Y[a:]
            coefs = np.zeros((nIdio, nInst)); intercepts = np.zeros(nIdio)
            for j in range(nIdio):
                y = Ytr[:, j]
                if Xtr.shape[0] < 30 or y.std() < 1e-12:
                    continue
                qr = QuantileRegressor(quantile=quantile, alpha=ALPHA_L1, solver='highs')
                qr.fit(Xtr, y)
                coefs[j] = qr.coef_; intercepts[j] = qr.intercept_
            last_refit = t
        xq = r[:, t - 1]
        pred = coefs @ xq + intercepts
        fi = pred - pred.mean()
        WZ[:, t] = fi / (fi.std() + 1e-12)
    return WZ


print("\n=== CANDIDATE: asymmetric quantile regression, quantile in {0.40, 0.60} ===")
results = []
for q in (0.40, 0.60):
    t0 = time.time()
    WZ_Q = build_wz_quantreg(q)
    c = SH.evaluate(f"quantreg q={q}", WZ_Q, SH.base_wo, SH.base_wn, SH.base_scs)
    results.append(c)
    print(f"  [{time.time()-t0:.0f}s]")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} quantile configs beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SH.SANITY_OK}")
