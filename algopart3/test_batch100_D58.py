"""
test_batch100_D58.py

D58: Student-t MLE-style robust regression for the idio ridge, via IRLS -- DISTINCT from the
already-tried Huber IRLS weighting (test_v12cand_huber.py). Huber's weight is quadratic-then-linear
(bounded influence, weight ~ delta/|resid| beyond a threshold). A Student-t (heavy-tailed) error model
instead gives an M-estimator with SMOOTHLY DECAYING weights over the WHOLE range (never flat at 1,
even for small residuals) -- the classic t-distribution IRLS weight from its MLE score equation:
    w_i = (nu + 1) / (nu + z_i^2),   z_i = resid_i / scale
where nu is the assumed degrees of freedom (small nu = heavier tails = more aggressive downweighting
even of moderately-large residuals). This is a genuinely different tail-behavior from Huber's
clipped-linear influence function, not a re-parameterization of it.

SIMPLIFICATION (same one test_v12cand_huber.py used, stated honestly): a fully faithful per-response
t-fit would need 50 separate IRLS reweightings (one per idio target column). Instead this computes ONE
pooled per-TRAINING-DAY robustness weight from the aggregate (z-scored, pooled-across-targets) residual
magnitude that day, multiplied into the existing EW time-decay weight -- keeps the single shared-weight
ridge solve intact. Implemented via IRLS: fit once with pure EW weights, compute t-weights from that
fit's residuals, refit with the product of EW and t-weights (N_IRLS-1 reweighting passes after the
initial fit).

Everything else (BLEND reversion, pairwise boost, rank-stability blend, ALGO leg) is reused VERBATIM
from V10 via batch100_shared's cached precompute (REV, BOOST, rs_blend, algo_pos) -- unaffected by
this idea, which only touches the ridge's LOSS/WEIGHTING.
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


def _ewls_fit_w(X, Y, w):
    p = X.shape[1]
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + RIDGE_A) * np.eye(p), XtWY)
    return B, mx, my


def _studentt_ridge(X, Y, hl, nu, n_irls):
    """IRLS scaffold identical to the Huber IRLS convention; weight formula swapped to the Student-t
    MLE IRLS weight w = (nu+1)/(nu+z^2), z = pooled per-day residual magnitude / scale."""
    n = X.shape[0]
    lam = 0.5 ** (1.0 / hl)
    w_ew = lam ** np.arange(n - 1, -1, -1)
    w = w_ew.copy()
    B = mx = my = None
    for it in range(max(1, n_irls)):
        B, mx, my = _ewls_fit_w(X, Y, w)
        if it == n_irls - 1 or nu is None:
            break
        Xc = X - mx; E = (Y - my) - Xc @ B
        Ez = E / (E.std(0, keepdims=True) + 1e-12)
        z = np.sqrt((Ez ** 2).mean(1))          # per-day pooled residual magnitude (z already unit-ish)
        t_w = (nu + 1.0) / (nu + z ** 2)
        w = w_ew * t_w
    return B, mx, my


def build_wz_ridge(nu, n_irls):
    WZ = np.full((nIdio, nt), np.nan)
    for t in S.days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = _studentt_ridge(X, Y, hl, nu, n_irls)
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


print("=== sanity check: nu=None, n_irls=1 (pure EW, mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = build_wz_ridge(None, 1)
POS_base = build_pos_from_ridge(WZ_BASE)
base_scs = S.scs_curve(POS_base)
base_wo, base_wn = S.wscore(POS_base, *S.OLD), S.wscore(POS_base, *S.NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


print("\n=== SWEEP: Student-t IRLS, nu (degrees of freedom) in {2,3,5,10}, n_irls=2 ===")
results = []
for nu in (2.0, 3.0, 5.0, 10.0):
    t0 = time.time()
    WZ_T = build_wz_ridge(nu, 2)
    c = evaluate(f"nu={nu} irls=2", WZ_T, base_wo, base_wn, base_scs)
    results.append(c)
    print(f"  [{time.time()-t0:.0f}s]")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} nu configs beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"best by rmean: {best['name']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61  "
      f"passed={best['passed']}")

print(f"\nRESULT D58 (best={best['name']}): passed={best['passed']}  OLD={best['wo']:.1f} "
      f"NEW={best['wn']:.1f} rmean={best['rm']:.1f} rfloor={best['rf']:.1f} n_worse={best['nworse']}/61")
print(f"SANITY_CHECK_PASSED={SANITY_OK}")
