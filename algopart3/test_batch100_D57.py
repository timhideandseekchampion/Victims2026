"""
test_batch100_D57.py

D57: replace the continuous ridge regression + take-sign step with a DIRECT sign-classification model
(logistic regression) predicting P(positive next-day return) per idio name, from the same
cross-sectional same-day-return feature vector the ridge uses. Since the shipped mechanism always
converts its continuous forecast to a traded SIGN anyway (pos = sign(wz) * ...), a classifier that
directly outputs P(up) is a structurally natural alternative -- centered probability (p-0.5) plays the
role of "wz" (positive => long, negative => short), then z-scored cross-sectionally exactly like the
ridge output before being blended with REV/boost/rank-stability.

NEW MODEL CLASS (per repo policy: a quick single-config precheck, not an exhaustive grid). Kept
tractable via:
  - ONE half-life's EW sample-weighting (hl=500, the middle of V10.HALF_LIVES) instead of the full
    4-half-life ensemble (sklearn's LogisticRegression supports sample_weight, so the SAME
    exponential-decay weight formula V10._ewls_ridge uses is reused here as sample_weight instead of a
    closed-form weighted normal-equations solve).
  - Periodic refit (every REFIT_FREQ=25 days) instead of a daily refit, holding coefficients fixed
    between refits.
  - Binary target: sign of V10._beta_adjusted_target (same target the ridge regresses on, just
    binarized), reused verbatim.

Everything else (BLEND reversion, pairwise boost, rank-stability blend, ALGO leg) is reused VERBATIM
from V10 via batch100_shared's cached precompute (REV, BOOST, rs_blend, algo_pos) -- unaffected by
this idea, which only touches the ridge-vs-classifier prediction step.
"""
import warnings
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
import SAFE_llboost_v10 as V10
import batch100_shared as S

warnings.filterwarnings("ignore")

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


def evaluate(nm, WZ_RIDGE, base_wo=None, base_wn=None, base_scs=None):
    Pz = build_pos_from_ridge(WZ_RIDGE); scs = S.scs_curve(Pz)
    wo = S.wscore(Pz, *S.OLD); wn = S.wscore(Pz, *S.NEW)
    passed = None
    if base_wo is not None:
        passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = None if base_scs is None else int((scs < base_scs).sum())
    tag = "  <== PASS" if passed else ("  <== fail" if passed is False else "")
    extra = f"  n_worse={nworse}/{len(scs)}" if nworse is not None else ""
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}"
          f"{extra}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("=== sanity check: baseline ridge ensemble (mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = np.full((nIdio, nt), np.nan)
for t in S.days:
    rr_ = r[:, :t]
    Y = V10._beta_adjusted_target(rr_)
    X = rr_[:, :-1].T
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    WZ_BASE[:, t] = np.mean(fs, 0)
POS_base = build_pos_from_ridge(WZ_BASE)
base_scs = S.scs_curve(POS_base)
base_wo, base_wn = S.wscore(POS_base, *S.OLD), S.wscore(POS_base, *S.NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


REFIT_FREQ = 25
LOGIT_HL = 500     # single representative half-life for the EW sample_weight (middle of HALF_LIVES)
C_REG = 1.0


def build_wz_logistic():
    WZ = np.full((nIdio, nt), np.nan)
    coefs = np.zeros((nIdio, nInst)); intercepts = np.zeros(nIdio)
    last_refit = -10_000
    for t in S.days:
        if t - last_refit >= REFIT_FREQ:
            rr_ = r[:, :t]
            Y = V10._beta_adjusted_target(rr_)          # (n_train, nIdio), same target as ridge
            X = rr_[:, :-1].T                             # (n_train, nInst)
            n = X.shape[0]
            lam = 0.5 ** (1.0 / LOGIT_HL)
            sw = lam ** np.arange(n - 1, -1, -1)
            for j in range(nIdio):
                y = (Y[:, j] > 0).astype(int)
                if n < 30 or len(np.unique(y)) < 2:
                    continue
                clf = LogisticRegression(C=C_REG, max_iter=200, solver='lbfgs')
                clf.fit(X, y, sample_weight=sw)
                coefs[j] = clf.coef_.ravel(); intercepts[j] = clf.intercept_[0]
            last_refit = t
        xq = r[:, t - 1]
        logit = coefs @ xq + intercepts
        p_up = 1.0 / (1.0 + np.exp(-logit))
        centered = p_up - 0.5
        fi = centered - centered.mean()
        WZ[:, t] = fi / (fi.std() + 1e-12)
    return WZ


print("\n=== CANDIDATE: logistic-regression sign classifier (single config, periodic refit) ===")
t0 = time.time()
WZ_LOGIT = build_wz_logistic()
c57 = evaluate("logistic P(up) classifier", WZ_LOGIT, base_wo, base_wn, base_scs)
print(f"  [{time.time()-t0:.0f}s]")

print(f"\nRESULT D57: passed={c57['passed']}  OLD={c57['wo']:.1f} NEW={c57['wn']:.1f} "
      f"rmean={c57['rm']:.1f} rfloor={c57['rf']:.1f} n_worse={c57['nworse']}/61")
print(f"SANITY_CHECK_PASSED={SANITY_OK}")
