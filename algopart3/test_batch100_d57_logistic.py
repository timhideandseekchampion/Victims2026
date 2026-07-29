"""
test_batch100_d57_logistic.py

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
    4-half-life ensemble (sklearn's LogisticRegression does support sample_weight, so the SAME
    exponential-decay weight formula V10._ewls_ridge uses is reused here, just as sample_weight instead
    of a closed-form weighted normal-equations solve).
  - Periodic refit (every REFIT_FREQ days) instead of a daily refit, holding coefficients fixed
    between refits (same convention test_q20_item03_quantreg.py used for its per-name QuantileRegressor
    fits).
  - Binary target: sign of V10._beta_adjusted_target (same target the ridge regresses on, just
    binarized), reused verbatim.

Everything else (BLEND reversal, pairwise boost, rank-stability blend, ALGO leg) is reused verbatim
from SAFE_llboost_v10.
"""
import warnings
import numpy as np, pandas as pd, time
from sklearn.linear_model import LogisticRegression
import SAFE_llboost_v10 as V10

warnings.filterwarnings("ignore")

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
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W


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

print("=== precompute: REV blend leg, pairwise boost, rank-stability signal, ALGO leg -- all "
      "UNCHANGED / reused verbatim from V10 (independent of the ridge-vs-classifier mechanism) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

RS_SIG = np.full((nIdio, nt), np.nan)
for t in days:
    if t < max(RS_SHORT_W, RS_LONG_W) + 5:
        continue
    short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
    long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        continue
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    RS_SIG[:, t] = np.where(disagree, -sz, 0.0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def combine_wz(wz_ridge, t):
    wz = (1 - V10.BLEND) * wz_ridge + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_pos(WZ_RIDGE):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = combine_wz(WZ_RIDGE[:, t], t)
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, WZ_RIDGE, base_wo=None, base_wn=None, base_scs=None, verbose=True):
    Pz = build_pos(WZ_RIDGE); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = None
    if base_wo is not None:
        passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = None if base_scs is None else int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ("  <== fail" if passed is False else "")
        extra = f"  n_worse={nworse}/{len(scs)}" if nworse is not None else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}"
              f"{extra}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== sanity check: baseline ridge ensemble (mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = np.full((nIdio, nt), np.nan)
for t in days:
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
POS_base = build_pos(WZ_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
    SANITY_OK = False
else:
    print("  OK -- matches v10 to within rounding.")
    SANITY_OK = True


REFIT_FREQ = 25
LOGIT_HL = 500     # single representative half-life for the EW sample_weight (middle of HALF_LIVES)
C_REG = 1.0


def build_wz_logistic():
    WZ = np.full((nIdio, nt), np.nan)
    coefs = np.zeros((nIdio, nInst)); intercepts = np.zeros(nIdio)
    last_refit = -10_000
    for t in days:
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
c = evaluate("logistic P(up) classifier", WZ_LOGIT, base_wo, base_wn, base_scs)
print(f"  [{time.time()-t0:.0f}s]")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
