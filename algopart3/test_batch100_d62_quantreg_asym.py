"""
test_batch100_d62_quantreg_asym.py

D62: quantile regression at a NON-MEDIAN quantile (0.60 and 0.40) as the ridge target, distinct from
the already-rejected median (0.50) quantile regression (test_q20_item03_quantreg.py, which replaced
the shipped ridge's MSE loss with a median-pinball loss and found no improvement). The idea here is
different: at q=0.60 the fitted line tracks the UPPER-middle of the conditional return distribution
(more sensitive to a positive skew / right-tail lean in a name's short-term return distribution) and
at q=0.40 the lower-middle (left-tail lean) -- i.e. deliberately asymmetric, not a "more robust median"
variant.

NEW MODEL CLASS (a per-name QuantileRegressor fit, not the shared ridge closed form) -- per repo policy
a quick single/few-config precheck is appropriate rather than an exhaustive grid. Periodic refit
(every REFIT_FREQ days, trailing TRAIN_W window) for tractability, same convention
test_q20_item03_quantreg.py used, but retargeted at V10's beta-adjusted target and plugged into the
full V10 pipeline (BLEND reversal, pairwise boost, rank-stability blend, ALGO leg -- all reused
verbatim) instead of the old pre-v9 ridge target.
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import QuantileRegressor
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
      "UNCHANGED / reused verbatim from V10 (independent of the ridge-vs-quantreg mechanism) ===",
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
    c = evaluate(f"quantreg q={q}", WZ_Q, base_wo, base_wn, base_scs)
    results.append(c)
    print(f"  [{time.time()-t0:.0f}s]")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} quantile configs beat v10 on OLD+NEW+rmean jointly.")
if not passing:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
