"""
test_batch100_d58_studentt.py

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
initial fit) -- exactly the test_v12cand_huber.py IRLS scaffold, with the weight FORMULA swapped.

Tested ON TOP OF the current best (SAFE_llboost_v10) -- reuses its _beta_adjusted_target,
_pairwise_boost, _algo_vol_shares, rank-stability blend verbatim. Must beat v10 on OLD, NEW,
rolling-mean JOINTLY to pass.
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
      "UNCHANGED / reused verbatim from V10 (independent of the ridge-loss mechanism under test) ===",
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
    """IRLS scaffold identical to test_v12cand_huber._huber_ridge; weight formula swapped to the
    Student-t MLE IRLS weight w = (nu+1)/(nu+z^2), z = pooled per-day residual magnitude / scale."""
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
    for t in days:
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


print("\n=== sanity check: nu=None, n_irls=1 (pure EW, mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = build_wz_ridge(None, 1)
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
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: {best['name']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
