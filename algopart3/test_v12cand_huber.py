"""
test_v12cand_huber.py

CANDIDATE: Huber-robustified ridge (via IRLS -- iteratively reweighted least squares) instead of
pure L2 loss, to reduce sensitivity of the fit to extreme training days.

MECHANISM: the shipped ridge minimizes squared residuals, which lets a handful of extreme-return
days (across the whole 50-name cross-section at once, since one shared EW time-weight multiplies
every response column) disproportionately drag the fit. Huber loss is quadratic for small residuals
(behaves like ridge there) but linear beyond a threshold delta, capping the influence of outlier
training days. Genuinely different from every other mechanism tested this session: RRR (response-side
rank), predictor-wise shrinkage (regularization structure), beta-demean (response transform, shipped
as v9) -- this changes the LOSS FUNCTION'S sensitivity to specific training observations.

SIMPLIFICATION, stated honestly: a fully faithful per-response Huber fit would need a separate
reweighting per one of the 50 target columns (each column's own residual could be an outlier on a
different day), which would break the shared-weight closed-form solve this whole file's ridge relies
on (50 separate p x p solves per half-life per iteration instead of 1). Instead, this computes ONE
combined per-day robustness weight from the AGGREGATE (z-scored, pooled-across-targets) residual
magnitude that day, multiplied into the existing EW time-decay weight -- i.e., a whole TRAINING DAY
is down-weighted if it was extreme for the cross-section as a whole, not per individual stock. This
keeps the single shared-weight ridge solve intact (just a different weight vector), at the cost of
losing per-name precision. Implemented via IRLS: fit once with pure EW weights, compute Huber weights
from that fit's residuals, refit with the product of EW and Huber weights (N_IRLS-1 reweighting
passes after the initial fit).

Tested ON TOP OF the current best (SAFE_llboost_v9) -- reuses its _beta_adjusted_target,
_pairwise_boost, _algo_vol_shares verbatim.

Baseline = SAFE_llboost_v9. Must beat it on OLD, NEW, rolling-mean JOINTLY to pass.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v9 as V9

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V9.WARMUP, V9.BOOST_MIN_DAY, V9.BOOST_K
RIDGE_A = V9.RIDGE_A
HALF_LIVES = V9.HALF_LIVES


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


def _ewls_fit_w(X, Y, w):
    """Same closed-form solve as V9._ewls_ridge, but takes an explicit weight vector w (instead of
    building it from a half-life) -- lets the caller multiply in Huber weights."""
    p = X.shape[1]
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + RIDGE_A) * np.eye(p), XtWY)
    return B, mx, my


def _huber_ridge(X, Y, hl, huber_k, n_irls):
    n = X.shape[0]
    lam = 0.5 ** (1.0 / hl)
    w_ew = lam ** np.arange(n - 1, -1, -1)
    w = w_ew.copy()
    B = mx = my = None
    for it in range(max(1, n_irls)):
        B, mx, my = _ewls_fit_w(X, Y, w)
        if it == n_irls - 1 or huber_k is None:
            break
        Xc = X - mx; E = (Y - my) - Xc @ B
        Ez = E / (E.std(0, keepdims=True) + 1e-12)          # z-score each response column's residual
        d = np.sqrt((Ez ** 2).mean(1))                       # per-day pooled residual magnitude
        med = np.median(d) + 1e-12
        delta = huber_k * med
        huber_w = np.where(d <= delta, 1.0, delta / d)
        w = w_ew * huber_w
    return B, mx, my


print("=== precompute: reversal leg, boost, ALGO leg (unchanged -- reused verbatim from v9) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V9.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V9._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V9._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(huber_k, n_irls):
    POS = np.zeros((nInst, nt))
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Y = V9._beta_adjusted_target(rr_)
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = _huber_ridge(X, Y, hl, huber_k, n_irls)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        wz = (1 - V9.BLEND) * wz + V9.BLEND * REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: huber_k=None (pure EW, n_irls=1) must reproduce SAFE_llboost_v9 exactly ===")
POS_base = build_pos(None, 1)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v9 docstring: 848.8/893.3/894.1/708.6)")
if not (abs(base_wo - 848.8) < 0.5 and abs(base_wn - 893.3) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v9 -- do not trust results below. ***")
else:
    print("  OK -- matches v9 to within rounding.")


def evaluate(nm, huber_k, n_irls, verbose=True):
    Pz = build_pos(huber_k, n_irls); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<24}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, huber_k=huber_k, n_irls=n_irls, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                nworse=nworse, passed=passed, scs=scs)


print("\n=== SWEEP: huber_k (n_irls=2, one reweighting pass) ===")
HUBER_KS = [1.0, 1.5, 2.0, 3.0, 5.0]
results = []
for hk in HUBER_KS:
    results.append(evaluate(f"huber_k={hk} irls=2", hk, 2))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v9 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: huber_k={best['huber_k']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
    print("\n=== checking a 2nd IRLS reweighting pass (irls=3) at the best huber_k ===")
    evaluate(f"huber_k={best['huber_k']} irls=3", best['huber_k'], 3)
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  huber_k={c['huber_k']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
