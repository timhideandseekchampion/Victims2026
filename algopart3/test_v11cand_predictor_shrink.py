"""
test_v11cand_predictor_shrink.py

CANDIDATE: predictor-wise (differential) ridge shrinkage, replacing the single scalar RIDGE_A
(applied identically to all 51 predictors) with a per-predictor penalty based on each predictor's
own trailing marginal reliability.

MECHANISM: uniform ridge shrinks every one of the 51 predictors' loadings by the same fixed amount,
regardless of whether that predictor's own signal is, historically, reliably estimated or mostly
noise. This is distinct from both prior structural changes tested this session: RRR (rejected, 0/14)
shrinks toward a low-rank RESPONSE-side subspace; the beta-demean fix (shipped as v9) transforms the
response Y. This instead reweights the REGULARIZATION itself, predictor by predictor -- an empirical-
Bayes-flavored idea: predictors with weak/noisy trailing marginal signal get MORE shrinkage (pulled
harder toward zero); predictors with strong, consistent trailing marginal signal get LESS.

CAUTION going in: the closest prior idea in this file, per-half-life RIDGE_A (a DIFFERENT axis of
non-uniform shrinkage -- by half-life, not by predictor), was already rejected ("uniformly lose").
That's a mild prior against this succeeding, not a reason to skip testing it -- the mechanism here is
genuinely different (predictor identity, not estimation horizon).

RELIABILITY MEASURE (fully causal, using only trailing history through the day before the current
decision): for predictor i, the average |correlation| between predictor i's return and each of the
50 idio next-day targets, over a trailing window -- pooling across all 50 targets gives a much larger
effective sample than any single pairwise correlation, mirroring how the boost's own leader-search
already pools evidence this way.

PENALTY:  a_i = RIDGE_A * (mean(r) / max(r_i, floor)) ** GAMMA
  GAMMA=0 reproduces uniform RIDGE_A exactly for every predictor (mandatory sanity check). GAMMA>0
  makes below-average-reliability predictors more penalized, above-average less.

Tested ON TOP OF the current best (SAFE_llboost_v9, which already includes the validated beta-
adjusted idio ridge target) -- reuses its _beta_adjusted_target, _pairwise_boost, _algo_vol_shares
verbatim, so the only thing under test is the shrinkage scheme itself.

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


def _ewls_ridge_diag(X, Y, hl, a_vec):
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + np.diag(a_vec) + eps * np.eye(p), XtWY)
    return B, mx, my


def predictor_reliability(X, Y, window):
    """X: (n,51) predictor history, Y: (n,50) idio targets -- both through the day before the
    current decision. Average |corr| of each predictor against all 50 targets, pooled, over the
    trailing `window` rows -- causal."""
    Xw = X[-window:]; Yw = Y[-window:]
    Xc = Xw - Xw.mean(0); Yc = Yw - Yw.mean(0)
    Xs = Xc / (Xc.std(0) + 1e-12); Ys = Yc / (Yc.std(0) + 1e-12)
    C = (Xs.T @ Ys) / Xw.shape[0]
    return np.abs(C).mean(1)


def a_vec_from_r(rel, gamma, floor=0.01):
    if gamma == 0:
        return np.full(rel.shape, RIDGE_A)
    rbar = rel.mean()
    mult = (rbar / np.maximum(rel, floor)) ** gamma
    return RIDGE_A * mult


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


def build_pos(gamma, window):
    POS = np.zeros((nInst, nt))
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Y = V9._beta_adjusted_target(rr_)   # reuse v9's validated response transform verbatim
        xq = rr_[:, -1]
        if gamma != 0:
            rel = predictor_reliability(X, Y, min(window, X.shape[0]))
            a_vec = a_vec_from_r(rel, gamma)
        else:
            a_vec = np.full(X.shape[1], RIDGE_A)
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = _ewls_ridge_diag(X, Y, hl, a_vec)
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


print("\n=== sanity check: gamma=0 must reproduce SAFE_llboost_v9 exactly ===")
POS_base = build_pos(0.0, 250)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  gamma=0: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v9 docstring: 848.8/893.3/894.1/708.6)")
if not (abs(base_wo - 848.8) < 0.5 and abs(base_wn - 893.3) < 0.5):
    print("  *** WARNING: gamma=0 does NOT reproduce v9 -- do not trust results below. ***")
else:
    print("  OK -- matches v9 to within rounding.")


def evaluate(nm, gamma, window, verbose=True):
    Pz = build_pos(gamma, window); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<20}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, gamma=gamma, window=window, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                nworse=nworse, passed=passed, scs=scs)


print("\n=== SWEEP: gamma x window ===")
GAMMAS = [0.5, 1.0, 1.5, 2.0, 3.0]
WINDOWS = [120, 250, 500]
results = []
for w in WINDOWS:
    for g in GAMMAS:
        results.append(evaluate(f"gamma={g} w={w}", g, w))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v9 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: gamma={best['gamma']} w={best['window']}  rmean={best['rm']:.1f}  "
          f"n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  gamma={c['gamma']:<4} w={c['window']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
