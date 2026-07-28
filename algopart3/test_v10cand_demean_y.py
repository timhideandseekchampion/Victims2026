"""
test_v10cand_demean_y.py

CANDIDATE: cross-sectionally demean the ridge's TRAINING TARGET (Y) before fitting, instead of only
demeaning the forecast at prediction time.

MECHANISM: the shipped ridge (`_ewls_ridge` in SAFE_llboost_v8.py) fits each half-life's B on the raw
next-day idio returns Y (T-2, 50). The shipped code DOES cross-sectionally demean the FORECAST before
taking the sign (`fi = pred - pred.mean()`), but the TRAINING TARGET itself still contains each day's
full common-mode return -- the same-day cross-sectional co-movement that `test_pc2_probe.py` measured
at +0.20 average residual correlation even after a ridge fit (a lagged regression cannot explain away
a contemporaneous common factor). If a large share of Y's variance is this shared, largely
unpredictable-from-yesterday's-info component, the least-squares fit spends estimation effort trying
to fit it across all 50 columns simultaneously, correlated noise that could inflate coefficient
variance for the RELATIVE, per-name signal actually used (since the forecast gets demeaned anyway
before sign()). Removing the daily equal-weighted mean from Y BEFORE fitting, rather than only from
the forecast AFTER fitting, targets this directly.

Distinct from RRR (rejected, 0/14): RRR constrained the fitted COEFFICIENT matrix to low rank. This
constrains/transforms the RESPONSE side instead -- a different lever entirely, and unlike RRR it does
not touch the model's degrees of freedom at all (same p, same q, same B shape).

Fully causal: Y[t,:] (next-day return, realized) is known once day t+1 has passed, same information
already used in the raw fit -- cross-sectionally demeaning it introduces no look-ahead.

SWEEP: partial demeaning `Y' = Y - lam * Y.mean(1, keepdims=True)`, lam in [0,1]. lam=0 reproduces
the shipped v8 ridge exactly (mandatory sanity check); lam=1 is full demeaning. This also gives a
natural neighbor-stability read: if a nonzero lam wins, is it a lone spike or a smooth interior optimum?

Baseline = SAFE_llboost_v8 (current shipped best). A candidate must beat v8 on OLD, NEW, and
rolling-mean JOINTLY (this repo's established bar); n_worse against the 61 rolling windows is the
cleanliness metric.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v8 as V8

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V8.WARMUP, V8.BOOST_MIN_DAY, V8.BOOST_K
RIDGE_A = V8.RIDGE_A
HALF_LIVES = V8.HALF_LIVES


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


def _ewls_ridge(X, Y, hl, a):
    """Identical to V8._ewls_ridge (verbatim, so lam=0 reproduces it exactly)."""
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


print("=== precompute: reversal leg, boost, ALGO leg (unchanged by lam -- identical to v8) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V8.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V8._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V8._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(lam):
    POS = np.zeros((nInst, nt))
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Yraw = rr_[1:, 1:].T
        Y = Yraw - lam * Yraw.mean(1, keepdims=True) if lam != 0 else Yraw
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = _ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        wz = (1 - V8.BLEND) * wz + V8.BLEND * REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: lam=0 must reproduce SAFE_llboost_v8 exactly ===")
POS_base = build_pos(0.0)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  lam=0: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v8 docstring: 847.4/888.9/886.2/674.4)")
if not (abs(base_wo - 847.4) < 0.5 and abs(base_wn - 888.9) < 0.5):
    print("  *** WARNING: lam=0 does NOT reproduce v8 -- do not trust results below. ***")
else:
    print("  OK -- matches v8 to within rounding.")


def evaluate(nm, lam, verbose=True):
    Pz = build_pos(lam); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<12}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, lam=lam, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse,
                passed=passed, scs=scs)


print("\n=== SWEEP: partial demeaning lam in [0,1] (lam=0 shipped, lam=1 full demean) ===")
LAMS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
results = [evaluate(f"lam={lam}", lam) for lam in LAMS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} lam values beat v8 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: lam={best['lam']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
    idx = LAMS.index(best["lam"])
    neighbors = LAMS[max(0, idx - 1):idx + 2]
    print("\n=== neighbor-stability check ===")
    for lam in neighbors:
        r_ = evaluate(f"  lam={lam}", lam, verbose=False)
        tag = " <== best" if lam == best["lam"] else ""
        print(f"  lam={lam:<4} OLD={r_['wo']:.1f} NEW={r_['wn']:.1f} rmean={r_['rm']:.1f} "
              f"rfloor={r_['rf']:.1f} n_worse={r_['nworse']}/61{tag}")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:5]:
        print(f"  lam={c['lam']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
