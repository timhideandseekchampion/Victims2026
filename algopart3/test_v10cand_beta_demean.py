"""
test_v10cand_beta_demean.py

Follow-up to test_v10cand_demean_y.py, which found UNIFORM cross-sectional demeaning of the ridge's
training target Y is a PROVABLE NO-OP here (verified algebraically and numerically, not just
empirically): since the ridge fit is linear in Y, subtracting the SAME value from every one of the
50 response columns on a given training day shifts every stock's forecast by an identical constant
that day -- which the shipped code's own `fi = pred - pred.mean()` step removes anyway. Every lam in
[0,1] gave bit-identical scores to v8, confirming this exactly (n_worse=0/61 at every lam, meaning
literally zero windows differ).

THIS test makes the transform non-trivial the natural way: instead of subtracting the SAME common-
mode value from every stock, subtract beta_j * (common mode) using each stock's OWN causally-
estimated beta to the idio cross-sectional common factor. Since beta_j varies by stock, the
correction is no longer a uniform per-day shift, so it does NOT cancel under the later demeaning step
(verified numerically on synthetic data before writing this test: a beta-weighted version changes the
z-scored forecast by a real amount, unlike the uniform version).

MECHANISM: the common factor here is the daily equal-weighted average return across the 50 idio names
(the same "common mode" `test_pc2_probe.py` found +0.20 average same-day residual correlation for,
even after a ridge fit). beta_j = stock j's own trailing covariance with that factor / the factor's
trailing variance -- a standard market-model beta, but against the IDIO common mode rather than
ALGO specifically (ALGO is already a predictor in X; this targets the residual co-movement left in Y
that isn't routed through ALGO). Removing beta_j*factor from Y before fitting lets the ridge spend its
degrees of freedom on each stock's IDIOSYNCRATIC-to-its-own-beta relative return, not on jointly
explaining a shared, largely unpredictable-from-yesterday's-info component.

Fully causal: beta_j is estimated from trailing history only (window BETA_W ending the day before the
current decision day), and the daily common-mode factor used to adjust Y is itself built from already-
realized (historical) idio returns -- no look-ahead.

SWEEP: lam in [0,1] (partial beta-adjustment) x BETA_W in {120, 250, 500} (how much trailing history
estimates each stock's own beta). lam=0 must reproduce v8 exactly (mandatory sanity check).

Baseline = SAFE_llboost_v8. Must beat it on OLD, NEW, rolling-mean JOINTLY to pass.
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

# daily idio common-mode factor (equal-weighted average return across the 50 idio names, fixed once)
CF = rs.mean(0)   # length nt-1, CF[i] corresponds to rs[:, i] i.e. r column i+1 (idio rows)


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
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


def betas_at(t, beta_w):
    """Causal per-stock beta to CF, using the BETA_W days of history strictly before day t's
    training set ends (rows 0..t-2 of rs/CF, matching Y's own row alignment -- see docstring)."""
    lo = max(0, (t - 1) - beta_w)
    hi = t - 1   # rs/CF index range used by Y at this t is rows [0, t-2] i.e. up to index t-2 inclusive -> hi=t-1 (exclusive)
    if hi - lo < 30:
        return np.ones(nIdio)   # not enough history yet -- fall back to beta=1 (reduces to uniform demean, itself a no-op, i.e. inert until enough history)
    seg_y = rs[:, lo:hi]; seg_f = CF[lo:hi]
    vf = seg_f.var()
    if vf < 1e-24:
        return np.ones(nIdio)
    cov = (seg_y * seg_f[None, :]).mean(1) - seg_y.mean(1) * seg_f.mean()
    return cov / vf


print("=== precompute: reversal leg, boost, ALGO leg (unchanged by lam/BETA_W -- identical to v8) ===",
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


def build_pos(lam, beta_w):
    POS = np.zeros((nInst, nt))
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Yraw = rr_[1:, 1:].T
        xq = rr_[:, -1]
        if lam != 0:
            b = betas_at(t, beta_w)
            cf_seg = CF[1:t]         # aligned to Y's row index (see test_v10cand_demean_y.py's derivation)
            Y = Yraw - lam * b[None, :] * cf_seg[:, None]
        else:
            Y = Yraw
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
POS_base = build_pos(0.0, 250)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  lam=0: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v8 docstring: 847.4/888.9/886.2/674.4)")
if not (abs(base_wo - 847.4) < 0.5 and abs(base_wn - 888.9) < 0.5):
    print("  *** WARNING: lam=0 does NOT reproduce v8 -- do not trust results below. ***")
else:
    print("  OK -- matches v8 to within rounding.")


def evaluate(nm, lam, beta_w, verbose=True):
    Pz = build_pos(lam, beta_w); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<20}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, lam=lam, beta_w=beta_w, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                nworse=nworse, passed=passed, scs=scs)


print("\n=== SWEEP: lam x BETA_W ===")
LAMS = [0.2, 0.4, 0.6, 0.8, 1.0]
BETA_WS = [120, 250, 500]
results = []
for bw in BETA_WS:
    for lam in LAMS:
        results.append(evaluate(f"lam={lam} bw={bw}", lam, bw))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v8 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: lam={best['lam']} bw={best['beta_w']}  rmean={best['rm']:.1f}  "
          f"n_worse={best['nworse']}/61")
    print("\n=== neighbor-stability check ===")
    for lam in LAMS:
        for bw in BETA_WS:
            if lam == best["lam"] and bw == best["beta_w"]:
                continue
            if abs(LAMS.index(lam) - LAMS.index(best["lam"])) <= 1 and bw == best["beta_w"]:
                r_ = evaluate(f"  lam={lam} bw={bw}", lam, bw, verbose=False)
                print(f"  lam={lam:<4} bw={bw:<4} OLD={r_['wo']:.1f} NEW={r_['wn']:.1f} "
                      f"rmean={r_['rm']:.1f} n_worse={r_['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  lam={c['lam']:<4} bw={c['beta_w']:<4} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
