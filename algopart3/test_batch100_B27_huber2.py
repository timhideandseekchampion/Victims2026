"""
test_batch100_B27_huber.py

B27: Re-test Huber-robustified ridge (IRLS) against v10. Previously tested (test_v12cand_huber.py)
against SAFE_llboost_v9 -- BEFORE the beta-adjusted ridge target existed on top of it... wait, v9
*already* had the beta-adjusted target (v9 = beta-demean, v10 = v9 + rank-stability). The task's
framing ("was tested pre-beta-demean") is re-verified against the actual file history: v12's baseline
WAS v9 (beta-adjusted target already active), so the ridge target itself hasn't changed since v12 --
what's NEW since then is the rank-stability blend layered on top in v10. Re-testing directly against
the CURRENT best (v10, target unchanged, but now with rank-stability additionally in the pipeline)
rather than assuming the old v12 numbers still apply once rank-stability sits downstream of the ridge.

MECHANISM: identical IRLS Huber-weighted ridge from test_v12cand_huber.py (one combined per-day
robustness weight from the pooled, z-scored residual magnitude across all 50 targets, multiplied into
the EW time-decay weight -- keeps the shared-weight closed-form ridge solve intact). huber_k=None
(pure EW, n_irls=1) must reproduce SAFE_llboost_v10 exactly.

SWEEP: huber_k around the old (pre-rank-stability) spike at 1.20-1.22, plus wider neighbors, to see if
that "isolated spike, not a plateau" verdict (README) still holds now that rank-stability sits
downstream. REV/boost/rank-stability/ALGO leg cached once (independent of the ridge estimator) and
reused across every huber_k, exactly like test_v19cand_boost_ncandidates.py's caching pattern.
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
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT


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
        Ez = E / (E.std(0, keepdims=True) + 1e-12)
        d = np.sqrt((Ez ** 2).mean(1))
        med = np.median(d) + 1e-12
        delta = huber_k * med
        huber_w = np.where(d <= delta, 1.0, delta / d)
        w = w_ew * huber_w
    return B, mx, my


print("=== precompute: BLEND reversion, pairwise boost, rank-stability signal, ALGO leg -- IDENTICAL "
      "for every huber_k (independent of the ridge estimator), cached once ===", flush=True)
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
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        RS_SIG[:, t] = rs_sig
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(huber_k, n_irls):
    POS = np.zeros((nInst, nt))
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Y = V10._beta_adjusted_target(rr_)
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = _huber_ridge(X, Y, hl, huber_k, n_irls)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: huber_k=None (pure EW, n_irls=1) must reproduce SAFE_llboost_v10 exactly ===")
POS_base = build_pos(None, 1)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


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
                nworse=nworse, passed=passed)


print("\n=== SWEEP: huber_k (n_irls=2), dense around the old 1.20-1.22 spike, wider neighbors too ===")
HUBER_KS = [0.8, 1.0, 1.15, 1.18, 1.20, 1.21, 1.22, 1.25, 1.5, 2.0, 3.0, 5.0]
results = [evaluate(f"huber_k={hk}", hk, 2) for hk in HUBER_KS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: huber_k={best['huber_k']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
    idx = HUBER_KS.index(best['huber_k'])
    neighbors = HUBER_KS[max(0, idx - 1):idx + 2]
    print("neighbor-stability check:")
    nbr_pass = []
    for hk in neighbors:
        r_ = evaluate(f"  nbr huber_k={hk}", hk, 2, verbose=False)
        nbr_pass.append(r_["passed"])
        tag = " <== best" if hk == best['huber_k'] else ""
        print(f"  huber_k={hk:<6} OLD={r_['wo']:.1f} NEW={r_['wn']:.1f} rmean={r_['rm']:.1f} "
              f"n_worse={r_['nworse']}/61 passed={r_['passed']}{tag}")
    if not all(nbr_pass):
        print("  *** SHARP/ISOLATED spike -- neighbors do not also pass -- suspicious, not a clean win ***")
    else:
        print("  plateau confirmed -- neighbors also pass")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  huber_k={c['huber_k']:<6} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
