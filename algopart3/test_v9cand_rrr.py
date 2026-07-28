"""
test_v9cand_rrr.py

CANDIDATE: reduced-rank regression (RRR) on the idio ridge ensemble, replacing SAFE_llboost_v8's
uniform-shrink-to-zero EW ridge with a shrink-toward-the-correct-low-rank-subspace estimator.

MECHANISM (why this is a real hypothesis, not just "try another ridge variant"): this repo's own
documented, exhaustively-mapped finding is that the synthetic market is a one-factor model (ALGO/PC1
common factor + lead-lag + idiosyncratic noise; see stress_test_synthetic.py, test_pc2_probe.py). The
idio ridge (`_ewls_ridge` in SAFE_llboost_v8.py) fits a 51x50 coefficient matrix B per half-life (all
51 instruments' current returns -> the 50 idio names' next-day returns) and regularizes by shrinking
EVERY ONE of the 2550 coefficients uniformly toward ZERO via a single scalar (RIDGE_A=0.1). If the
true relationship is genuinely low-rank, uniform shrinkage-to-zero is the wrong prior -- RRR shrinks
toward the correct low-rank SUBSPACE instead.

Honest evidence check (test_pc2_probe.py, re-run to confirm before trusting the story): PC1 (~ALGO)
has real predictive power (p=0%). PC2 does NOT (p=25%, clearly null). PC3 is borderline (p=9%, not
conventionally significant). Net read: a real dominant factor exists (supports trying rank reduction)
but not obviously a SECOND one beyond it -- so watch for a sharp isolated optimum at r=1 rather than
a broad plateau (see the ranking/verdict section at the bottom: per this repo's neighbor-stability
convention, a spike reads as suspicious/overfit, not a clean pass, even if the raw numbers look good).

THE ESTIMATOR (verified correct by hand -- do not naively SVD-truncate raw B, and do not weight by
unregularized X'X; both give a statistically different, inferior estimator):
Given the existing per-half-life fit's S = XtWX + (eps+a)*I (p x p) and B = S^-1 XtWY (p x q), the
loss being minimized is L(C) = sum_t w_t||y_t - C'x_t||^2 + (eps+a)*(sum w_t)*||C||_F^2. Completing
the square: L(C) = const + tr((C-B)'S(C-B)) -- minimizing this subject to rank(C)<=r is a WEIGHTED
(by S) low-rank approximation of B, whose closed form is C_r = B @ V_r @ V_r' where V_r are the top-r
right singular vectors of D0 = L' @ B (S = L @ L', Cholesky -- numerically stable, avoids forming
B'SB directly). At r >= min(p,q)=50, this reproduces B exactly (a mandatory sanity check below).

EFFICIENCY: the expensive part (B, S, its Cholesky, the SVD of D0) depends only on (day, half-life),
NOT on rank r. Precomputed ONCE below; every candidate rank is then a cheap O(q*r) projection reusing
the already-computed full-rank prediction offset ("full"), so the whole rank sweep costs barely more
than building the baseline once.

DESIGN: one shared rank across all 4 half-lives (not per-half-life independent ranks) -- direct repo
precedent: the README's 80-idea ledger already found "per-half-life RIDGE_A... uniformly lose"; a
per-half-life rank is the same kind of fragmentation of a single shared regularization knob. RIDGE_A
held fixed at the shipped 0.1 in this primary sweep (S depends on it, so it isn't a free axis the way
r is -- a joint confirmatory grid is a separate follow-up, only if this sweep looks promising).

Baseline = SAFE_llboost_v8 (the current shipped best, NOT v7) -- reuses V8._pairwise_boost and
V8._algo_vol_shares verbatim (the ALGO leg is untouched by this candidate; called sequentially in
increasing-day order exactly once so v8's real HOLD-deadband state is reproduced correctly, per
validate_llboost_v8_full.py's own convention). A candidate must beat v8 on OLD, NEW, and rolling-mean
JOINTLY to "pass" (test_v7_algo_deadband_v2.py's convention); n_worse against the 61 rolling windows
is the cleanliness metric.
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
n_hl = len(HALF_LIVES)


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


# ==================================================================================================
# RRR core: fit once (B/S/SVD), truncate to any rank cheaply afterward
# ==================================================================================================
def _ewls_rrr_fit(X, Y, hl, a, xq):
    """Identical fit to V8._ewls_ridge, additionally returns what's needed for cheap rank truncation:
    pred_r = my + (full @ Vt[:r].T) @ Vt[:r]   -- and at r>=min(p,q)=q, Vt[:r] is the FULL square
    orthogonal Vt (since p=51>q=50 here), so this reduces to my+full exactly, no special-case needed."""
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    S = XtWX + (eps + a) * np.eye(p)
    B = np.linalg.solve(S, XtWY)
    L = np.linalg.cholesky(S)
    D0 = L.T @ B
    _, _, Vt = np.linalg.svd(D0, full_matrices=False)   # Vt: min(p,q) x q, top singular vec first
    full = (xq - mx) @ B
    return my, full, Vt


def _rrr_pred(my, full, Vt, rank):
    rr_ = min(rank, Vt.shape[0])
    coef = full @ Vt[:rr_].T
    proj = coef @ Vt[:rr_]
    return my + proj


print("=== precompute: RRR cache (fit + SVD once per day/half-life; rank sweep will be cheap) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
n_days = len(days)
MY = np.zeros((n_hl, n_days, nIdio))
FULL = np.zeros((n_hl, n_days, nIdio))
VT = np.zeros((n_hl, n_days, nIdio, nIdio))
for di, t in enumerate(days):
    rr_ = r[:, :t]
    X = rr_[:, :-1].T; Y = rr_[1:, 1:].T; xq = rr_[:, -1]
    for hi, hl in enumerate(HALF_LIVES):
        my, full, Vt = _ewls_rrr_fit(X, Y, hl, RIDGE_A, xq)
        MY[hi, di] = my; FULL[hi, di] = full; VT[hi, di] = Vt
print(f"  done ({time.time()-t0:.0f}s, {n_days} days x {n_hl} half-lives)", flush=True)

print("=== precompute: reversal leg, boost, ALGO leg (unchanged by rank -- identical to v8) ===",
      flush=True)
t0 = time.time()
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


def build_pos(rank):
    POS = np.zeros((nInst, nt))
    for di, t in enumerate(days):
        fs = []
        for hi in range(n_hl):
            pred = _rrr_pred(MY[hi, di], FULL[hi, di], VT[hi, di], rank)
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


# ==================================================================================================
# MANDATORY sanity check: rank=50 (=min(p,q)) must reproduce SAFE_llboost_v8 exactly
# ==================================================================================================
print("\n=== sanity check: rank=50 (full rank) must reproduce SAFE_llboost_v8 exactly ===")
POS_full = build_pos(50)
full_scs = scs_curve(POS_full)
full_wo, full_wn = wscore(POS_full, *OLD), wscore(POS_full, *NEW)
print(f"  rank=50 (backtest-equiv): OLD={full_wo:.1f}  NEW={full_wn:.1f}  rmean={full_scs.mean():.1f}  "
      f"rfloor={full_scs.min():.1f}   (v8 docstring: 847.4/888.9/886.2/674.4)")
if not (abs(full_wo - 847.4) < 0.5 and abs(full_wn - 888.9) < 0.5):
    print("  *** WARNING: rank=50 does NOT reproduce v8's numbers -- STOP, do not trust any "
          "smaller-rank result until this is fixed. ***")
else:
    print("  OK -- matches v8 to within rounding. Safe to trust the rank sweep below.")

base_scs = full_scs
base_wo, base_wn = full_wo, full_wn


def evaluate(nm, rank, verbose=True):
    Pz = build_pos(rank); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<12}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, rank=rank, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse,
                passed=passed, scs=scs)


# ==================================================================================================
print("\n=== RANK SWEEP vs SAFE_llboost_v8 (dense at the low end -- see PC2/PC3 evidence above) ===")
RANKS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 35, 50]
results = []
for rk in RANKS:
    results.append(evaluate(f"r={rk}", rk))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} ranks beat v8 on OLD+NEW+rmean jointly.")

if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"\nbest by rmean: r={best['rank']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")

    print("\n=== neighbor-stability check around the winner (plateau vs isolated spike) ===")
    idx = RANKS.index(best["rank"])
    neighbors = RANKS[max(0, idx - 1):idx + 2]
    for rk in neighbors:
        r_ = evaluate(f"  r={rk}", rk, verbose=False)
        tag = " <== best" if rk == best["rank"] else ""
        print(f"  r={rk:<3} OLD={r_['wo']:.1f} NEW={r_['wn']:.1f} rmean={r_['rm']:.1f} "
              f"rfloor={r_['rf']:.1f} n_worse={r_['nworse']}/61{tag}")

    spike = len(neighbors) < 2 or not all(
        evaluate(f"nbr{rk}", rk, verbose=False)["passed"] for rk in neighbors if rk != best["rank"])
    if spike:
        print("\n  *** SHARP/ISOLATED -- neighbors do not also pass -- treat as suspicious, not a "
              "clean win, per this repo convention ***")
    else:
        print("\n  plateau confirmed -- neighbors also pass")
else:
    print("\nNo rank beats v8 on OLD+NEW+rmean jointly. Ranking by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  r={c['rank']:<3} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
    print("\nRRR does not clear the bar against SAFE_llboost_v8 -- stopping here, no v9 file, no "
          "RIDGE_A confirmatory grid. Verdict to be written up in README.md as a rejected idea.")
