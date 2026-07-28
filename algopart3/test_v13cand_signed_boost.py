"""
test_v13cand_signed_boost.py

CANDIDATE: let the pairwise boost use NEGATIVE leader relationships (invert instead of discard).

MECHANISM: `_pairwise_boost` selects each follower j's leader by the single strongest ABSOLUTE
correlation among the 39 candidates (`ci = argmax(|col|)`, `if abs(col[ci]) <= thr: continue`) --
already symmetric to sign at the SELECTION and SIGNIFICANCE stage. But the realized-quality recheck
that follows is NOT symmetric:
    lead_boost = sign(lead) * (|lead|/scale)**BOOST_P      # same sign as the leader's own move
    ic = corr(lead_boost, follower's next return)
    if ic <= 0: continue                                    # <-- discards negative relationships
If leader and follower are genuinely INVERSELY related (leader up -> follower down), `ic` comes out
negative and the pair is thrown away entirely, even though a Bonferroni-significant, structurally
real relationship was just found. The fix: when `ic < 0`, INVERT the boost instead of discarding it
(`-lead_boost[-1]` moves opposite to the leader's last move, which is exactly the up-leader-down-
follower relationship). This can only ever ADD boosts that were previously discarded -- every pair
that already passed with ic>0 is untouched -- so any score change is attributable to what was
previously being thrown away.

Tested ON TOP OF the current best (SAFE_llboost_v9) -- reuses its `_beta_adjusted_target`,
`_algo_vol_shares` verbatim; only `_pairwise_boost`'s final ic-sign handling changes.

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
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W = (
    V9.BOOST_N_CANDIDATES, V9.BOOST_IC_L, V9.BOOST_P, V9.BOOST_SCALE_W)
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


def _pairwise_boost_signed(rsl, allow_negative, mag_floor=0.0):
    """Identical to V9._pairwise_boost's leader SELECTION and SIGNIFICANCE gate (verbatim). Only the
    final ic-sign handling changes: allow_negative=False reproduces V9 exactly (mandatory sanity
    check); allow_negative=True inverts instead of discarding when ic<0 (and |ic|>=mag_floor)."""
    n, T = rsl.shape
    boost = np.zeros(n)
    if T < BOOST_MIN_DAY:
        return boost
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V9._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V9._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if not allow_negative:
            if ic <= 0:
                continue
            boost[j] = lead_boost[-1]
        else:
            if abs(ic) <= mag_floor:
                continue
            boost[j] = lead_boost[-1] if ic > 0 else -lead_boost[-1]
    return boost


print("=== precompute: reversal leg, ALGO leg (unchanged -- reused verbatim from v9) ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V9.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V9._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_RIDGE = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V9._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V9._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    WZ_RIDGE[:, t] = (1 - V9.BLEND) * wz + V9.BLEND * REV[:, t]
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(allow_negative, mag_floor=0.0):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_RIDGE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            boost = _pairwise_boost_signed(rs[:, :t], allow_negative, mag_floor)
            wz = wz + BOOST_K * boost
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: allow_negative=False must reproduce SAFE_llboost_v9 exactly ===")
POS_base = build_pos(False)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v9 docstring: 848.8/893.3/894.1/708.6)")
if not (abs(base_wo - 848.8) < 0.5 and abs(base_wn - 893.3) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v9 -- do not trust results below. ***")
else:
    print("  OK -- matches v9 to within rounding.")

# how many extra boosts does this unlock, and how many flip sign vs staying positive-only?
n_pos = 0; n_neg_unlocked = 0
for t in range(BOOST_MIN_DAY, nt):
    b0 = _pairwise_boost_signed(rs[:, :t], False)
    b1 = _pairwise_boost_signed(rs[:, :t], True)
    n_pos += int((b0 != 0).sum())
    n_neg_unlocked += int(((b0 == 0) & (b1 != 0)).sum())
print(f"\n  baseline (positive-only) boost coverage: {n_pos} stock-days")
print(f"  NEWLY unlocked (previously-discarded negative-IC) stock-days: {n_neg_unlocked} "
      f"(+{100*n_neg_unlocked/max(n_pos,1):.0f}%)")


def evaluate(nm, allow_negative, mag_floor=0.0, verbose=True):
    Pz = build_pos(allow_negative, mag_floor); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<24}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, allow_negative=allow_negative, mag_floor=mag_floor, wo=wo, wn=wn,
                rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed, scs=scs)


print("\n=== PRIMARY TEST: allow_negative=True (pure sign rule, matches shipped code's own bar) ===")
evaluate("allow_negative=True", True, 0.0)

print("\n=== sweep: require a minimum |ic| before trusting the flip (avoid near-zero noise flips) ===")
results = [evaluate(f"mag_floor={mf}", True, mf) for mf in (0.02, 0.05, 0.08, 0.10, 0.15)]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} mag_floor configs beat v9 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: mag_floor={best['mag_floor']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
