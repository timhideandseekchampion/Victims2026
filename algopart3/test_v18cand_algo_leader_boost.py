"""
test_v18cand_algo_leader_boost.py

NEW hypothesis (not previously tested in this file's history): the pairwise boost's leader search
(`_pairwise_boost`) only ever considers OTHER IDIO NAMES as candidate "leaders" for a given idio
follower -- ALGO itself is excluded from the candidate pool, even though ALGO already sits inside
the linear ridge as one of 51 predictors. The ridge only captures a LINEAR ALGO->name relationship;
the boost's own convex sign(x)*(|x|/scale)^P transform (P=2.0, validated) is specifically designed to
capture a nonlinear one -- "big index moves predict disproportionately bigger idio moves in sensitive
names" (a crash-beta/tail-sensitivity effect) is a structurally different hypothesis from the ridge's
linear beta, and untested here. The previously-rejected "ALGO crossover" idea (v17) tested something
unrelated: ALGO's own price pattern predicting ALGO's OWN next return (time-series). This tests
ALGO's move predicting OTHER STOCKS' next moves (cross-sectional), the leader/follower framing
already validated for idio-vs-idio pairs, extended by exactly one candidate.

DESIGN: add ALGO as a 40th, unconditionally-included candidate leader (on top of the existing
BOOST_N_CANDIDATES=39 vol-ranked idio candidates), Bonferroni divisor raised 39->40 to control the
one extra simultaneous test. Every other mechanic (BOOST_P=2.0 convex transform, BOOST_SCALE_W=1000
leader-scale window, BOOST_IC_L=250 sign-check window, BOOST_MIN_DAY=480, `ic<=0: discard`) is
IDENTICAL and reused verbatim -- this isolates exactly one variable: is ALGO a useful *additional*
leader candidate, never whether the existing mechanics need retuning.

Tested against the real, shipped SAFE_llboost_v10 (current best) -- ridge, beta-demean target, REV
blend, ALGO leg, and the rank-stability blend are all reused verbatim, unchanged.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs_full = r[1:]
algo_r_full = r[0]
nIdio, nt_r = rs_full.shape
WARMUP = V10.WARMUP


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


def _sig_threshold_n(n_samples, n_cand):
    if n_samples < 10:
        return 1.0
    alpha_adj = V10.BOOST_ALPHA / n_cand
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def _pairwise_boost_with_algo(rs, algo_ret):
    """Identical to V10._pairwise_boost, plus ALGO unconditionally appended as one extra candidate
    leader (Bonferroni divisor raised by 1 to match)."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V10.BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = _sig_threshold_n(n_samples, V10.BOOST_N_CANDIDATES + 1)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V10.BOOST_N_CANDIDATES]
    Xi_idio = Xi_full[cand_idx]
    algo_x = algo_ret[:T - 1][None, :]
    Xi = np.vstack([Xi_idio, algo_x])
    C = V10._corrmat(Xi, Yj)
    n_idio_cand = len(cand_idx)
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
        is_algo = (ci == n_idio_cand)
        lead = algo_ret[:T] if is_algo else rs[cand_idx[ci]]
        scale = np.nanstd(lead[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V10.BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


print("=== precompute: ridge+REV base (WZ_PRE, boost-independent), ALGO leg, both boost variants ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))

WZ_PRE = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in V10.HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, V10.RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V10.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
        rv_ = rv_ - rv_.mean()
        rv_ = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - V10.BLEND) * wz + V10.BLEND * rv_
    WZ_PRE[:, t] = wz
print(f"  ridge+REV done ({time.time()-t0:.0f}s)", flush=True)

def _pairwise_boost_stricter_divisor(rs):
    """Control: identical to V10._pairwise_boost (idio-only candidates, no ALGO), but with the
    Bonferroni divisor raised 39->40 anyway -- isolates whether the drop-in's loss comes from the
    stricter multiple-testing correction alone, or from ALGO's presence in the pool."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V10.BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = _sig_threshold_n(n_samples, V10.BOOST_N_CANDIDATES + 1)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V10.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
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
        lead = rs[i]
        scale = np.nanstd(lead[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V10.BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


t0 = time.time()
BOOST_BASE = np.zeros((nIdio, nt))
BOOST_ALGO = np.zeros((nIdio, nt))
BOOST_STRICT = np.zeros((nIdio, nt))
for k in range(V10.BOOST_MIN_DAY, nt):
    BOOST_BASE[:, k] = V10._pairwise_boost(rs_full[:, :k])
    BOOST_ALGO[:, k] = _pairwise_boost_with_algo(rs_full[:, :k], algo_r_full[:k])
    BOOST_STRICT[:, k] = _pairwise_boost_stricter_divisor(rs_full[:, :k])
print(f"  all boost variants done ({time.time()-t0:.0f}s)", flush=True)

t0 = time.time()
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  ALGO leg done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(boost_arr):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t] + V10.BOOST_K * boost_arr[:, t]
        rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
        if rs_sig is not None:
            s_std = rs_sig.std()
            s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
            wz = (1 - V10.RS_WEIGHT) * wz + V10.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: BOOST_BASE path must reproduce shipped SAFE_llboost_v10 exactly ===")
t0 = time.time()
POS_base = build_pos(BOOST_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   ({time.time()-t0:.0f}s)")
print("  (v10 docstring: 871.0 / 912.6 / 909.8 / 709.7)")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, boost_arr, verbose=True):
    Pz = build_pos(boost_arr); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== how often is ALGO actually selected as a leader, once eligible? ===")
active_days = range(V10.BOOST_MIN_DAY, nt)
algo_leader_days = sum(1 for k in active_days if (BOOST_ALGO[:, k] != 0).any() and
                        not np.array_equal(BOOST_ALGO[:, k], BOOST_BASE[:, k]))
print(f"  boost output differs from baseline on {algo_leader_days}/{len(list(active_days))} days")

print("\n=== SWEEP: drop-in ALGO-as-extra-leader vs shipped v10 boost ===")
evaluate("ALGO-as-40th-leader", BOOST_ALGO)
print("\n=== CONTROL: stricter divisor alone (idio-only pool, no ALGO) -- isolates the mechanism ===")
evaluate("stricter-divisor-only", BOOST_STRICT)
