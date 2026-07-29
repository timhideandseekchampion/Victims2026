"""
test_batch100_catA_peer_aggregation.py

CATEGORY A of the current batch: "aggregate multiple peers" variants of the shipped pairwise boost,
which (in SAFE_llboost_v10) picks exactly ONE best-|corr| Bonferroni-significant leader per follower
and boosts the follower's forecast by that single leader's convex-transformed move. HYPOTHESIS: a
single leader is a noisy estimator of "what the informed peers are doing" -- averaging over several
peers (either restricted to the same statistically-significant leaders, or a plain unconditional peer
/cluster consensus) might be a lower-variance version of the same lead-lag edge. Tested as four
distinct, independently-evaluated ideas against the real shipped v10 baseline via the verified
`_v10_harness.py`. Most repo history on "multi-leader" ideas has failed (see README's 100-idea sweep:
"new lead-lag variants (multi-leader averaging, cluster-restricted pools, ...) ... all rejected
cleanly") -- these four are more precisely specified re-attempts, checked against that history and
believed to be genuinely new constructions, not exact repeats.

Fully causal throughout: every feature at day t (harness convention: the most recently known idio
return is at column t-1 of `rs_full`/`H.r`) uses only data through that column. No future indices.

--------------------------------------------------------------------------------------------------
IDEA 1 -- Weighted multi-leader blend
--------------------------------------------------------------------------------------------------
Same candidate pool (top BOOST_N_CANDIDATES=39 by trailing realized vol) and same Bonferroni
significance bar (`_sig_threshold`) as shipped. Instead of keeping only the single best-|corr|
significant candidate per follower, keep up to the top-3 by |corr| that clear the bar. Each of those
candidates is *also* required to individually pass the shipped secondary stability gate (rolling
BOOST_IC_L-day IC of its own convex-transformed lead value vs the follower's forward return must be
positive) -- i.e. every one of the (0-3) selected leaders passes BOTH shipped gates, exactly as the
single leader would have. Contribution = sign(x)*(|x|/scale)**BOOST_P (identical convex transform,
each leader's own trailing-BOOST_SCALE_W-day scale), weighted by |corr| normalized to sum to 1 across
however many leaders actually clear both bars that day, summed.//
SIMPLIFICATION vs. a literal reading of the assignment: the assignment text only names the
correlation/significance bar as "the bar" leaders must clear; we additionally kept the shipped
IC-positive stability gate per leader (dropping it felt like an unfair advantage over the single-leader
baseline, since it exists precisely to filter out leaders whose convex-transformed signal doesn't
actually forecast the follower). Documented explicitly since it's a judgment call.

--------------------------------------------------------------------------------------------------
IDEA 2 -- Peer-consensus broadcast
--------------------------------------------------------------------------------------------------
SIG[j,t] = mean return (at the most recently known day, column t-1) of stock j's 3 peers with the
highest trailing-250-day |correlation| to j -- a plain unconditional top-3-by-|corr| set, NO
significance gate at all (deliberately distinct from Idea 1 and from the shipped mechanism). Peer sets
recomputed every 20 trading days for speed (still fully causal -- each recompute only uses data through
the day it's computed on). Blended into the *already-complete* v10 forecast (WZ_PRE + BOOST_K*shipped
boost + rank-stability, i.e. `H.BASE_WZ`) via `H.blend_signal`, weight swept over {0.02, 0.05, 0.1, 0.2}.

--------------------------------------------------------------------------------------------------
IDEA 3 -- Leader-surprise boost
--------------------------------------------------------------------------------------------------
Identical leader SELECTION to shipped (single best-|corr| Bonferroni-significant candidate, same
secondary IC-positive stability gate). The only change: the quantity fed through the convex transform
is the leader's "surprise" -- its return minus a short trailing EW mean (halflife=10 days, computed
causally with a strict shift so today's surprise never uses today's own value in the mean) of its OWN
past returns -- instead of its raw return. Scale (for the (|x|/scale)**BOOST_P normalization) is
likewise computed from the surprise series' own trailing-BOOST_SCALE_W-day std, analogous to how the
shipped scale is computed from the raw leader's own trailing std. HYPOTHESIS: an unusually large
*surprise* move (relative to the leader's own recent behavior) may be a cleaner lead-lag trigger than
the leader's raw, potentially trend-inflated, return.

--------------------------------------------------------------------------------------------------
IDEA 4 -- Cluster momentum
--------------------------------------------------------------------------------------------------
Every 50 trading days, hierarchical (average-linkage) clustering of the 50 idio names into 6 clusters
using a trailing-250-day correlation distance (1-corr), computed causally on data through the day of
the recompute. SIG[j,t] = mean return (column t-1) of stock j's cluster-mates, excluding j itself (0 if
j is a singleton cluster that period). Blended into `H.BASE_WZ` via `H.blend_signal`, weight swept over
{0.02, 0.05, 0.1, 0.2} (same grid as Idea 2, for comparability -- not separately re-derived).
SIMPLIFICATION: cluster count fixed at 6 (middle of the assignment's "~5-8" range), not itself swept --
only the blend weight is swept, consistent with what the assignment asked for.

--------------------------------------------------------------------------------------------------
Pass bar (identical for all four, via `H.evaluate`): must beat the real shipped v10 baseline on OLD
(days 501-750), NEW (751-1000), AND rolling-61-window mean, simultaneously. Most ideas are expected to
fail, matching this repo's extensive multi-leader/peer-aggregation track record -- reported honestly
either way.
"""
import numpy as np
import pandas as pd
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import _v10_harness as H

V10 = H.V10
rs_full = H.rs_full          # (nIdio, nt-1)
nIdio, nt = H.nIdio, H.nt

RESULTS = []


# ====================================================================================================
# IDEA 1 -- weighted multi-leader blend (top-3 Bonferroni-significant, |corr|-weighted)
# ====================================================================================================
def _pairwise_boost_multi(rs, k=3):
    """Same mechanics as V10._pairwise_boost, generalized from "keep only the single best leader" to
    "keep up to the top-k leaders that individually clear BOTH shipped gates" (correlation-significance
    AND the rolling-IC-positive stability check), weighted by |corr| normalized to sum to 1."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V10.BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]
    Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V10.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)              # (n_cand, n)

    # precompute each candidate's convex-transformed lead_boost series ONCE (shared across followers)
    scale = np.nanstd(rs[cand_idx][:, max(0, T - 1 - V10.BOOST_SCALE_W):T - 1], axis=1) + 1e-12
    lead_boost_all = np.sign(rs[cand_idx]) * (np.abs(rs[cand_idx]) / scale[:, None]) ** V10.BOOST_P
    a = max(0, T - 1 - V10.BOOST_IC_L)

    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        valid = np.where(~np.isnan(col) & (np.abs(col) > thr))[0]
        if len(valid) == 0:
            continue
        valid = valid[np.argsort(-np.abs(col[valid]))][:k]
        ws, contribs = [], []
        for ci in valid:
            xs = lead_boost_all[ci, a:T - 1]
            ys = rs[j, a + 1:T]
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            if ok.sum() < 60 or xs[ok].std() < 1e-12:
                continue
            ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
            if ic <= 0:
                continue
            ws.append(abs(col[ci]))
            contribs.append(lead_boost_all[ci, -1])
        if not ws:
            continue
        ws = np.array(ws)
        ws = ws / ws.sum()
        boost[j] = float((ws * np.array(contribs)).sum())
    return boost


print("=== Idea 1: weighted multi-leader blend (top-3, |corr|-weighted) ===", flush=True)
t0 = time.time()
BOOST_MULTI = np.zeros((nIdio, nt))
for k in range(V10.BOOST_MIN_DAY, nt):
    BOOST_MULTI[:, k] = _pairwise_boost_multi(rs_full[:, :k], k=3)
print(f"  boost computed ({time.time()-t0:.0f}s)", flush=True)

WZ1 = np.zeros((nIdio, nt))
for t in H.days:
    wz = H.WZ_PRE[:, t] + V10.BOOST_K * BOOST_MULTI[:, t]
    WZ1[:, t] = H.rs_blend(wz, t)
RESULTS.append(H.evaluate("idea1_multi_leader_top3", WZ1))


# ====================================================================================================
# IDEA 2 -- peer-consensus broadcast (unconditional top-3-by-|corr| peers, no significance gate)
# ====================================================================================================
def _top3_peers(rs, hi, window=250):
    lo = max(0, hi - window)
    if hi - lo < 30:
        return None
    X = rs[:, lo:hi]
    C = V10._corrmat(X, X)
    np.fill_diagonal(C, np.nan)
    absC = np.where(np.isnan(C), -1.0, np.abs(C))
    return np.argsort(-absC, axis=1)[:, :3]


print("=== Idea 2: peer-consensus broadcast (recompute every 20 days) ===", flush=True)
t0 = time.time()
SIG2 = np.zeros((nIdio, nt))
peer_idx = None
next_recalc = -1
for t in H.days:
    if t >= next_recalc:
        pi = _top3_peers(rs_full, t)
        if pi is not None:
            peer_idx = pi
        next_recalc = t + 20
    if peer_idx is not None:
        retvec = rs_full[:, t - 1]
        SIG2[:, t] = retvec[peer_idx].mean(axis=1)
print(f"  signal computed ({time.time()-t0:.0f}s)", flush=True)

for w in (0.02, 0.05, 0.1, 0.2):
    WZ2 = np.zeros((nIdio, nt))
    for t in H.days:
        WZ2[:, t] = H.blend_signal(H.BASE_WZ[:, t], SIG2[:, t], w)
    RESULTS.append(H.evaluate(f"idea2_peer_consensus_w{w}", WZ2))


# ====================================================================================================
# IDEA 3 -- leader-surprise boost (leader's return minus its own trailing EW(halflife=10) mean)
# ====================================================================================================
print("=== Idea 3: leader-surprise boost ===", flush=True)
t0 = time.time()
EW_HALFLIFE = 10
surprise_full = np.zeros_like(rs_full)
for i in range(nIdio):
    s = pd.Series(rs_full[i])
    ewm_prior = s.ewm(halflife=EW_HALFLIFE, adjust=False).mean().shift(1)   # strictly PRIOR returns only
    surprise_full[i] = (s - ewm_prior).values


def _pairwise_boost_surprise(rs, surprise):
    """Identical leader SELECTION to V10._pairwise_boost (based on raw `rs`); the convex-transform
    INPUT is `surprise` (leader's own return minus its short trailing EW mean) instead of raw return."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V10.BOOST_MIN_DAY:
        return boost
    Xi_full = rs[:, :-1]
    Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
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
        sur = surprise[i]
        scale = np.nanstd(sur[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(sur) * (np.abs(sur) / scale) ** V10.BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]
        ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        v = lead_boost[-1]
        boost[j] = 0.0 if np.isnan(v) else float(v)
    return boost


BOOST_SURPRISE = np.zeros((nIdio, nt))
for k in range(V10.BOOST_MIN_DAY, nt):
    BOOST_SURPRISE[:, k] = _pairwise_boost_surprise(rs_full[:, :k], surprise_full[:, :k])
print(f"  boost computed ({time.time()-t0:.0f}s)", flush=True)

WZ3 = np.zeros((nIdio, nt))
for t in H.days:
    wz = H.WZ_PRE[:, t] + V10.BOOST_K * BOOST_SURPRISE[:, t]
    WZ3[:, t] = H.rs_blend(wz, t)
RESULTS.append(H.evaluate("idea3_leader_surprise", WZ3))


# ====================================================================================================
# IDEA 4 -- cluster momentum (hierarchical clustering, 6 clusters, recompute every 50 days)
# ====================================================================================================
def _cluster_labels(rs, hi, window=250, n_clusters=6):
    lo = max(0, hi - window)
    if hi - lo < 30:
        return None
    X = rs[:, lo:hi]
    C = V10._corrmat(X, X)
    C = np.clip(C, -1.0, 1.0)
    D = 1.0 - C
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return fcluster(Z, t=n_clusters, criterion="maxclust")


print("=== Idea 4: cluster momentum (6 clusters, recompute every 50 days) ===", flush=True)
t0 = time.time()
SIG4 = np.zeros((nIdio, nt))
mate_lists = None
next_recalc = -1
for t in H.days:
    if t >= next_recalc:
        lab = _cluster_labels(rs_full, t)
        if lab is not None:
            mate_lists = []
            for j in range(nIdio):
                m = np.where(lab == lab[j])[0]
                mate_lists.append(m[m != j])
        next_recalc = t + 50
    if mate_lists is not None:
        retvec = rs_full[:, t - 1]
        for j in range(nIdio):
            m = mate_lists[j]
            SIG4[j, t] = retvec[m].mean() if len(m) else 0.0
print(f"  signal computed ({time.time()-t0:.0f}s)", flush=True)

for w in (0.02, 0.05, 0.1, 0.2):
    WZ4 = np.zeros((nIdio, nt))
    for t in H.days:
        WZ4[:, t] = H.blend_signal(H.BASE_WZ[:, t], SIG4[:, t], w)
    RESULTS.append(H.evaluate(f"idea4_cluster_momentum_w{w}", WZ4))


# ====================================================================================================
# summary
# ====================================================================================================
print("\n=== SUMMARY: batch100 category A (peer aggregation) vs shipped v10 "
      f"(OLD={H.BASE_WO:.1f} NEW={H.BASE_WN:.1f} rmean={H.BASE_SCS.mean():.1f} rfloor={H.BASE_SCS.min():.1f}) ===")
passing = [r for r in RESULTS if r["passed"]]
print(f"{len(passing)}/{len(RESULTS)} configs beat v10 on OLD+NEW+rmean jointly.\n")
for r in RESULTS:
    tag = "  <== PASS" if r["passed"] else ""
    print(f"  {r['name']:<32}OLD={r['wo']:7.1f}  NEW={r['wn']:7.1f}  rmean={r['rm']:7.1f}  "
          f"rfloor={r['rf']:7.1f}  n_worse={r['nworse']}/61{tag}")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"\nBest passing config by rmean: {best['name']} (rmean={best['rm']:.1f})")
else:
    print("\nNo configs passed. Rejected -- consistent with this repo's extensive multi-leader/"
          "peer-aggregation track record.")
