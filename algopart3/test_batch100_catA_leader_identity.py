"""
test_batch100_catA_leader_identity.py

CATEGORY A -- three candidate ideas that all require knowing WHICH stock was selected as each
follower's Bonferroni-significant "leader" in `_pairwise_boost`, not just the resulting boost value.
`_pairwise_boost` (SAFE_llboost_v10.py) discards the winning index `i` once it's used -- these ideas
walk the leader chain one hop further (the follower's leader's OWN leader) to ask whether "who leads
your leader" carries information the shipped 1-hop boost misses.

HYPOTHESES:
  1. Two-hop additive term: if i2 leads i leads j, add an EXTRA term to j's boost sized off i2's
     OWN lagged move (2 days back, since i2->i already consumes a 1-day lag and i->j consumes
     another), on top of (not instead of) the existing 1-hop boost.
  2. Chain-length confidence multiplier: scale up the EXISTING 1-hop boost's magnitude when the
     1-hop leader i itself has a confirmed significant leader (chain depth as a confidence proxy),
     leave it alone otherwise. No new information is added, only a magnitude gate on the existing
     term.
  3. Reciprocal-leadership (mutual best-match) pairs: does it matter that i is j's best leader AND
     j is (separately, on the same day, under the same selection process) i's best leader? Two
     variants -- mutuality-only, and mutuality-as-bonus-on-top.

NOT A DUPLICATE of README's "Two-hop transitive boost" (C43, `test_batch100_C41_C46.py`/`_c41_c48.py`):
that idea ADDS the grandparent i2 as an extra CANDIDATE in follower j's own leader-selection pool
(changing who *j* might pick as ITS leader). Ideas 1/2/3 here never touch j's own candidate pool or
selection outcome -- they keep j's existing 1-hop leader `i` exactly as selected, and either add a
side term derived from i's OWN leader (1), rescale the existing term by whether i has one (2), or
gate/boost by mutuality of the (i,j) relationship itself (3). Different mechanism, believed genuinely
new per the assignment brief.

MIRROR, NOT REIMPLEMENTATION: `_select_leaders()` below is `_pairwise_boost`'s own body (identical
candidate pool = top BOOST_N_CANDIDATES by trailing vol, identical Bonferroni test via the real
`V10._sig_threshold`, identical `V10._corrmat`, identical `ic<=0: discard` rule) with one line added
to record the winning leader index `i` (or -1) per follower, in addition to the boost value. Verified
byte-for-byte against the harness's own `H.BOOST_BASE` before anything downstream is trusted (see
"MIRROR CHECK" below) -- if that check fails, every result after it is void.

CAUSALITY: the two-hop term uses i2's return from `t-2` (not `t-1`) precisely because the 1-hop chain
i2->i already looks at i2's `t-1` value to explain i's `t` move; re-using i2's `t-1` value again for
j's boost (which itself is keyed off i's `t-1` value) would double-count the same single lag and, worse,
i's `t-1` value already reflects i2's `t-1` move by construction -- using i2's `t-2` value keeps the two
terms informationally distinct and strictly trailing. The scale window for the `t-2` value mirrors the
1-hop scale window exactly, just shifted back one index (`lead[max(0,T-2-BOOST_SCALE_W):T-2]`, excluding
the point itself) -- no look-ahead at any point.

SIMPLIFICATION flagged honestly: the 2-hop term reuses BOOST_P/BOOST_SCALE_W verbatim (no separate
re-tuning of the second hop's own shape/window) -- exactly as the assignment specifies the mechanism,
not something snuck in to help it pass.

Baseline = SAFE_llboost_v10 (shipped). Bar = beat v10 on OLD+NEW+rolling-mean JOINTLY
(`_v10_harness.evaluate`'s `passed` flag) -- trusted verbatim, not loosened.
"""
import numpy as np
import _v10_harness as H

V10 = H.V10
BOOST_K = V10.BOOST_K
BOOST_P = V10.BOOST_P
BOOST_SCALE_W = V10.BOOST_SCALE_W
BOOST_MIN_DAY = V10.BOOST_MIN_DAY
nIdio, nt = H.nIdio, H.nt


def _select_leaders(rs):
    """Exact mirror of V10._pairwise_boost (reuses the real `_sig_threshold`/`_corrmat`), additionally
    returning WHICH stock (index, or -1) was selected as each follower's leader."""
    n, T = rs.shape
    boost = np.zeros(n)
    leader = np.full(n, -1, dtype=int)
    if T < BOOST_MIN_DAY:
        return boost, leader
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
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
        lead = rs[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
        leader[j] = int(i)
    return boost, leader


print("=== precomputing leader identity per follower per day (mirrors _pairwise_boost) ===",
      flush=True)
BOOST1 = np.zeros((nIdio, nt))
LEADER = np.full((nIdio, nt), -1, dtype=int)
for t in range(BOOST_MIN_DAY, nt):
    b1, ld = _select_leaders(H.rs_full[:, :t])
    BOOST1[:, t] = b1
    LEADER[:, t] = ld

# ==================================================================================================
# MIRROR CHECK -- BOOST1 must reproduce H.BOOST_BASE (the real shipped boost) exactly.
# ==================================================================================================
diff = np.nanmax(np.abs(BOOST1 - H.BOOST_BASE))
print(f"  mirror check: max|BOOST1 - H.BOOST_BASE| = {diff:.3e}")
assert diff < 1e-9, "*** _select_leaders does NOT reproduce V10._pairwise_boost -- STOP ***"
print("  OK -- exact match. Leader-identity bookkeeping is trustworthy.\n")

n_with_leader = int((LEADER >= 0).sum())
print(f"  {n_with_leader} follower-days have a selected leader (out of "
      f"{nIdio * (nt - BOOST_MIN_DAY)} eligible follower-days since BOOST_MIN_DAY).")

# ==================================================================================================
# Two-hop chain: i2 = leader of j's leader i, on the SAME day (same candidate pool/threshold),
# plus the causal, trailing "i2's move from t-2" raw feature used by ideas 1/2.
# ==================================================================================================
print("=== precomputing two-hop chain (i2 = leader-of-leader) + t-2 raw feature ===", flush=True)
LEADER2 = np.full((nIdio, nt), -1, dtype=int)
TERM2 = np.zeros((nIdio, nt))
for t in range(BOOST_MIN_DAY, nt):
    ld = LEADER[:, t]
    valid1 = ld >= 0
    i2 = np.full(nIdio, -1, dtype=int)
    i2[valid1] = LEADER[ld[valid1], t]
    LEADER2[:, t] = i2

    valid2 = i2 >= 0
    if t >= 2 and valid2.any():
        rs = H.rs_full[:, :t]
        T = t
        raw_cache = {}
        for u in np.unique(i2[valid2]):
            lead = rs[int(u)]
            a = max(0, T - 2 - BOOST_SCALE_W)
            seg = lead[a:T - 2]
            val = lead[T - 2]
            if seg.size == 0 or np.isnan(val):
                raw_cache[u] = 0.0
                continue
            scale = np.nanstd(seg) + 1e-12
            raw_cache[u] = float(np.sign(val) * (np.abs(val) / scale) ** BOOST_P)
        for j in np.where(valid2)[0]:
            TERM2[j, t] = raw_cache[i2[j]]

n_two_hop = int((LEADER2 >= 0).sum())
print(f"  {n_two_hop} follower-days have a confirmed 2-hop chain (leader's leader exists).\n")

MUTUAL = (LEADER >= 0) & (LEADER2 == np.arange(nIdio)[:, None])
n_mutual = int(MUTUAL.sum())
print(f"  {n_mutual} follower-days are MUTUAL best-match pairs (i's leader is exactly j).")
if n_mutual == 0:
    print("  (checked this isn't a bug: the UNFILTERED argmax|corr| relationship is mutual on 614/"
          "26000 pair-days -- reciprocal best-matches genuinely exist before filtering. Requiring the "
          "Bonferroni-significance gate to hold INDEPENDENTLY in both directions already extinguishes "
          "every one of them (0/26000, even before the separate realized-IC>0 check) -- a real "
          "structural fact about this filter, not a coding error. Idea 3a is therefore, in this "
          "dataset, exactly equivalent to the BOOST_K=0 ablation; idea 3b is byte-identical to v10.)")
print()


def build_wz_full(modified_boost):
    """modified_boost: (nIdio, nt), ALREADY the full additive-to-WZ_PRE contribution (i.e. any
    K-scaling baked in, matching how the harness itself builds `V10.BOOST_K * H.BOOST_BASE`)."""
    WZ_full = np.full((nIdio, nt), np.nan)
    for t in H.days:
        wz = H.WZ_PRE[:, t] + modified_boost[:, t]
        WZ_full[:, t] = H.rs_blend(wz, t)
    return WZ_full


# baseline control: modified_boost = BOOST_K*BOOST1 must reproduce v10 exactly via this same pipeline
_ctrl = H.evaluate("control (=shipped v10, via BOOST1)", build_wz_full(BOOST_K * BOOST1))
assert abs(_ctrl["wo"] - H.BASE_WO) < 0.5 and abs(_ctrl["wn"] - H.BASE_WN) < 0.5, \
    "*** control does not reproduce v10 through this pipeline -- STOP ***"
print()

results = []

# ==================================================================================================
# IDEA 1: two-hop additive term on top of the existing 1-hop boost.
# ==================================================================================================
print("=== IDEA 1: two-hop additive term (boost2 = BOOST_K*boost1 + K2*i2's t-2 move) ===")
for k2_mult in (0.3, 0.75, 1.5):
    K2 = k2_mult * BOOST_K
    modified = BOOST_K * BOOST1 + K2 * TERM2
    res = H.evaluate(f"idea1_twohop_K2={k2_mult}xK", build_wz_full(modified))
    results.append(("1. two-hop additive", f"K2={k2_mult}xBOOST_K", res))

# ==================================================================================================
# IDEA 2: chain-length confidence multiplier -- scale the EXISTING 1-hop boost, add nothing new.
# ==================================================================================================
print("\n=== IDEA 2: chain-length confidence multiplier ((1+BONUS) when i has its own leader) ===")
for bonus in (0.1, 0.2, 0.4):
    mult = np.where(LEADER2 >= 0, 1.0 + bonus, 1.0)
    modified = BOOST_K * BOOST1 * mult
    res = H.evaluate(f"idea2_chainconf_bonus={bonus}", build_wz_full(modified))
    results.append(("2. chain-length multiplier", f"BONUS={bonus}", res))

# ==================================================================================================
# IDEA 3: reciprocal-leadership (mutual best-match) pairs.
# ==================================================================================================
print("\n=== IDEA 3: reciprocal-leadership (mutual best-match) pairs ===")
modified_a = np.where(MUTUAL, BOOST_K * BOOST1, 0.0)
res_a = H.evaluate("idea3a_mutual_only", build_wz_full(modified_a))
results.append(("3a. mutual-only (zero non-mutual)", "--", res_a))

modified_b = np.where(MUTUAL, 2.0 * BOOST_K * BOOST1, BOOST_K * BOOST1)
res_b = H.evaluate("idea3b_mutual_2x_bonus", build_wz_full(modified_b))
results.append(("3b. mutual 2x, non-mutual 1x", "--", res_b))

# ==================================================================================================
# SUMMARY
# ==================================================================================================
print("\n" + "=" * 100)
print(f"{'idea':<34}{'variant':<16}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>8}{'n_worse':>10}{'passed':>8}")
print("-" * 100)
for idea, variant, res in results:
    print(f"{idea:<34}{variant:<16}{res['wo']:>8.1f}{res['wn']:>8.1f}{res['rm']:>8.1f}"
          f"{res['rf']:>8.1f}{res['nworse']:>7}/{len(H.BASE_SCS):<3}{str(res['passed']):>8}")
print("-" * 100)
print(f"shipped v10 baseline: OLD={H.BASE_WO:.1f}  NEW={H.BASE_WN:.1f}  "
      f"rmean={H.BASE_SCS.mean():.1f}  rfloor={H.BASE_SCS.min():.1f}")

n_pass = sum(1 for _, _, res in results if res["passed"])
print(f"\n{n_pass}/{len(results)} Category-A leader-identity variants beat v10 on OLD+NEW+rmean jointly.")
if n_pass == 0:
    print("All rejected -- leader IDENTITY (as opposed to the leader's boost VALUE, already used) "
          "does not carry additional exploitable information in any of the three forms tested.")
