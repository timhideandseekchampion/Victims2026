"""
test_batch100_B33_leader_stability.py

B33: Re-test the leader-identity-stability gate (H3) against v10, min_stab swept 10-200 days.

Same mechanism as test_h3_leader_stability.py / test_v14cand_leader_stability.py ("H3
leader-identity-stability gate"), already rejected twice: once against the original SAFE_llboost
baseline (README: "H3, re-confirmed" -- every min_stab 10-200 worsened rmean monotonically, floor
never moved), and once against SAFE_llboost_v9 (test_v14cand_leader_stability.py, 0/21 configs
passed). Re-running against the CURRENT best (SAFE_llboost_v10) since rank-stability (v10's own new
mechanism) changes the wz landscape the boost sits on top of -- not because there's a mechanistic
reason to expect a different outcome for the boost's own leader-persistence gate itself.

MECHANISM (identical to prior runs): for each follower j, track which of the 39 candidates was
selected as its leader each day. Only trust today's boost if the SAME leader was ALSO j's identified
leader on >= frac_req of the trailing min_stab days -- otherwise HARD-gate (discard, shrink=0) the
boost for that name-day. min_stab swept 10-200 (dense, matching the original range), frac_req=0.7
fixed (the original's headline threshold).

Reuses V10._pairwise_boost's exact math (via batch100_shared's BOOST/LEADER_ID, precomputed once) and
V10._beta_adjusted_target / _ewls_ridge / _rank_stability_signal / _algo_vol_shares verbatim -- only
the boost-trust mask is new.
"""
import numpy as np, time
from batch100_shared import (
    nIdio, nt, BOOST_MIN_DAY, BOOST_K, WZ_PRE, BOOST, LEADER_ID, rs_blend, days,
    build_pos_from_wz, base_wo, base_wn, base_scs, SANITY_OK, evaluate, POS_BASE
)

print(f"\n=== B33 sanity check (shared precompute) reproduces v10: {'PASS' if SANITY_OK else 'FAIL'} ===")
print(f"  OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")


def stability_mask(stab_w, frac_req):
    trusted = np.zeros((nIdio, nt), dtype=bool)
    for t in range(BOOST_MIN_DAY, nt):
        lo = max(BOOST_MIN_DAY, t - stab_w)
        if t - lo < max(10, stab_w // 4):
            continue
        today = LEADER_ID[:, t]
        hist = LEADER_ID[:, lo:t]
        match_frac = (hist == today[:, None]).mean(1)
        trusted[:, t] = (today != -1) & (match_frac >= frac_req)
    return trusted


def build_pos_stab(stab_w, frac_req=0.7, shrink=0.0):
    trusted = stability_mask(stab_w, frac_req)
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            b = BOOST[:, t]
            b = np.where(trusted[:, t], b, b * shrink)
            wz = wz + BOOST_K * b
        WZ[:, t] = rs_blend(wz, t)
    return build_pos_from_wz(WZ)


print("\n=== B33 SWEEP: HARD GATE, min_stab in {10,20,40,60,100,150,200}, frac_req=0.7 ===")
results = []
t0 = time.time()
for stab_w in (10, 20, 40, 60, 100, 150, 200):
    Pz = build_pos_stab(stab_w, 0.7, 0.0)
    results.append(evaluate(f"min_stab={stab_w}", Pz))
print(f"  sweep done ({time.time()-t0:.0f}s)")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} min_stab thresholds beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<16} rmean={c['rm']:>7.1f}  rfloor={c['rf']:>7.1f}  n_worse={c['nworse']}/61")

best = max(results, key=lambda c: c["rm"])
print(f"\nBest by rolling mean: {best['name']} (rmean={best['rm']:.1f} vs v10 rmean={base_scs.mean():.1f})")
