"""
test_batch100_J98_orphans.py

J98 (DIAGNOSTIC): does the pairwise-boost mechanism have structural "orphan" names -- idio names that
essentially never get selected as a boost LEADER (i.e. never in the causal top-BOOST_N_CANDIDATES=39
by-volatility pool, or in the pool but never actually chosen as anyone's best-correlated leader), and/or
never get a significant boost applied to themselves as a FOLLOWER (never find a significant leader) --
and if so, whether those specific names underperform.

Recomputes LEADER_ID (which candidate, if any, was selected as follower j's leader on day t) using
V10._sig_threshold / V10._corrmat directly (identical math to V10._pairwise_boost, matching
batch100_shared.py's B33 precompute pattern) -- not cached anywhere already built, so rebuilt here
directly from the cached `rs` (idio returns) that batch100_common_gi already provides.
"""
import numpy as np, time
from batch100_common_gi import (
    nInst, nt, nIdio, P_, rs, POS_BASE, BOOST_MIN_DAY, BOOST_N_CANDIDATES, print_sanity
)
import SAFE_llboost_v10 as V10

SANITY_OK = print_sanity("(J98 orphans)")

print(f"\n=== recompute LEADER_ID (which candidate was selected as each follower's leader, per day) ===")
t0 = time.time()
LEADER_ID = np.full((nIdio, nt), -1, dtype=int)
CAND_MEMBER = np.zeros((nIdio, nt), dtype=bool)
for t in range(BOOST_MIN_DAY, nt):
    rsl = rs[:, :t]
    n, T = rsl.shape
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    CAND_MEMBER[cand_idx, t] = True
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp):
            col[cp[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V10.BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        LEADER_ID[j, t] = i
print(f"  done ({time.time()-t0:.0f}s)")

nboostdays = nt - BOOST_MIN_DAY
leader_active_days = np.array([(LEADER_ID == i).any(axis=0).sum() for i in range(nIdio)])
leader_total_assign = np.array([(LEADER_ID == i).sum() for i in range(nIdio)])
follower_recipient_days = np.array([(LEADER_ID[i, :] != -1).sum() for i in range(nIdio)])
cand_freq = CAND_MEMBER.sum(axis=1) / nboostdays

print(f"\n=== per-name leader/follower activity across {nboostdays} boost-eligible days ===")
print(f"{'name':>5} {'cand_freq':>10} {'leader_days':>12} {'leader_assigns':>15} {'follower_days':>14}")
for i in range(nIdio):
    print(f"{i:>5} {cand_freq[i]:>10.2%} {leader_active_days[i]:>12} {leader_total_assign[i]:>15} "
          f"{follower_recipient_days[i]:>14}")

ORPHAN_THRESH = 0.01  # <1% of boost days
orphans = [i for i in range(nIdio)
           if leader_active_days[i] <= ORPHAN_THRESH * nboostdays
           and follower_recipient_days[i] <= ORPHAN_THRESH * nboostdays]
non_orphans = [i for i in range(nIdio) if i not in orphans]
print(f"\n=== ORPHANS (leader_active_days AND follower_recipient_days both <{ORPHAN_THRESH:.0%} of "
      f"{nboostdays} days) ===")
print(f"  {len(orphans)}/{nIdio} names: {orphans}")
print(f"  cand_freq of orphans: {[f'{cand_freq[i]:.2%}' for i in orphans]}")

print("\n=== per-name gross daily $ PnL (v10's actual traded position, t in [480, nt-1)), "
      "orphans vs rest ===")
per_name_pnl = np.zeros(nIdio)
for t in range(480, nt - 1):
    pos_t = POS_BASE[1:, t]
    pnl_t = pos_t * (P_[1:, t + 1] - P_[1:, t])
    per_name_pnl += pnl_t
n_days_pnl = (nt - 1) - 480
per_name_avg = per_name_pnl / n_days_pnl

if orphans:
    orphan_avg = per_name_avg[orphans].mean()
    print(f"  orphans      (n={len(orphans):2d}): avg ${orphan_avg:.2f}/name-day  "
          f"(per-name: {[f'{per_name_avg[i]:.2f}' for i in orphans]})")
else:
    print("  orphans: none found")
if non_orphans:
    rest_avg = per_name_avg[non_orphans].mean()
    print(f"  rest         (n={len(non_orphans):2d}): avg ${rest_avg:.2f}/name-day")

print(f"\n  overall avg across all {nIdio} names: ${per_name_avg.mean():.2f}/name-day")
print(f"\nINTERPRETATION: {'orphans found and their avg $/name-day is ' + ('LOWER' if orphans and orphan_avg < rest_avg else 'not clearly lower') + ' than non-orphans -- ' if orphans else 'no structural orphans meeting the <1% threshold on BOTH axes were found -- '}"
      f"{'every name gets boosted at least occasionally, even if via infrequent leader/follower activity.' if not orphans else 'see per-name PnL above for the underperformance comparison.'}")
