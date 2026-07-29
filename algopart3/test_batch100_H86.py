"""
test_batch100_H86.py

QUESTION (H86): idio positions are ALWAYS sized at the full $10k/cur allowance (sign-only sizing,
never partial) -- so pos[i,t] = sign(wz_i,t) * trunc(dlr_i / cur_i(t)), truncated toward zero every
single day. Since cur_i(t) drifts continuously with price, trunc(dlr_i/cur_i(t)) changes by +-1 share
on many days purely from price movement, EVEN WHEN THE SIGN DOES NOT CHANGE -- pure rounding churn
that pays commission for no signal reason. Does commission-aware rounding (round-to-nearest instead of
truncate, and/or a small same-sign share-count deadband that ignores +-k share drift) reduce this
churn enough to matter for score?

MECHANISM tested (idio names only -- ALGO's leg has its own separate, already-causal/stateful sizing
logic and is left untouched here, out of scope for a screening-level test):
  ROUND: shares = sign(wz) * round(dlr/cur)              (vs shipped trunc)
  DEADBAND(k): shares = sign(wz) * round(dlr/cur), but if sign is unchanged from yesterday and the
               new rounded magnitude differs from yesterday's held magnitude by <= k shares, keep
               yesterday's magnitude instead (only sign flips, or magnitude moves > k, actually update)
DEADBAND(0) reduces to plain ROUND (the "always accept the new magnitude" case) and is used as an
idea-specific sanity/continuity check.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V10, days, nIdio, nt, nInst = B.WZ_V10, B.days, B.nIdio, B.nt, B.nInst
P_, dlr, algo_pos = B.P_, B.dlr, B.algo_pos


def build_pos_deadband(k):
    POS = np.zeros((nInst, nt))
    prev_sign = np.zeros(nIdio)
    prev_mag = np.zeros(nIdio)
    for t in days:
        wz = WZ_V10[:, t]
        sign = np.sign(wz)
        cur = P_[1:, t]
        cap_mag = (dlr[1:] / cur).astype(int)  # integer cap, matches shipped trunc cap
        raw_mag = np.clip(np.round(dlr[1:] / cur), 0, cap_mag)
        same_sign = (sign == prev_sign) & (sign != 0)
        small_move = np.abs(raw_mag - prev_mag) <= k
        keep = same_sign & small_move
        new_mag = np.where(keep, prev_mag, raw_mag)
        POS[1:, t] = sign * new_mag
        prev_sign, prev_mag = sign, new_mag
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check (idea-specific): DEADBAND(k=0) = plain round-to-nearest, should be close to "
      "(not bit-identical to, since shipped truncates rather than rounds) v10 ===")
POS_k0 = build_pos_deadband(0)
r0 = B.evaluate("DEADBAND(k=0)=ROUND", POS_k0)

print("\n=== how much commission is spent on same-sign magnitude drift (pure rounding churn) vs on "
      "actual sign flips, in the SHIPPED v10 position path? ===")
POS_base = B.POS_BASE
comm_flip, comm_drift = 0.0, 0.0
prev = POS_base[1:, days[0]]
for t in days[1:]:
    cur_pos = POS_base[1:, t]
    dP = cur_pos - prev
    cur_px = P_[1:, t]
    comm = B.commRate[1:] * np.abs(dP) * cur_px
    flipped = np.sign(cur_pos) != np.sign(prev)
    comm_flip += comm[flipped & (prev != 0) & (cur_pos != 0)].sum()
    comm_drift += comm[~flipped].sum()
    prev = cur_pos
print(f"  total idio commission (shipped v10, day>=96): sign-flip days=${comm_flip:.0f}  "
      f"same-sign-drift days=${comm_drift:.0f}  (drift fraction={comm_drift/(comm_flip+comm_drift):.2%})")

print("\n=== SWEEP: same-sign share-count deadband k (shares) ===")
KS = [1, 2, 3, 5]
results = [r0] + [B.evaluate(f"DEADBAND(k={k})", build_pos_deadband(k)) for k in KS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"Best by rolling mean: {best['name']} rmean={best['rm']:.1f} (v10 rmean={B.base_scs.mean():.1f})")
