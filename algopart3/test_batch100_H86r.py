"""
test_batch100_H86r.py  (DIAGNOSTIC)

QUESTION (H86): idio positions are always sign-sized at the full $10k/cur allowance, TRUNCATED toward
zero every day: pos[i,t] = sign(wz_i,t) * floor(dlr_i/cur_i(t)) (via clip to lim=int(dlr/cur), the
shipped convention). Since cur_i(t) drifts continuously, floor(dlr_i/cur_i(t)) can change by +-1 share
on many days purely from price movement, even when the sign does not change -- rounding churn that
pays commission for no signal reason. Does commission-aware rounding (round-to-nearest instead of
truncate, and/or a small same-sign share-count deadband ignoring +-k share drift) change realized
commission or score MATERIALLY? This is a magnitude-of-impact question, not a beat/lose-v10 candidate
test in the usual sense -- reported as a diagnostic, though the same pass-bar numbers are computed
too since they cost nothing extra.

MECHANISM tested (idio names only -- ALGO's leg has its own separate causal/stateful sizing logic,
out of scope for a screening-level test):
  ROUND:        shares = sign(wz) * round(dlr/cur)                    (vs shipped truncate)
  DEADBAND(k):  shares = sign(wz) * round(dlr/cur), but if sign is unchanged from yesterday AND the
                new rounded magnitude differs from yesterday's held magnitude by <= k shares, keep
                yesterday's magnitude (only sign flips, or magnitude moves > k, actually update)
DEADBAND(0) reduces to plain ROUND exactly (used as an idea-specific continuity check).
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V10, days, nIdio, nt, nInst = B.WZ_V10, B.days, B.nIdio, B.nt, B.nInst
algo_pos, dlr, P_, commRate = B.algo_pos, B.dlr, B.P_, B.commRate


def build_pos_round():
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]
        POS[1:, t] = np.sign(wz) * np.round(dlr[1:] / cur[1:])
    POS[0, :] = algo_pos
    return POS


def build_pos_deadband(k):
    POS = np.zeros((nInst, nt))
    held_sign = np.zeros(nIdio); held_mag = np.zeros(nIdio)
    first = True
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]
        sgn = np.sign(wz)
        newmag = np.round(dlr[1:] / cur[1:])
        if first:
            held_sign = sgn.copy(); held_mag = newmag.copy(); first = False
        else:
            keep = (sgn == held_sign) & (np.abs(newmag - held_mag) <= k)
            held_mag = np.where(keep, held_mag, newmag)
            held_sign = sgn
        POS[1:, t] = held_sign * held_mag
    POS[0, :] = algo_pos
    return POS


def total_comm_idio(POS):
    """Rough total $ commission on the idio book across the whole `days` range (not windowed like
    wscore -- just for relative comparison across rounding schemes)."""
    sub = POS[1:, days[0]:days[-1] + 1]
    pcur = P_[1:, days[0]:days[-1] + 1]
    dP = np.diff(sub, axis=1, prepend=sub[:, :1])
    comm = (commRate[1:, None] * np.abs(dP) * pcur).sum()
    return float(comm)


print("\n=== idea-specific check: DEADBAND(0) must equal plain ROUND exactly ===")
POS_round = build_pos_round()
POS_db0 = build_pos_deadband(0)
match = np.allclose(POS_round, POS_db0)
print("  OK -- exact match." if match else "  *** WARNING: DEADBAND(0) != ROUND ***")

print("\n=== H86: commission-aware rounding, score impact (pass-bar numbers, for reference) ===")
results = []
results.append(B.evaluate("ROUND", POS_round))
for k in [1, 2, 3]:
    results.append(B.evaluate(f"DEADBAND(k={k})", build_pos_deadband(k)))

passing = [r for r in results if r["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly (informational only -- diagnostic).")

print("\n=== H86: commission totals ($, idio book only, rough full-history sum) ===")
c_base = total_comm_idio(B.POS_BASE)
c_round = total_comm_idio(POS_round)
c_db1 = total_comm_idio(build_pos_deadband(1))
c_db2 = total_comm_idio(build_pos_deadband(2))
print(f"  baseline (shipped truncate): ${c_base:,.0f}")
print(f"  ROUND:                       ${c_round:,.0f}  ({100*(c_round/c_base-1):+.1f}% vs baseline)")
print(f"  DEADBAND(k=1):               ${c_db1:,.0f}  ({100*(c_db1/c_base-1):+.1f}% vs baseline)")
print(f"  DEADBAND(k=2):               ${c_db2:,.0f}  ({100*(c_db2/c_base-1):+.1f}% vs baseline)")
