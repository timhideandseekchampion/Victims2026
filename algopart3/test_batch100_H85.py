"""
test_batch100_H85.py

QUESTION (H85): does a blanket N-day flip cooldown -- once an idio name's position flips sign, force
holding that new direction for at least N days regardless of what the signal does in between -- beat
v10? Distinct from the per-conviction magnitude deadband (test_v20cand_idio_deadband.py): this rule
ignores conviction/magnitude entirely and only restricts the FREQUENCY of sign flips.

MECHANISM: track, per idio name, the day of its last flip. On each day t, compute the shipped v10 raw
sign (sign(WZ_V10[i,t])); if it disagrees with the currently-held direction AND fewer than N days have
elapsed since the last flip, ignore it (keep holding); otherwise adopt the new sign and reset the
flip-day counter. N=1 is a no-op (trivially always >=1 day since any last flip) and is used as an
extra idea-specific sanity check.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V10, days, nIdio, nt, nInst = B.WZ_V10, B.days, B.nIdio, B.nt, B.nInst
P_, dlr, algo_pos = B.P_, B.dlr, B.algo_pos


def build_pos_cooldown(N):
    POS = np.zeros((nInst, nt))
    prev_dir = np.zeros(nIdio)
    last_flip_day = np.full(nIdio, -10 ** 9)
    for t in days:
        wz = WZ_V10[:, t]
        raw_sign = np.sign(wz)
        can_flip = (t - last_flip_day) >= N
        want_flip = (raw_sign != 0) & (raw_sign != prev_dir)
        do_flip = want_flip & can_flip
        new_dir = np.where(do_flip, raw_sign, prev_dir)
        last_flip_day = np.where(do_flip, t, last_flip_day)
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(new_dir * (dlr[1:] / cur[1:]), -lim, lim)
        prev_dir = new_dir
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check (idea-specific): N=1 (no real restriction) must reproduce v10 exactly ===")
POS1 = build_pos_cooldown(1)
r1 = B.evaluate("N=1 (no-op)", POS1)
idea_sanity_ok = sanity_ok and abs(r1["wo"] - B.base_wo) < 0.5 and abs(r1["wn"] - B.base_wn) < 0.5
print("  OK." if idea_sanity_ok else "  *** WARNING: N=1 does not reproduce v10 -- check logic. ***")

print("\n=== SWEEP: cooldown length N (trading days) ===")
NS = [2, 3, 5, 10, 20]
results = [B.evaluate(f"N={n}", build_pos_cooldown(n)) for n in NS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} cooldown lengths beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"Best by rolling mean: {best['name']} rmean={best['rm']:.1f} (v10 rmean={B.base_scs.mean():.1f})")
