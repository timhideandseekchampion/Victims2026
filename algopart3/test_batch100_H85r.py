"""
test_batch100_H85r.py

QUESTION (H85): does a blanket N-day flip cooldown -- once an idio name's held position flips sign,
force holding that new direction for at least N days regardless of what the signal does in between --
beat v10? Distinct from the already-tested per-conviction MAGNITUDE deadband
(test_v20cand_idio_deadband.py): this rule ignores conviction/magnitude entirely and only restricts
the FREQUENCY of sign flips, based purely on elapsed time since the last flip.

MECHANISM: track, per idio name, the day of its last flip and the currently-held sign. Each day t,
compute the shipped v10 raw sign sign(WZ_V10[i,t]); if it disagrees with the held direction AND fewer
than N days have elapsed since the last flip, ignore it (keep holding); otherwise adopt the new sign
and reset the flip-day counter. Sizing at the held sign is otherwise identical to shipped (full
$10k/cur allowance, same cap). N=1 is a no-op (the very next day already satisfies "N days elapsed")
and is used as an idea-specific check on top of the shared baseline sanity check.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V10, days, nIdio, nt, nInst = B.WZ_V10, B.days, B.nIdio, B.nt, B.nInst
algo_pos, dlr, P_ = B.algo_pos, B.dlr, B.P_


def build_pos_cooldown(N):
    POS = np.zeros((nInst, nt))
    held = np.zeros(nIdio)
    last_flip = np.zeros(nIdio)
    first = True
    for t in days:
        wz = WZ_V10[:, t]
        raw_sign = np.sign(wz)
        cur = P_[:, t]
        lim = (dlr[1:] / cur[1:]).astype(int)
        if first:
            held = raw_sign.copy()
            last_flip = np.full(nIdio, float(t))
            first = False
        else:
            flip_now = (raw_sign != held) & (raw_sign != 0)
            allowed = flip_now & ((t - last_flip) >= N)
            last_flip = np.where(allowed, t, last_flip)
            held = np.where(allowed, raw_sign, held)
        POS[1:, t] = np.clip(held * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== idea-specific check: N=1 must reproduce shipped v10 exactly (no-op cooldown) ===")
POS_n1 = build_pos_cooldown(1)
r1 = B.evaluate("N=1 (no-op)", POS_n1)
match = abs(r1["wo"] - B.base_wo) < 1e-6 and abs(r1["wn"] - B.base_wn) < 1e-6
print("  OK -- exact match to baseline." if match else "  *** WARNING: N=1 does NOT match baseline exactly ***")

print("\n=== H85: blanket N-day flip cooldown (idio only), N sweep ===")
results = [r1]
for N in [3, 5, 10, 20, 40]:
    POS = build_pos_cooldown(N)
    results.append(B.evaluate(f"N={N}", POS))

cand_results = results[1:]
passing = [r for r in cand_results if r["passed"]]
print(f"\n{len(passing)}/{len(cand_results)} non-trivial configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for r in passing:
        print(f"  {r['name']:<28} rmean={r['rm']:.1f} n_worse={r['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for r in sorted(cand_results, key=lambda r: -r["rm"]):
        print(f"  {r['name']:<28} OLD={r['wo']:>7.1f} NEW={r['wn']:>7.1f} rmean={r['rm']:>7.1f} "
              f"rfloor={r['rf']:>7.1f} n_worse={r['nworse']}/61")
