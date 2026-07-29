"""
test_batch100_I93.py

I93 (DIAGNOSTIC): stress-test sensitivity of the OLD/NEW/rolling-mean RANKING of v7 through v10 to
the exact commission-rate assumption. Try commRate at 0.5x and 1.5x the shipped 1e-4 (idio) / 2e-5
(ALGO) values, holding every version's ACTUAL positions fixed (commission rate only affects scoring,
not position generation), and check whether the monotonic improvement story (v7 < v8 < v9 < v10 on
OLD, NEW, and rolling mean) survives.
"""
import numpy as np
import batch100_versions_shared as S

nt = S.nt
end_days = S.end_days
NUMTEST = S.NUMTEST
OLD, NEW = S.OLD, S.NEW
ORDER = ["orig", "v7", "v8", "v9", "v10"]


def scs_curve_mult(POS, mult):
    return np.array([S.wscore_commmult(POS, E - NUMTEST, E, mult) for E in end_days])


MULTS = [0.5, 1.0, 1.5]
results = {m: {} for m in MULTS}
for mult in MULTS:
    print(f"\n=== commRate x{mult} (idio {1e-4*mult:.1e}, ALGO {2e-5*mult:.1e}) ===")
    for name in ORDER:
        wo = S.wscore_commmult(S.POS[name], *OLD, mult)
        wn = S.wscore_commmult(S.POS[name], *NEW, mult)
        rm = scs_curve_mult(S.POS[name], mult).mean()
        results[mult][name] = (wo, wn, rm)
        print(f"  {name:<6} OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={rm:7.1f}")

print("\n=== is the ranking v7 < v8 < v9 < v10 preserved at each commRate level? ===")
for mult in MULTS:
    old_seq = [results[mult][n][0] for n in ["v7", "v8", "v9", "v10"]]
    new_seq = [results[mult][n][1] for n in ["v7", "v8", "v9", "v10"]]
    rm_seq = [results[mult][n][2] for n in ["v7", "v8", "v9", "v10"]]
    old_mono = all(old_seq[i] < old_seq[i + 1] for i in range(3))
    new_mono = all(new_seq[i] < new_seq[i + 1] for i in range(3))
    rm_mono = all(rm_seq[i] < rm_seq[i + 1] for i in range(3))
    print(f"  x{mult}: OLD monotonic={old_mono}  NEW monotonic={new_mono}  rmean monotonic={rm_mono}")

print("\n=== v10's margin over v9 (rmean) at each commRate level (does the smallest gap ever flip?) ===")
for mult in MULTS:
    gap = results[mult]["v10"][2] - results[mult]["v9"][2]
    print(f"  x{mult}: v10-v9 rmean gap = {gap:+.1f}")
