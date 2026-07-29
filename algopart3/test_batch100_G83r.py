"""
test_batch100_G83r.py

QUESTION (G83): instead of committing entirely to SAFE_llboost_v10 (v9 + rank-stability blend), does
a simple model average of v9's and v10's own final wz outputs -- an ensemble of the whole book rather
than a single committed model -- do better?

v9 and v10 are identical except v10 adds the RS blend at the very end:
  WZ_V9  = WZ_PRE + BOOST_K * BOOST         (v9's actual traded wz, no RS blend)
  WZ_V10 = rs_blend(WZ_V9)                  (v10's actual traded wz, cached)
Ensemble: WZ_ens = alpha*WZ_V9 + (1-alpha)*WZ_V10, then position = sign(WZ_ens) * full size (matching
the shipped sign-only sizing convention). alpha=0.5 is the natural default; 0.3/0.7 are a light
robustness check. alpha=0 exactly reproduces WZ_V10 (=shipped v10, sanity-equivalent); alpha=1 would
reproduce v9 exactly (not itself claimed a pass/fail case here, just useful context).
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V9, WZ_V10 = B.WZ_V9, B.WZ_V10


def build_pos_ens(alpha):
    WZ = alpha * WZ_V9 + (1 - alpha) * WZ_V10
    return B.build_pos_from_wz(WZ)


print("\n=== G83: model average of v9's and v10's final wz (sign-sized) ===")
results = []
for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
    tag = f"alpha(v9 weight)={alpha}"
    POS = build_pos_ens(alpha)
    results.append(B.evaluate(tag, POS))

passing = [r for r in results if r["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for r in passing:
        print(f"  {r['name']:<28} rmean={r['rm']:.1f} n_worse={r['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for r in sorted(results, key=lambda r: -r["rm"]):
        print(f"  {r['name']:<28} OLD={r['wo']:>7.1f} NEW={r['wn']:>7.1f} rmean={r['rm']:>7.1f} "
              f"rfloor={r['rf']:>7.1f} n_worse={r['nworse']}/61")
