"""
test_batch100_B35_confidence_ramp.py

B35: Re-test confidence-ramp / continuous-magnitude position sizing against v10. Rank-stability (v10's
own new mechanism) changes the conviction landscape (|wz| distribution) the sizing scheme would act
on, so re-testing directly rather than citing the old batch80 rejection (README: "sizing/smoothing
schemes uniformly lose ... confidence ramps").

MECHANISM A (confidence-ramp, batch80 item 60): scale = clip(|wz| / ramp, min_scale, 1.0) -- saturating
ramp above the flip point, size grows continuously with conviction instead of full-conviction sign-only.
MECHANISM B (continuous-magnitude / rank-based sizing, batch80 item 59): scale by the cross-sectional
RANK of |wz| that day, bounded 0.5x-1.0x -- a different continuous-magnitude scheme, same spirit.

Both applied on top of v10's ACTUAL final wz (WZ_FULL: ridge ensemble + beta-adjusted target + BLEND
reversion + boost + rank-stability blend, exactly as v10 trades it) -- only the idio SIZING changes,
sign and everything upstream of it is identical to v10.
"""
import numpy as np, time
from scipy.stats import rankdata
from batch100_shared import (
    nInst, nIdio, nt, P_, dlr, days, algo_pos, WZ_FULL, base_wo, base_wn, base_scs, SANITY_OK, evaluate
)

print(f"\n=== B35 sanity check (shared precompute) reproduces v10: {'PASS' if SANITY_OK else 'FAIL'} ===")
print(f"  OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")


def build_confidence_ramp(ramp, min_scale=0.3):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_FULL[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        scale = np.clip(np.abs(wz) / ramp, min_scale, 1.0)
        POS[1:, t] = np.clip(scale * np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


def build_rank_sized(lo=0.5, hi=1.0):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_FULL[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        rank = (rankdata(np.abs(wz)) - 1) / (nIdio - 1)
        scale = lo + (hi - lo) * rank
        POS[1:, t] = np.clip(scale * np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== B35 SWEEP A: confidence-ramp sizing, ramp in {0.1, 0.25, 0.5, 1.0, 2.0}, min_scale=0.3 ===")
t0 = time.time()
results = []
for ramp in (0.1, 0.25, 0.5, 1.0, 2.0):
    Pz = build_confidence_ramp(ramp, 0.3)
    results.append(evaluate(f"confidence-ramp ramp={ramp}", Pz))
print(f"  sweep done ({time.time()-t0:.0f}s)")

print("\n=== B35 SWEEP B: cross-sectional-rank sizing, bounded 0.5x-1.0x ===")
Pz = build_rank_sized(0.5, 1.0)
results.append(evaluate("rank-sized 0.5-1.0x", Pz))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} continuous-magnitude sizing configs beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<28} rmean={c['rm']:>7.1f}  rfloor={c['rf']:>7.1f}  n_worse={c['nworse']}/61")
