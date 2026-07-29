"""
test_batch100_G83.py

QUESTION (G83): instead of committing entirely to SAFE_llboost_v10 (v9 + rank-stability blend),
does a simple model average of v9's and v10's own final wz outputs -- an ensemble of "the whole
book" rather than a single committed model -- do better? v9 and v10 are identical except v10 adds
the RS blend at the very end, so:
  WZ_V9  = WZ_PRE + BOOST_K * BOOST                       (v9's actual traded wz, no RS blend)
  WZ_V10 = rs_blend(WZ_V9)                                 (v10's actual traded wz, cached)
Ensemble: WZ_ens = alpha * WZ_V9 + (1 - alpha) * WZ_V10, then position = sign(WZ_ens) * full size
(matching the shipped sign-only sizing convention). alpha=0.5 is the natural "simple ensemble"
default; alpha=0.3/0.7 included as a light robustness check around it (one obvious free parameter).
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_V9, WZ_V10 = B.WZ_V9, B.WZ_V10

print("\n=== sanity check (idea-specific): alpha=0 must reduce to plain SAFE_llboost_v10 exactly ===")
POS_a0 = B.build_pos_from_wz(0.0 * WZ_V9 + 1.0 * WZ_V10)
r_a0 = B.evaluate("alpha=0 (pure v10)", POS_a0)
idea_sanity_ok = sanity_ok and abs(r_a0["wo"] - B.base_wo) < 0.5 and abs(r_a0["wn"] - B.base_wn) < 0.5
print("  OK." if idea_sanity_ok else "  *** WARNING: alpha=0 does not reproduce v10 -- check logic. ***")

print("\n=== reference: pure v9 (alpha=1) for context ===")
POS_v9 = B.build_pos_from_wz(WZ_V9)
B.evaluate("alpha=1 (pure v9)", POS_v9)

print("\n=== SWEEP: ensemble weight alpha (WZ_ens = alpha*WZ_V9 + (1-alpha)*WZ_V10) ===")
ALPHAS = [0.3, 0.5, 0.7]
results = []
for alpha in ALPHAS:
    WZ_ens = alpha * WZ_V9 + (1 - alpha) * WZ_V10
    POS = B.build_pos_from_wz(WZ_ens)
    results.append(B.evaluate(f"alpha={alpha}", POS))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} ensemble weights beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"Best by rolling mean: {best['name']} rmean={best['rm']:.1f} (v10 rmean={B.base_scs.mean():.1f})")
