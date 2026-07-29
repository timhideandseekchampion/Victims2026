"""
test_batch100_J99_boost_nensemble.py

J99: test_v19cand_boost_ncandidates.py found an isolated single-point spike in rolling-mean at
BOOST_N_CANDIDATES=39, with N=38 and N=40 both worse -- suspicious for a real optimum vs a lucky point.
Direct attempted fix: SMOOTH across the choice by averaging the boost value computed independently at
N=35, 39, 43 (equally spaced around 39), instead of committing to the single isolated spike.

Expensive precompute (WZ_PRE, RS raw signal, ALGO leg) reused verbatim from batch100_common_gi/cache.
BOOST at N=39 is already cached there; only N=35 and N=43 are recomputed here (monkeypatching
V10.BOOST_N_CANDIDATES and calling V10._pairwise_boost directly, per house convention).
"""
import numpy as np, time
from batch100_common_gi import (
    nInst, nt, nIdio, rs, days, WZ_PRE, BOOST, BOOST_MIN_DAY, BOOST_K, RS_RAW, RS_WEIGHT,
    build_pos_from_wz, evaluate, print_sanity, base_wo, base_wn, base_scs,
)
import SAFE_llboost_v10 as V10

SANITY_OK = print_sanity("(J99 boost N-ensemble)")


def compute_boost_N(N):
    orig = V10.BOOST_N_CANDIDATES
    V10.BOOST_N_CANDIDATES = N
    B = np.zeros((nIdio, nt))
    for t in range(BOOST_MIN_DAY, nt):
        B[:, t] = V10._pairwise_boost(rs[:, :t])
    V10.BOOST_N_CANDIDATES = orig
    return B


print("\n=== recompute boost at N=35 and N=43 (N=39 reused from cache) ===")
t0 = time.time()
B35 = compute_boost_N(35)
print(f"  N=35 done ({time.time()-t0:.0f}s)")
t0 = time.time()
B43 = compute_boost_N(43)
print(f"  N=43 done ({time.time()-t0:.0f}s)")

AVG_BOOST = (B35 + BOOST + B43) / 3.0


def rs_blend(wz, t):
    s = RS_RAW[:, t]
    if not np.isfinite(s).all():
        return wz
    sstd = s.std()
    s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
    return (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)


def build_pos_boost(B):
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * B[:, t]
        WZ[:, t] = rs_blend(wz, t)
    return build_pos_from_wz(WZ)


print("\n=== evaluate: individual N=35, N=39(=v10), N=43, and the 3-way average ensemble ===")
r35 = evaluate("N=35 alone", build_pos_boost(B35))
r39 = evaluate("N=39 alone (=v10)", build_pos_boost(BOOST))
r43 = evaluate("N=43 alone", build_pos_boost(B43))
rens = evaluate("ensemble{35,39,43}", build_pos_boost(AVG_BOOST))

print(f"\nSANITY: N=39-alone should match cached v10 baseline exactly "
      f"(OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f}) -> got "
      f"OLD={r39['wo']:.1f} NEW={r39['wn']:.1f} rmean={r39['rm']:.1f}")

print(f"\nEnsemble passed vs v10: {rens['passed']}  (n_worse={rens['nworse']}/61, "
      f"rmean {rens['rm']:.1f} vs v10 {base_scs.mean():.1f})")
