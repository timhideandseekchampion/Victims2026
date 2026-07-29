"""
test_batch100_E65_slowic.py

E65: add a THIRD, slow (500-day half-life) IC estimator to the existing ALGO double-IC gate.

The ALGO leg's _side() sub-routine (inside V10._algo_vol_shares) computes a fast raw-window IC
(icf, over IC_FAST=90 days) and an ensemble ("blend") IC ice = mean of exponentially-weighted ICs at
IC_EW_HL=(20, 45) half-lives (window IC_EW_W=200) -- the sign of ice must AGREE with the sign of icf or
the day's signal is zeroed (the "double-IC gate": fast estimator vs a 2-half-life EW blend estimator).
This tests whether adding IC_EW_HL=(20, 45, 500) -- a slow, 500-day-half-life estimator alongside the
existing two -- to the ice ensemble changes the gate's behavior for the better.

Mechanism only touches IC_EW_HL, which V10._algo_vol_shares reads as a plain module GLOBAL each call
(not a captured default), so this is tested by monkeypatching V10.IC_EW_HL for the duration of the
recompute (module state _PREV_ALGO_SHARES/_PREV_T reset first so the day-by-day walk starts clean,
matching the house convention's causal, from-scratch position build). The idio side of the book is
totally unaffected by this idea (ALGO-only mechanism) -- reused verbatim from the shared precompute
(batch100_d6x_shared.py) unchanged.
"""
import numpy as np, time
import SAFE_llboost_v10 as V10
import batch100_d6x_shared as SH

logp, nt, dlr, P_ = SH.logp, SH.nt, SH.dlr, SH.P_
score, wscore, scs_curve, OLD, NEW = SH.score, SH.wscore, SH.scs_curve, SH.OLD, SH.NEW

print(f"\nSANITY_CHECK_PASSED (shared baseline, idio side identical for this idea) = {SH.SANITY_OK}")


def compute_algo(ic_ew_hl):
    V10._PREV_ALGO_SHARES = 0
    V10._PREV_T = -1
    orig = V10.IC_EW_HL
    V10.IC_EW_HL = ic_ew_hl
    try:
        algo = np.zeros(nt)
        for k in range(130, nt):
            cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
            algo[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
    finally:
        V10.IC_EW_HL = orig
    return algo


def build_pos_with_algo(algo):
    POS = SH.POS_BASE.copy()
    POS[0, :] = algo
    return POS


print("\n=== sanity check: recompute ALGO leg with IC_EW_HL=(20,45) (unchanged) -- must exactly match "
      "the shared-cache algo_pos array AND reproduce SAFE_llboost_v10's official numbers ===")
t0 = time.time()
algo_base = compute_algo(V10.IC_EW_HL)  # V10.IC_EW_HL is (20, 45) at import time, unmodified
exact_match = np.array_equal(algo_base, SH.algo_pos)
print(f"  array-identical to shared algo_pos: {exact_match}  [{time.time()-t0:.0f}s]")
POS_base2 = build_pos_with_algo(algo_base)
scs_base2 = scs_curve(POS_base2)
wo0, wn0 = wscore(POS_base2, *OLD), wscore(POS_base2, *NEW)
print(f"  OLD={wo0:.1f}  NEW={wn0:.1f}  rmean={scs_base2.mean():.1f}  rfloor={scs_base2.min():.1f}   "
      f"(v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = exact_match and abs(wo0 - 871.0) < 0.5 and abs(wn0 - 912.6) < 0.5 and SH.SANITY_OK
print("  OK -- matches v10 exactly." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


def evaluate(nm, ic_ew_hl):
    t0 = time.time()
    algo = compute_algo(ic_ew_hl)
    Pz = build_pos_with_algo(algo)
    scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > wo0) and (wn > wn0) and (scs.mean() > scs_base2.mean())
    nworse = int((scs < scs_base2).sum())
    tag = "  <== PASS" if passed else ""
    print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={nworse}/{len(scs)}{tag}  [{time.time()-t0:.0f}s]")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== CANDIDATE: IC_EW_HL=(20, 45, 500) -- add a slow 500-day-half-life estimator to the ice "
      "ensemble ===")
results = [evaluate("IC_EW_HL=(20,45,500)", (20, 45, 500))]

print("\n=== robustness: a couple of other slow half-lives, for context ===")
for hl_slow in (250, 1000):
    results.append(evaluate(f"IC_EW_HL=(20,45,{hl_slow})", (20, 45, hl_slow)))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
          f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
