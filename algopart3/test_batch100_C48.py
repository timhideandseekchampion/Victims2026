"""
test_batch100_C48.py

C48: use a 2-day trailing AVERAGE of the leader's return (instead of the single most recent day) as
the boost input. Leader SELECTION (argmax|corr| over the causal-vol-ranked BOOST_N_CANDIDATES=39
pool, gated by the same Bonferroni-corrected significance threshold V10._sig_threshold uses) is
unchanged -- only the series fed into the scale calc, the historical IC-gate regression, and the
final applied value is replaced with a 2-day trailing average of the selected leader's return,
applied CONSISTENTLY through all three uses (matching V10._pairwise_boost's own structure exactly,
otherwise).

Everything else (idio ridge ensemble + beta-adjusted target, BLEND reversion, rank-stability blend,
ALGO leg) is reused VERBATIM from V10 via batch100_shared's cached precompute -- unaffected by this
idea, which only touches the boost's leader-return INPUT.
"""
import time
import numpy as np
import SAFE_llboost_v10 as V10
import batch100_shared as S

nInst, nt, nIdio = S.nInst, S.nt, S.nIdio
rs = S.rs
BOOST_MIN_DAY, BOOST_K = S.BOOST_MIN_DAY, S.BOOST_K
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W = (
    V10.BOOST_N_CANDIDATES, V10.BOOST_IC_L, V10.BOOST_P, V10.BOOST_SCALE_W)

print("=== MANDATORY sanity check: reuses batch100_shared's cached V10 baseline verbatim ===")
print(f"  baseline: OLD={S.base_wo:.1f}  NEW={S.base_wn:.1f}  rmean={S.base_scs.mean():.1f}  "
      f"rfloor={S.base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = S.SANITY_OK
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")

# internal check: BOOST_MIN_DAY..nt boost recomputed with AVG_DAYS=1 must equal V10._pairwise_boost
# exactly (uses the cached BOOST from batch100_shared as the reference).


def boost_avgN_at_day(k, navg):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = int(cand_idx[ci])
        lead_raw = rsl[i]
        if navg <= 1:
            lead = lead_raw
        else:
            csum = np.concatenate(([0.0], np.cumsum(lead_raw)))
            lead = np.full_like(lead_raw, np.nan)
            for idx in range(navg - 1, len(lead_raw)):
                lead[idx] = (csum[idx + 1] - csum[idx + 1 - navg]) / navg
            # fallback for the short warmup head: raw value (matches original series length/shape)
            lead[:navg - 1] = lead_raw[:navg - 1]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


def build_pos_from_boost(BOOST):
    WZ = np.full((nIdio, nt), np.nan)
    for t in S.days:
        wz = S.WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        wz = S.rs_blend(wz, t)
        WZ[:, t] = wz
    return S.build_pos_from_wz(WZ)


print("\n=== internal check: navg=1 must reproduce V10._pairwise_boost / cached S.BOOST exactly ===")
t0 = time.time()
BOOST_1 = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST_1[:, k] = boost_avgN_at_day(k, 1)
max_diff = float(np.nanmax(np.abs(BOOST_1 - S.BOOST)))
print(f"  max|diff| vs cached V10 boost = {max_diff:.3g} (should be ~0)  [{time.time()-t0:.0f}s]")

print("\n=== CANDIDATE: 2-day trailing average of the selected leader's return as boost input ===")
t0 = time.time()
BOOST_2 = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST_2[:, k] = boost_avgN_at_day(k, 2)
print(f"  boost computed [{time.time()-t0:.0f}s]")
c48 = S.evaluate("2-day avg leader", build_pos_from_boost(BOOST_2))

print(f"\nRESULT C48: passed={c48['passed']}  OLD={c48['wo']:.1f} NEW={c48['wn']:.1f} "
      f"rmean={c48['rm']:.1f} rfloor={c48['rf']:.1f} n_worse={c48['nworse']}/61")
print(f"SANITY_CHECK_PASSED={SANITY_OK}")
