"""
test_batch100_C47.py

C47: Replace the significance-thresholded correlation TEST with a formal Granger-causality test for
leader detection in the pairwise boost.

Selection of the CANDIDATE leader itself is left as argmax|corr| over the same BOOST_N_CANDIDATES=39
causal-vol-ranked pool V10._pairwise_boost already uses (that's the "leader detection" search step,
unaffected by this idea's wording, which targets the significance TEST). What changes is the GATE:
instead of |corr| > Bonferroni-corrected t-test threshold, a Granger-causality test (statsmodels,
maxlag=1, lag-1 "does x help predict y beyond y's own lag-1?") on (candidate return -> follower
next-day return) must reject the null at BOOST_ALPHA (same Bonferroni correction across
BOOST_N_CANDIDATES simultaneous tests) for the boost to fire.

SCOPE NOTE (screening pass): a full pairwise Granger re-search over all 39 candidates x 50 followers
x ~520 days would be ~20x the compute of this design for the same conclusion (replacing a TEST, not
the whole search), so only the already-argmax-selected best candidate is Granger-tested per
(follower, day) -- this is the "replace the significance-thresholded correlation test" reading, not
"replace the whole search with Granger".

Everything else (idio ridge ensemble + beta-adjusted target, BLEND reversion, rank-stability blend,
ALGO leg) is reused VERBATIM from V10 via batch100_shared's cached precompute -- unaffected by this
idea, which only touches the boost's leader-detection gate.
"""
import io, contextlib, time
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import SAFE_llboost_v10 as V10
import batch100_shared as S

nInst, nt, nIdio = S.nInst, S.nt, S.nIdio
rs = S.rs
BOOST_MIN_DAY, BOOST_K = S.BOOST_MIN_DAY, S.BOOST_K
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W, BOOST_ALPHA = (
    V10.BOOST_N_CANDIDATES, V10.BOOST_IC_L, V10.BOOST_P, V10.BOOST_SCALE_W, V10.BOOST_ALPHA)

print("=== MANDATORY sanity check: reuses batch100_shared's cached V10 baseline verbatim ===")
print(f"  baseline: OLD={S.base_wo:.1f}  NEW={S.base_wn:.1f}  rmean={S.base_scs.mean():.1f}  "
      f"rfloor={S.base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = S.SANITY_OK
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


def granger_pvalue(x, y):
    """H0: x's lag-1 does NOT help predict y beyond y's own lag-1 (standard Granger F-test)."""
    data = np.column_stack([y, x])
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = grangercausalitytests(data, maxlag=1, verbose=False)
        return float(res[1][0]['ssr_ftest'][1])
    except Exception:
        return 1.0


def boost_granger_at_day(k):
    rsl = rs[:, :k]; n, T = rsl.shape
    boost = np.zeros(n)
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    alpha_adj = BOOST_ALPHA / BOOST_N_CANDIDATES
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        i = int(cand_idx[ci])
        a = max(0, n_samples - 250)          # windowed for tractability, same convention as C45/C46
        xw = Xi_full[cand_idx[ci], a:]; yw = Yj[j, a:]
        ok = ~np.isnan(xw) & ~np.isnan(yw)
        if ok.sum() < 60 or xw[ok].std() < 1e-12:
            continue
        p = granger_pvalue(xw[ok], yw[ok])
        if p >= alpha_adj:
            continue
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        aa = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[aa:T - 1]; ys = rsl[j, aa + 1:T]
        ok2 = ~np.isnan(xs) & ~np.isnan(ys)
        if ok2.sum() < 60 or xs[ok2].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok2], ys[ok2])[0, 1])
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


print("\n=== CANDIDATE: Granger-causality gate (maxlag=1, window=250) replacing corr significance test "
      "on the argmax-selected candidate ===")
t0 = time.time()
BOOST_G = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST_G[:, k] = boost_granger_at_day(k)
print(f"  boost computed [{time.time()-t0:.0f}s]")
c47 = S.evaluate("granger gate", build_pos_from_boost(BOOST_G))

print(f"\nRESULT C47: passed={c47['passed']}  OLD={c47['wo']:.1f} NEW={c47['wn']:.1f} "
      f"rmean={c47['rm']:.1f} rfloor={c47['rf']:.1f} n_worse={c47['nworse']}/61")
print(f"SANITY_CHECK_PASSED={SANITY_OK}")
