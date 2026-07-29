"""
test_batch100_I88.py

DIAGNOSTIC/METHODOLOGY (I88): strict walk-forward check. The reported NEW=912.6 for SAFE_llboost_v10
was reached with the researcher having visibility into BOTH the OLD (500-750) and NEW (750-1000)
windows while choosing RIDGE_A, BOOST_K, RS_WEIGHT (and everything else). How much of that number
survives if those three key parameters are instead RESELECTED using performance measurable ONLY through
day 750, then simply applied forward with no further tuning?

METHOD: grid RIDGE_A x BOOST_K x RS_WEIGHT. For each RIDGE_A candidate, rebuild the ridge ensemble (the
only piece of the pipeline that actually depends on RIDGE_A) restricted to days <= 750 (cheap subset --
selection never needs data past 750). BOOST and RS_RAW do not depend on RIDGE_A/BOOST_K/RS_WEIGHT so
the cached arrays are reused (and are themselves already fully causal per-day, so slicing them at <=750
introduces no leakage). SELECTION metric: rolling mean of wscore over end_days in range(400, 751, 10)
(i.e. every walk-forward test window that is entirely resolvable using only days <=750 -- this is
exactly the same rolling-mean philosophy the actual model selection used, just truncated at the 750
boundary instead of running to 1000). The winning combo is then applied with NO further changes, and
scored on the genuinely-blind NEW window (750-1000) plus the full-range OLD/rolling-mean/rolling-floor,
for direct comparison to the already-reported v10 numbers.
"""
import time
import numpy as np
import batch100_common_gi as B
import SAFE_llboost_v10 as V10

sanity_ok = B.print_sanity("(shared cache)")

P_, logp, r, rs = B.P_, B.logp, B.r, B.rs
nIdio, nt, dlr = B.nIdio, B.nt, B.dlr
HALF_LIVES = B.HALF_LIVES
BLEND, REV_W = V10.BLEND, V10.REV_W
BOOST_MIN_DAY = B.BOOST_MIN_DAY
BOOST_cached, RS_RAW = B.BOOST, B.RS_RAW
REV_cached, WZ_RIDGE_cached = B.REV, B.WZ_RIDGE

SEL_END_DAYS = list(range(400, 751, 10))  # every rolling-mean test window resolvable using only <=750
FIT_UPTO = 751  # need ridge/positions for days < FIT_UPTO (750 inclusive) for selection


def ridge_ensemble(ridge_a, day_list):
    """Rebuild the per-half-life ridge ensemble (WZ_RIDGE) for the given RIDGE_A, only for the days in
    day_list (causal per-day, identical math to precompute_batch100.py / V10 itself)."""
    WZR = np.zeros((nIdio, B.nt))
    for t in day_list:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Y = V10._beta_adjusted_target(rr_)
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            Bm, mx, my = V10._ewls_ridge(X, Y, hl, ridge_a)
            pred = my + (xq - mx) @ Bm
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZR[:, t] = np.mean(fs, 0)
    return WZR


def build_and_select(RIDGE_As, BOOST_Ks, RS_Ws, day_list, upto_label):
    t0 = time.time()
    ridge_cache = {}
    for ra in RIDGE_As:
        if abs(ra - V10.RIDGE_A) < 1e-12:
            ridge_cache[ra] = WZ_RIDGE_cached  # reuse full-range cached ridge, just index into day_list
        else:
            ridge_cache[ra] = ridge_ensemble(ra, day_list)
    print(f"  ridge ensembles for RIDGE_A in {RIDGE_As} built ({time.time() - t0:.0f}s)", flush=True)

    results = []
    for ra in RIDGE_As:
        WZR = ridge_cache[ra]
        WZ_PRE = np.zeros((nIdio, B.nt))
        for t in day_list:
            WZ_PRE[:, t] = (1 - BLEND) * WZR[:, t] + BLEND * REV_cached[:, t]
        for bk in BOOST_Ks:
            wz_preboost = np.zeros((nIdio, B.nt))
            for t in day_list:
                wz = WZ_PRE[:, t].copy()
                if t >= BOOST_MIN_DAY:
                    wz = wz + bk * BOOST_cached[:, t]
                wz_preboost[:, t] = wz
            for rw in RS_Ws:
                WZ = np.zeros((nIdio, B.nt))
                for t in day_list:
                    wz = wz_preboost[:, t]
                    s = RS_RAW[:, t]
                    if np.isfinite(s).all():
                        sstd = s.std()
                        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
                        wz = (1 - rw) * wz + rw * s_z * (np.abs(wz).mean() + 1e-12)
                    WZ[:, t] = wz
                POS = np.zeros((B.nInst, B.nt))
                for t in day_list:
                    wzc = WZ[:, t]
                    cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
                    POS[1:, t] = np.clip(np.sign(wzc) * (dlr[1:] / cur[1:]), -lim, lim)
                POS[0, :] = B.algo_pos
                sel_scs = np.array([B.wscore(POS, E - B.NUMTEST, E) for E in SEL_END_DAYS])
                results.append(dict(ra=ra, bk=bk, rw=rw, sel=sel_scs.mean(), WZ=WZ if upto_label == "keep" else None))
    return results


print("\n=== FIT stage: grid search using ONLY days <= 750 (selection metric: rolling mean of wscore "
      "over end_days 400..750) ===")
print("  NOTE (scope, screening pass): re-deriving the ridge ensemble for a new RIDGE_A is the one "
      "genuinely expensive step here (~minutes per value, since _ewls_ridge/_beta_adjusted_target "
      "cost grows with trailing history length); BOOST_K and RS_WEIGHT are cheap post-hoc multipliers "
      "on cached arrays and are swept more densely. RIDGE_A is tested at shipped=0.1 (free, cached) "
      "plus ONE alternative (0.05) rather than a dense grid, consistent with the batch's time budget.")
RIDGE_As = [0.1, 0.05]
BOOST_Ks = [1.0, 1.5, 2.0]
RS_Ws = [0.010, 0.015, 0.020]
fit_days = [t for t in B.days if t < FIT_UPTO]
fit_results = build_and_select(RIDGE_As, BOOST_Ks, RS_Ws, fit_days, upto_label="drop")

fit_results.sort(key=lambda d: -d["sel"])
print(f"  {len(fit_results)} combos evaluated. Top 5 by in-sample (<=750) selection metric:")
for d in fit_results[:5]:
    print(f"    RIDGE_A={d['ra']:<5} BOOST_K={d['bk']:<4} RS_WEIGHT={d['rw']:<6} sel_rmean={d['sel']:.1f}")

best = fit_results[0]
print(f"\n  SELECTED (using only days<=750): RIDGE_A={best['ra']}  BOOST_K={best['bk']}  "
      f"RS_WEIGHT={best['rw']}  (shipped v10: RIDGE_A={V10.RIDGE_A} BOOST_K={V10.BOOST_K} "
      f"RS_WEIGHT={V10.RS_WEIGHT})")

print("\n=== BLIND stage: apply the SELECTED combo forward, full range, NO further tuning ===")
full_days = B.days
full_res = build_and_select([best["ra"]], [best["bk"]], [best["rw"]], full_days, upto_label="keep")[0]
WZ_final = full_res["WZ"]
POS_final = B.build_pos_from_wz(WZ_final)
r_final = B.evaluate(f"walk-forward selected (RIDGE_A={best['ra']},BOOST_K={best['bk']},"
                      f"RS_WEIGHT={best['rw']})", POS_final)

print(f"\n=== comparison ===")
print(f"  shipped v10 (selected with visibility into BOTH windows): "
      f"OLD={B.base_wo:.1f} NEW={B.base_wn:.1f} rmean={B.base_scs.mean():.1f} rfloor={B.base_scs.min():.1f}")
print(f"  strict walk-forward (selected using ONLY days<=750):        "
      f"OLD={r_final['wo']:.1f} NEW={r_final['wn']:.1f} rmean={r_final['rm']:.1f} rfloor={r_final['rf']:.1f}")
gap = B.base_wn - r_final['wn']
print(f"  gap in blind NEW score: {gap:.1f} points "
      f"({'walk-forward selection would have cost real out-of-sample performance' if gap > 5 else 'walk-forward selection is nearly free here -- little visible overfitting from dual-window tuning'})")
