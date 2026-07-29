"""
test_batch100_catC_lag_structure.py

CATEGORY C -- "own-series lag structure": seven candidate ideas about each idio name's OWN return
history (point autocorrelation at specific lags, a genuine VAR(2) extension of the shipped ridge, run-
length/streak persistence, momentum curvature/acceleration, and two variance/AR(1)-based differential-
sizing schemes), tested against the real shipped SAFE_llboost_v10.py via the verified `_v10_harness.py`.

Every idea is held to this repo's standard bar: a candidate must beat v10 on OLD (501-750), NEW
(751-1000), AND rolling mean jointly to "pass"; n_worse (of 61 rolling 250-day windows) is the
cleanliness metric. `H.evaluate()` enforces this exactly -- trusted as-is, not reimplemented.

Honest simplifications, stated up front (not hidden):
  - Variance-ratio (idea 6) uses the simple Lo-MacKinlay OVERLAPPING-window ratio
    Var(q-day sum)/(q*Var(1-day)) with NO small-sample / heteroskedasticity-robust correction term --
    a standard simplification, not the full LM(q) test statistic.
  - AR(1) (idea 7) is a plain causal rolling OLS slope of r_t on r_(t-1), not a bias-corrected or
    Yule-Walker estimator.
  - Streak (idea 4) treats an exact-zero daily return as a run-break (streak resets to 0), an edge
    case that essentially never occurs on continuous synthetic price data.
  - Ideas 6/7 modify the shipped 10-day REVERSAL leg's per-stock weight (not the rank-stability leg)
    -- chosen because the raw REV z-score is reconstructed directly in this file already (needed for
    idea 3's sanity gate anyway), whereas rank-stability's blend is only exposed as an opaque
    per-day helper (`H.rs_blend`) with no per-stock weight hook without reimplementing
    `V10._rank_stability_signal` from scratch.

Causality: every signal at day t uses only data available through column t of `H.logp` / column t-1
of `H.r` (yesterday's return) -- day t may never see its own future return.
"""
import numpy as np
import _v10_harness as H

RNG_SEED = 0
SUMMARY = []


def record(res):
    SUMMARY.append(res)
    return res


# ======================================================================================================
# Shared helper: pooled IC + circular-shift permutation p-value + H1/H2 persistence split.
# Shifts whole DAY-COLUMNS (not per-name) so cross-sectional structure within a day is preserved and
# only the day-to-day timing link between signal and next-day idio return is broken -- cheap and causal
# (uses only already-computed, causal SIG/target arrays; no new look-ahead is introduced).
# ======================================================================================================
def pooled_ic_perm(SIG, t_lo, t_hi, label, n_perm=200, seed=RNG_SEED):
    cols = np.array([t for t in range(t_lo, t_hi) if np.isfinite(SIG[:, t]).all()])
    Scols = SIG[:, cols]            # (nIdio, ncols)
    Ycols = H.rs_full[:, cols]      # (nIdio, ncols) -- next-day idio return aligned to day t
    X = Scols.ravel(); Y = Ycols.ravel()
    ic = float(np.corrcoef(X, Y)[0, 1])
    half = len(cols) // 2
    ic1 = float(np.corrcoef(Scols[:, :half].ravel(), Ycols[:, :half].ravel())[0, 1])
    ic2 = float(np.corrcoef(Scols[:, half:].ravel(), Ycols[:, half:].ravel())[0, 1])
    rng = np.random.default_rng(seed)
    ncols = len(cols)
    perm = np.empty(n_perm)
    for p in range(n_perm):
        shift = int(rng.integers(1, ncols - 1))
        perm[p] = np.corrcoef(np.roll(Scols, shift, axis=1).ravel(), Y)[0, 1]
    pval = float((np.abs(perm) >= abs(ic)).mean())
    print(f"  {label:<28}IC={ic:+.4f}  H1={ic1:+.4f}  H2={ic2:+.4f}  perm_p={pval:.3f}  n={len(cols)}d")
    return dict(ic=ic, ic1=ic1, ic2=ic2, pval=pval, ncols=len(cols))


def build_blended(SIG, weight):
    """Blend a standalone causal signal array SIG[nIdio,nt] into the shipped v10 BASE_WZ, per-day
    (blend_signal's internal z-score/std MUST be computed per day, not pooled over the whole array, or
    it silently uses future days' std to normalize a past day -- a look-ahead bug)."""
    WZ = H.BASE_WZ.copy()
    for t in H.days:
        s = SIG[:, t]
        if np.isfinite(s).all():
            WZ[:, t] = H.blend_signal(H.BASE_WZ[:, t], s, weight)
    return WZ


def sweep_weights(name, SIG, weights=(0.02, 0.05, 0.10, 0.15, 0.20, 0.30)):
    results = [record(H.evaluate(f"{name} w={w}", build_blended(SIG, w))) for w in weights]
    passing = [r for r in results if r["passed"]]
    if not passing:
        print(f"  -> 0/{len(results)} weights pass for {name}.")
    else:
        best = max(passing, key=lambda r: r["rm"])
        print(f"  -> {len(passing)}/{len(results)} weights pass; best rmean={best['rm']:.1f} ({best['name']})")
    return results


T_LO, T_HI = 150, H.nt - 1   # common pooled-IC sample window for all lag-structure ICs below

print("=" * 100)
print("IDEAS 1-2: point autocorrelation at lags k in {2,3,5,7} -- cheap IC sanity check FIRST")
print("=" * 100)


def lag_point_signal(k):
    """SIG[j,t] = logp[j+1,t-k] - logp[j+1,t-k-1] (== r[j+1,t-k-1], the single-day return exactly k
    days before day t per the assignment's own formula), cross-sectionally z-scored across the 50 idio
    names each day. Causal: only uses logp through column t-k <= t."""
    SIG = np.full((H.nIdio, H.nt), np.nan)
    for t in range(k + 1, H.nt):
        raw = H.logp[1:, t - k] - H.logp[1:, t - k - 1]
        s = raw.std()
        SIG[:, t] = (raw - raw.mean()) / (s + 1e-12) if s > 1e-12 else 0.0
    return SIG


LAGS = [2, 3, 5, 7]
lag_ic = {}
for k in LAGS:
    lag_ic[k] = pooled_ic_perm(lag_point_signal(k), T_LO, T_HI, f"lag k={k}")

NONTRIVIAL = {k: v for k, v in lag_ic.items() if abs(v["ic"]) > 0.02 and v["pval"] < 0.20}
if not NONTRIVIAL:
    print("\n  VERDICT: all 4 lags show IC indistinguishable from noise (|IC|<=0.02 or perm_p>=0.20 at "
          "every lag) -- consistent with this repo's prior finding (README: ridge-residual own-\n"
          "  autocorrelation at lags 1-30 all under 0.013, no pattern). NOT running full backtests for "
          "any of these 4 -- there is nothing here to backtest.")
else:
    print(f"\n  {len(NONTRIVIAL)}/4 lags look non-trivial ({list(NONTRIVIAL)}) -- backtesting those via "
          "blend-weight sweep.")
    for k in NONTRIVIAL:
        sweep_weights(f"lag_k{k}", lag_point_signal(k))


# ======================================================================================================
print("\n" + "=" * 100)
print("IDEA 3: VAR(2) ridge extension -- add each instrument's lag-2 return as extra predictor columns")
print("=" * 100)


def build_wz_var(t, use_lag2):
    """Reproduces the ridge half-life ensemble (mean of _ewls_ridge forecasts across HALF_LIVES) for a
    single day t. use_lag2=False is a byte-for-byte reimplementation of the lag-1-only fit inside
    _v10_harness.py's own WZ_PRE precompute loop (the sanity gate). use_lag2=True widens X to 102
    columns: [r[:,t-1] (yesterday), r[:,t-2] (day before yesterday)], with the SAME target row
    (Y row k pairs with predictor row k -> both derived from rr_ columns k and k-1, target rs[:,k+1]),
    dropping the first training row (index 0) since it has no k-1 predictor available."""
    rr_ = H.r[:, :t]
    X1 = rr_[:, :-1].T                     # (t-1, 51); row k = rr_[:,k], target Y[k] = rs[:,k+1]
    Y = H.V10._beta_adjusted_target(rr_)   # (t-1, 50)
    if use_lag2:
        X = np.concatenate([X1[1:], X1[:-1]], axis=1)   # (t-2, 102): [rr_[:,k], rr_[:,k-1]] for k=1..t-2
        Yr = Y[1:]
        xq = np.concatenate([rr_[:, -1], rr_[:, -2]])
    else:
        X, Yr, xq = X1, Y, rr_[:, -1]
    fs = []
    for hl in H.V10.HALF_LIVES:
        B, mx, my = H.V10._ewls_ridge(X, Yr, hl, H.V10.RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    return np.mean(fs, 0)


def build_wz_var_full(use_lag2, apply_blend):
    WZ = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        wz = build_wz_var(t, use_lag2)
        if apply_blend and H.V10.BLEND > 0:
            rv_ = H.logp[1:, t] - H.logp[1:, t - H.V10.REV_W]
            rv_ = rv_ - rv_.mean()
            rv_ = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - H.V10.BLEND) * wz + H.V10.BLEND * rv_
        WZ[:, t] = wz
    return WZ


def full_pipeline(WZ_ridge_rev):
    WZ = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        wz = WZ_ridge_rev[:, t] + H.V10.BOOST_K * H.BOOST_BASE[:, t]
        WZ[:, t] = H.rs_blend(wz, t)
    return WZ


print("\n--- sanity gate: lag-1-only reproduction of H.WZ_PRE (must match to within rounding) ---")
WZ_lag1_sanity = build_wz_var_full(use_lag2=False, apply_blend=True)
diff = np.nanmax(np.abs(WZ_lag1_sanity[:, H.days] - H.WZ_PRE[:, H.days]))
print(f"  max |diff| vs H.WZ_PRE across all days = {diff:.2e}")
if diff > 1e-6:
    print("  *** SANITY GATE FAILED -- lag-1 reimplementation does not match H.WZ_PRE. STOPPING idea 3, "
          "do not trust anything below for this idea. ***")
    IDEA3_OK = False
else:
    print("  OK -- exact reproduction confirmed. Safe to add lag-2 on top of this code path.")
    IDEA3_OK = True

if IDEA3_OK:
    print("\n--- double-check: full_pipeline(H.WZ_PRE) must reproduce H.BASE_WZ / real v10 numbers ---")
    res = record(H.evaluate("lag1 full-pipeline (=v10 double-check)", full_pipeline(H.WZ_PRE)))
    print(f"  (docstring v10 numbers: OLD=871.0 NEW=912.6 rmean=909.8 rfloor=709.7)")

    print("\n--- RAW comparison (no REV blend, no boost, no RS -- isolates the ridge estimator itself) ---")
    WZ_lag1_raw = build_wz_var_full(use_lag2=False, apply_blend=False)
    WZ_lag2_raw = build_wz_var_full(use_lag2=True, apply_blend=False)
    ic1 = pooled_ic_perm(WZ_lag1_raw, T_LO, T_HI, "raw lag1-only ridge")
    ic2 = pooled_ic_perm(WZ_lag2_raw, T_LO, T_HI, "raw lag1+lag2 ridge")
    r1 = record(H.evaluate("raw lag1-only (vs v10, expect much lower -- missing REV/boost/RS)", WZ_lag1_raw))
    r2 = record(H.evaluate("raw lag1+lag2  (vs v10, expect much lower -- missing REV/boost/RS)", WZ_lag2_raw))
    promising = (ic2["ic"] > ic1["ic"]) and (r2["rm"] > r1["rm"])
    print(f"\n  lag2 raw IC {'>' if ic2['ic']>ic1['ic'] else '<='} lag1 raw IC "
          f"({ic2['ic']:+.4f} vs {ic1['ic']:+.4f}); lag2 raw rmean {'>' if r2['rm']>r1['rm'] else '<='} "
          f"lag1 raw rmean ({r2['rm']:.1f} vs {r1['rm']:.1f}) -- {'PROMISING' if promising else 'NOT promising'}")

    if promising:
        print("\n--- raw comparison promising -- running FULL pipeline (+REV blend, +boost, +RS) for lag2 ---")
        WZ_lag2_rev = build_wz_var_full(use_lag2=True, apply_blend=True)
        WZ_lag2_full = full_pipeline(WZ_lag2_rev)
        record(H.evaluate("VAR(2) full pipeline vs real v10", WZ_lag2_full))
    else:
        print("\n  Raw comparison is NOT promising -- per this repo's own stopping convention (only run "
              "the expensive full-pipeline confirmatory test if the cheap raw check looks promising), "
              "skipping the full-pipeline VAR(2) run. Verdict: adding lag-2 does not sharpen the raw "
              "ridge estimator here.")
else:
    WZ_lag1_raw = build_wz_var_full(use_lag2=False, apply_blend=False)  # still needed below for ideas 6/7


# ======================================================================================================
print("\n" + "=" * 100)
print("IDEA 4: streak length (run-length persistence) -- distinct from a plain k-day cumulative return")
print("=" * 100)
print("""  Construction note: streak counts RUN LENGTH (how many consecutive same-sign daily moves just
  happened), completely blind to move SIZE -- three +0.1% days and three +5% days both give streak=+3.
  A plain k-day cumulative return is the opposite: it is exactly the net displacement and is blind to
  the internal path shape -- three alternating +1%/-1% days net to ~0 (no reversal-worthy signal) even
  though each individual day was a full-sized move, while a streak signal on that same path is also
  ~0 (it keeps breaking) -- so on THAT path they happen to agree. They diverge sharply whenever the
  within-window move sizes are uneven: a path of +0.1%,+0.1%,+5% has the same 3-day cumulative return
  as +5%,+0.1%,+0.1%, but streak (+3 either way, sign/count only) cannot tell them apart at all, whereas
  the two would (in general) give different k-day returns unless the total happens to match by
  coincidence -- streak is a pure shape/persistence statistic, cumulative return is a pure level
  statistic; neither is a special case of the other.""")


def streak_signal():
    streak = np.zeros((H.nIdio, H.nt - 1))
    for j in range(H.nIdio):
        rj = H.r[j + 1]
        run = 0
        for i in range(len(rj)):
            s = np.sign(rj[i])
            run = (run + 1) if (s != 0 and (i == 0 or s == np.sign(rj[i - 1]))) else (1 if s != 0 else 0)
            streak[j, i] = run * s
    SIG = np.full((H.nIdio, H.nt), np.nan)
    for t in range(1, H.nt):
        raw = streak[:, t - 1]     # ends at r[:,t-1] -- causal, most recent info before day t
        s = raw.std()
        SIG[:, t] = (raw - raw.mean()) / (s + 1e-12) if s > 1e-12 else 0.0
    return SIG


SIG_STREAK = streak_signal()
pooled_ic_perm(SIG_STREAK, T_LO, T_HI, "streak")
sweep_weights("streak", SIG_STREAK)


# ======================================================================================================
print("\n" + "=" * 100)
print("IDEA 5: acceleration (momentum curvature) -- distinct from a plain k-day return")
print("=" * 100)
print("""  Construction note: acceleration = (last-5d return) - (prior-5d return) = logp[t] - 2*logp[t-5]
  + logp[t-10], the discrete SECOND difference of log-price -- it measures whether the trend is
  speeding up or slowing down. A plain 10-day return (logp[t]-logp[t-10]) is the discrete FIRST
  difference (total displacement) and is blind to curvature entirely: a flat +1%/5days, +1%/5days ramp
  (constant speed) and a +0%/5days, +2%/5days ramp (accelerating) can share the exact same 10-day
  total return while acceleration is ~0 for the first and strongly positive for the second.""")


def acceleration_signal(w=5):
    SIG = np.full((H.nIdio, H.nt), np.nan)
    for t in range(2 * w, H.nt):
        raw = H.logp[1:, t] - 2 * H.logp[1:, t - w] + H.logp[1:, t - 2 * w]
        s = raw.std()
        SIG[:, t] = (raw - raw.mean()) / (s + 1e-12) if s > 1e-12 else 0.0
    return SIG


SIG_ACCEL = acceleration_signal(5)
pooled_ic_perm(SIG_ACCEL, T_LO, T_HI, "acceleration(5)")
sweep_weights("accel5", SIG_ACCEL)


# ======================================================================================================
print("\n" + "=" * 100)
print("IDEAS 6-7: variance-ratio / AR(1) differential sizing of the shipped 10-day reversal leg")
print("=" * 100)
print("""  Both ideas differentially scale each stock's contribution to the REV leg (not rank-stability --
  the raw REV z-score is already reconstructed above for idea 3's sanity gate, whereas RS's blend is
  only exposed as an opaque per-day helper with no per-stock weight hook). Repo prior (README): this
  book's positions are FIXED-DOLLAR, SIGN-ONLY per name -- '+1% mean = +1.03% score, -1% stdev =
  +0.03% score' (30:1 mean-vs-variance elasticity) -- every differential-sizing scheme tried so far
  (Kelly, vol-targeting, confidence ramps, cluster-neutral, leader-stability soft-multiplier) has lost
  or been a razor-thin non-replicable spike. Both ideas here are further tests of that same wall, not
  a new hypothesis about it being wrong.""")

REV_RAW = np.zeros((H.nIdio, H.nt))
for t in H.days:
    rv_ = H.logp[1:, t] - H.logp[1:, t - H.V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV_RAW[:, t] = -rv_ / (rv_.std() + 1e-12)

WZ_RIDGE_ONLY = WZ_lag1_raw  # reuse idea 3's lag-1, no-blend ridge-only forecast


def gated_rev_wz(strength_2d, gain):
    """gain=0 -> uniform weight=1 for every stock every day -> must reproduce the real, shipped v10
    exactly (sanity check). gain>0 -> per-stock REV weight = clip(1 + gain*zscore(strength), 0.1, inf),
    renormalized to average 1 across the 50 names each day (so the LEG's overall average magnitude is
    unchanged -- only the cross-sectional allocation within it shifts)."""
    WZ = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        rev = REV_RAW[:, t]
        s = strength_2d[:, t]
        if gain > 0 and np.isfinite(s).all() and s.std() > 1e-12:
            z = (s - s.mean()) / (s.std() + 1e-12)
            w = np.clip(1.0 + gain * z, 0.1, None)
            rev_mod = (w / w.mean()) * rev
        else:
            rev_mod = rev
        wz = (1 - H.V10.BLEND) * WZ_RIDGE_ONLY[:, t] + H.V10.BLEND * rev_mod
        wz = wz + H.V10.BOOST_K * H.BOOST_BASE[:, t]
        WZ[:, t] = H.rs_blend(wz, t)
    return WZ


print("\n--- sanity gate: gain=0 (uniform weight) must reproduce real v10 exactly ---")
res0 = record(H.evaluate("gated_rev gain=0 sanity", gated_rev_wz(np.zeros((H.nIdio, H.nt)), 0.0)))
if abs(res0["wo"] - H.BASE_WO) > 0.5 or abs(res0["wn"] - H.BASE_WN) > 0.5:
    print("  *** SANITY GATE FAILED for gated_rev_wz -- does not reproduce v10 at gain=0. STOPPING "
          "ideas 6/7, do not trust results below. ***")
    IDEA67_OK = False
else:
    print("  OK -- exact reproduction confirmed at gain=0.")
    IDEA67_OK = True

print("\n--- idea 6: variance-ratio VR(5) reversion-strength gate (Lo-MacKinlay style, overlapping-window) ---")


def compute_VR(q=5, window=250):
    VR = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        if t - window < 0:
            continue
        seg = H.r[1:, t - window:t]                      # (nIdio, window), causal
        var1 = seg.var(axis=1)
        csum = np.concatenate([np.zeros((H.nIdio, 1)), np.cumsum(seg, axis=1)], axis=1)
        segq = csum[:, q:] - csum[:, :-q]
        varq = segq.var(axis=1)
        VR[:, t] = varq / (q * (var1 + 1e-12))
    return VR


VR = compute_VR()
valid = np.isfinite(VR[:, H.days[-1]])
print(f"  VR(5) at final day: mean={np.nanmean(VR[:, H.days[-1]]):.3f}  "
      f"(<1 => reversion, >1 => momentum; {int((VR[:, H.days[-1]] < 1).sum())}/{H.nIdio} names <1)")
STRENGTH_VR = np.clip(1.0 - VR, 0.0, None)   # larger = more reversion evidence = more weight

if IDEA67_OK:
    GAINS = [0.3, 0.6, 1.0, 1.5, 2.5]
    for g in GAINS:
        record(H.evaluate(f"VR-gated REV gain={g}", gated_rev_wz(STRENGTH_VR, g)))

print("\n--- idea 7a: mathematical no-op demo -- scaling the FINAL wz by a positive per-stock multiplier ---")
print("""  This book's position sizing is sign(wz)*fixed_dollar_amount (see build_pos_from_wz) --
  multiplying the FINAL, already-fully-combined wz by any per-stock multiplier that stays strictly
  POSITIVE cannot change sign(wz) for a single name, hence cannot change a single position, hence
  cannot change the score AT ALL. Demonstrated directly (not just argued) before testing the more
  meaningful placement (7b, mirroring idea 6's REV-leg construction):""")


def compute_AR1(window=250):
    PHI = np.full((H.nIdio, H.nt), np.nan)
    for t in H.days:
        if t - window - 1 < 0:
            continue
        seg = H.r[1:, t - window:t]
        x, y = seg[:, :-1], seg[:, 1:]
        mx = x.mean(axis=1, keepdims=True); my = y.mean(axis=1, keepdims=True)
        cov = ((x - mx) * (y - my)).mean(axis=1)
        varx = ((x - mx) ** 2).mean(axis=1)
        PHI[:, t] = cov / (varx + 1e-12)
    return PHI


PHI = compute_AR1()
STRENGTH_AR1 = np.abs(PHI)   # monotonic proxy for implied half-life = ln(0.5)/ln|phi| (order-preserving
                              # for |phi| in (0,1), so a weight built from a rank/normalized function of
                              # |phi| is identical to one built from the half-life itself)
print(f"  AR(1) phi at final day: mean={np.nanmean(PHI[:, H.days[-1]]):+.4f}  "
      f"std={np.nanstd(PHI[:, H.days[-1]]):.4f}")

mult = np.ones((H.nIdio, H.nt))
for t in H.days:
    s = STRENGTH_AR1[:, t]
    if np.isfinite(s).all() and s.mean() > 1e-12:
        mult[:, t] = np.clip(s / s.mean(), 0.3, 3.0)     # always strictly positive by construction
WZ_noop = H.BASE_WZ * mult
sign_diff = int((np.sign(WZ_noop[:, H.days]) != np.sign(H.BASE_WZ[:, H.days])).sum())
print(f"  sign flips introduced by this positive-only rescaling: {sign_diff} (must be 0)")
res_noop = record(H.evaluate("AR1 final-wz positive rescale (no-op demo)", WZ_noop))
print(f"  -> scores byte-identical to real v10, as predicted: "
      f"{'CONFIRMED' if abs(res_noop['wo']-H.BASE_WO)<1e-6 and abs(res_noop['wn']-H.BASE_WN)<1e-6 else 'MISMATCH -- investigate'}")

if IDEA67_OK:
    print("\n--- idea 7b: AR(1)-gated REV leg (meaningful placement, same construction as idea 6) ---")
    for g in GAINS:
        record(H.evaluate(f"AR1-gated REV gain={g}", gated_rev_wz(STRENGTH_AR1, g)))


# ======================================================================================================
print("\n" + "=" * 100)
print("FINAL SUMMARY -- all candidates tested this batch")
print("=" * 100)
print(f"  {'name':<48}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>8}{'n_worse':>10}  passed")
for r in SUMMARY:
    print(f"  {r['name']:<48}{r['wo']:8.1f}{r['wn']:8.1f}{r['rm']:8.1f}{r['rf']:8.1f}{r['nworse']:>7}/61  "
          f"{'PASS' if r['passed'] else 'fail'}")
n_pass = sum(1 for r in SUMMARY if r["passed"])
print(f"\n{n_pass}/{len(SUMMARY)} total configurations passed OLD+NEW+rmean jointly vs real v10 "
      f"(baseline: OLD={H.BASE_WO:.1f} NEW={H.BASE_WN:.1f} rmean={H.BASE_SCS.mean():.1f} "
      f"rfloor={H.BASE_SCS.min():.1f}).")
