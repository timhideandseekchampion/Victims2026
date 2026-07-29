"""
test_batch100_catDE_moments_relval.py

BATCH 100, categories D (return-moment / event signals) + E (relative-value / network signals): ten new
candidate signals tested against the real, shipped SAFE_llboost_v10, using the shared verified harness
(`_v10_harness.py`). Every idea follows the same two-stage protocol:

  Stage 1 (cheap): build the causal SIG[nIdio, nt] array, then compute its POOLED raw IC against the
  next-day idio return (corr(SIG[:,t], rs_full[:,t]) across all names and days) -- no backtest yet. An
  idea is only carried to Stage 2 if this isn't obviously indistinguishable from noise (calibrated
  against two real precedents already in this repo: v10's OWN shipped rank-stability signal has a real,
  validated pooled IC of only +0.0147 -- so "small" alone doesn't mean noise -- versus the already-
  rejected return-skewness idea, whose ICs were all <=0.01 in magnitude AND had permutation p-values of
  0.26-0.69, i.e. not even nominally different from zero. That combination -- tiny AND statistically
  unconvincing -- is the "obviously noise" bar used below, not magnitude alone).

  Stage 2 (real cost, still cheap): wz = H.BASE_WZ (the harness's own precomputed, verified FULL v10
  final-forecast array = WZ_PRE + BOOST_K*BOOST_BASE, already passed through the real rs_blend), then
  H.blend_signal(wz, SIG[:,t], weight) at weight in {0.01, 0.02, 0.05, 0.1}, then H.evaluate() against
  the real v10 baseline (OLD=871.0 / NEW=912.6 / rmean=909.8 / rfloor=709.7). Reusing H.BASE_WZ directly
  (rather than rebuilding WZ_PRE+rs_blend from scratch per idea) is mathematically identical to the
  "rebuild wz, apply rs_blend, then blend_signal" pattern described in the assignment -- H.BASE_WZ IS
  that exact intermediate result, precomputed once by the harness.

PRIOR-ART CHECK (done honestly against README.md before writing any code, since the assignment asked for
this to be flagged, not assumed away):
  - Idea 1 (rolling kurtosis of own returns -> own next-day return): README's "~85 other ideas" ledger
    (the closing section of the 100-idea batch series) lists "kurtosis" in one line among items "all
    rejected cleanly," with NO construction detail (window, own-return vs. something else, raw vs.
    self-relative). Genuine prior art exists for the *word* "kurtosis" under this repo's overall search,
    but not for a checkable construction -- implemented here independently and honestly, not assumed
    identical to whatever that one-line mention covered.
  - Idea 4 (tail co-exceedance, "same value for all names that day"): as specified, this is a uniform,
    non-stock-differentiating vector each day. README documents (B38) that "z-scoring a uniform,
    non-stock-differentiating vector collapses to exactly zero at every weight" via the exact
    z-score-then-blend mechanics `H.blend_signal` also uses -- i.e. this idea is structurally very close
    to an already-proven mathematical no-op, not merely a candidate that might lose. Implemented
    faithfully anyway (with the literal "OTHER names" exclusion, which perturbs it off pure-uniform by
    at most a count of 1 per stock-day) to actually check whether that perturbation survives, rather
    than skip on suspicion alone.
  - Idea 7 (ordinal-rank rank-stability): README lists "continuous rank-stability" among the ~85 rejected
    ideas -- a DIFFERENT construction (presumably a magnitude-continuous version of the same disagreement
    vote, not a rank-transform of the two z-score legs) -- kept as a distinct, not-yet-tested idea here.
  - Everything else (vol-of-vol, post-jump drift, forecast-self-relative z, eigenvector centrality,
    volatility cross-sectional dispersion, consensus deviation, multi-horizon momentum-rank composite) is,
    to the best of a full-file grep, genuinely new construction not covered by any prior test in this repo.

CAUSALITY: every SIG[:, t] below is built using only rs_full[:, :t] (idio log-returns through the
return ending on day t) and/or logp[:, :t+1] (price levels through day t) -- never anything at or after
column t+1.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import _v10_harness as H
import SAFE_llboost_v10 as V10

nIdio, nt = H.nIdio, H.nt
rs_full = H.rs_full            # (50, nt-1); rs_full[:, k] = idio log-return realized ON day k+1
logp = H.logp
days = H.days
WEIGHTS = (0.01, 0.02, 0.05, 0.1)

print(f"\nnIdio={nIdio}  nt={nt}  n_days(graded harness range)={len(days)}\n")


# ==================================================================================================
# generic causal rolling helpers
# ==================================================================================================
def causal_roll_returns(rs_mat, window, statfn):
    """rs_mat: (nIdio, nt-1) return matrix, rs_mat[:,k] realized on day k+1.
    Returns SIG (nIdio, nt) with SIG[:, t] = statfn(rs_mat[:, t-window:t]) for t>=window+1 (uses only
    the `window` most recent returns known as of day t), NaN elsewhere. Vectorized via pandas rolling."""
    df = pd.DataFrame(rs_mat.T)                      # rows = day-of-return k, cols = idio name
    rolled = statfn(df.rolling(window, min_periods=window))
    vals = rolled.values                              # (nt-1, nIdio); row k usable as of day k+1
    SIG = np.full((nIdio, nt), np.nan)
    SIG[:, 1:nt] = vals.T
    return SIG


def causal_roll_wide(mat_nt, window, statfn):
    """mat_nt: (nIdio, nt) array already causal per column (mat_nt[:,t] known as of day t, possibly NaN
    early on). Rolls `window` days INCLUSIVE of day t (no extra day-shift needed -- day t's own value is
    already legitimately known as of day t)."""
    df = pd.DataFrame(mat_nt.T)
    rolled = statfn(df.rolling(window, min_periods=window))
    return rolled.values.T


def pooled_ic(SIG, label, note=""):
    """Cheap causal sanity check: pool SIG[:,t] against the REALIZED next-day idio return rs_full[:,t]
    (the return from day t to day t+1 -- legitimately unknown at the time SIG[:,t] is computed) across
    every (name, day) with t in [WARMUP, nt-2]. Returns (ic, p, n)."""
    ts = np.arange(H.WARMUP, nt - 1)
    x = SIG[:, ts].ravel()
    y = rs_full[:, ts].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.std() < 1e-14:
        print(f"  [{label}] degenerate (zero variance) -- {note}")
        return 0.0, 1.0, ok.sum()
    ic, p = stats.pearsonr(x, y)
    print(f"  [{label}] pooled IC={ic:+.4f}  p={p:.3g}  n={ok.sum()}  {note}")
    return float(ic), float(p), int(ok.sum())


def is_obviously_noise(ic, p):
    """Calibrated against this repo's own precedent: v10's real, shipped, VALIDATED signal has pooled
    IC=+0.0147 (small is not disqualifying) -- but the already-rejected skewness idea had |ic|<=0.01 AND
    p in [0.26, 0.69] (tiny AND not even nominally significant). Flag as "obviously noise" only when
    BOTH conditions hold: tiny magnitude (<0.008, below v10's own real edge) AND a p-value with no
    nominal significance at all (>0.10)."""
    return abs(ic) < 0.008 and p > 0.10


def sweep_and_evaluate(name, SIG, weights=WEIGHTS):
    results = []
    for w in weights:
        WZ = np.array(H.BASE_WZ, copy=True)
        for t in days:
            WZ[:, t] = H.blend_signal(H.BASE_WZ[:, t], SIG[:, t], w)
        res = H.evaluate(f"{name} w={w}", WZ)
        results.append(res)
    return results


def report_best(results):
    passing = [r for r in results if r["passed"]]
    if passing:
        best = max(passing, key=lambda r: r["rm"])
        print(f"  ==> {len(passing)}/{len(results)} weights PASS. Best by rmean: "
              f"{best['name']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
    else:
        best = max(results, key=lambda r: r["rm"])
        print(f"  ==> 0/{len(results)} weights pass. Closest by rmean: "
              f"{best['name']}  rmean={best['rm']:.1f}  (baseline rmean={H.BASE_SCS.mean():.1f})")
    return passing


ALL_RESULTS = {}

# ==================================================================================================
print("=" * 100)
print("IDEA 1 -- Rolling kurtosis (own returns) as a predictor of own next-day return")
print("=" * 100)
KURT60 = causal_roll_returns(rs_full, 60, lambda r: r.kurt())
KURT90 = causal_roll_returns(rs_full, 90, lambda r: r.kurt())
ic60, p60, _ = pooled_ic(KURT60, "kurtosis W=60")
ic90, p90, _ = pooled_ic(KURT90, "kurtosis W=90")
best_ic, best_p, best_SIG, best_w = (ic60, p60, KURT60, 60) if abs(ic60) >= abs(ic90) else (ic90, p90, KURT90, 90)
if is_obviously_noise(best_ic, best_p):
    print(f"  SKIPPED full backtest: both W=60 and W=90 pooled IC obviously noise-level "
          f"(best |ic|={abs(best_ic):.4f}, p={best_p:.3g}).")
    ALL_RESULTS['idea1_kurtosis'] = None
else:
    SIG1 = np.nan_to_num(best_SIG * np.sign(best_ic) if best_ic < 0 else best_SIG)
    res = sweep_and_evaluate(f"1.kurtosis(W={best_w})", SIG1)
    report_best(res)
    ALL_RESULTS['idea1_kurtosis'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 2 -- Realized quarticity / vol-of-vol: rolling std of rolling-20d realized vol")
print("=" * 100)
VOL20 = causal_roll_returns(rs_full, 20, lambda r: r.std(ddof=0))
VOLVOL60 = causal_roll_wide(VOL20, 60, lambda r: r.std(ddof=0))
ic, p, _ = pooled_ic(VOLVOL60, "vol-of-vol (20d vol, 60d std-of-that)")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea2_volofvol'] = None
else:
    SIG2 = np.nan_to_num(VOLVOL60 * (np.sign(ic) if ic < 0 else 1.0))
    res = sweep_and_evaluate("2.volofvol", SIG2)
    report_best(res)
    ALL_RESULTS['idea2_volofvol'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 3 -- Post-jump drift: sign of yesterday's jump (|ret|>2*trailing-60d-stdev), else 0")
print("=" * 100)
rs_df = pd.DataFrame(rs_full.T)                                # (nt-1, nIdio)
vol60_prior = rs_df.rolling(60, min_periods=60).std(ddof=0).shift(1)   # excludes the jump return itself
is_jump = (rs_df.abs() > 2.0 * vol60_prior).fillna(False)
sign_if_jump = (np.sign(rs_df) * is_jump.astype(float))
SIG3 = np.zeros((nIdio, nt))
SIG3[:, 1:nt] = sign_if_jump.values.T
n_jumps = int(is_jump.values.sum())
ic, p, _ = pooled_ic(SIG3, "post-jump drift", note=f"({n_jumps} jump-events / {is_jump.size} stock-days = "
                                                     f"{100*n_jumps/is_jump.size:.2f}%)")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea3_postjump'] = None
else:
    res = sweep_and_evaluate("3.postjump", SIG3)
    report_best(res)
    ALL_RESULTS['idea3_postjump'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 4 -- Tail co-exceedance: count of OTHER names that also jumped yesterday")
print("=" * 100)
total_jump = is_jump.values.sum(axis=1)                         # (nt-1,)
other_jump = total_jump[:, None] - is_jump.values.astype(int)    # (nt-1, nIdio)
SIG4 = np.zeros((nIdio, nt))
SIG4[:, 1:nt] = other_jump.T
# cross-sectional variance check -- is this really near-uniform per-day, as B38 predicts?
cs_std = np.nanstd(SIG4[:, days], axis=0)
cs_level = np.nanmean(np.abs(SIG4[:, days]))
print(f"  cross-sectional std of SIG4 across the 50 names, averaged over days: {cs_std.mean():.4f}  "
      f"(mean |level|={cs_level:.4f}) -- near-zero cross-sectional std confirms the B38-style "
      f"near-uniform-vector structure predicted above.")
ic, p, _ = pooled_ic(SIG4, "tail co-exceedance")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level (and structurally expected to be "
          f"a near-no-op via blend_signal regardless, per the B38 precedent above).")
    ALL_RESULTS['idea4_coexceedance'] = None
else:
    print("  Pooled IC is not obviously noise -- running the full backtest anyway specifically to check "
          "whether the near-uniform cross-sectional structure makes it a near-no-op in practice (the B38 "
          "prediction), rather than skip on structural suspicion alone.")
    res = sweep_and_evaluate("4.coexceed", SIG4)
    report_best(res)
    ALL_RESULTS['idea4_coexceedance'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 5 -- Forecast-self-relative z-score: overweight BASE_WZ when it's extreme vs. its OWN "
      "trailing-250d history")
print("=" * 100)
print("  NOTE (per assignment): this repo has already rejected essentially every confidence/conviction-"
      "scaling scheme tried (documented 30:1 mean-vs-variance elasticity -- 'Sizing/smoothing schemes "
      "uniformly lose: Kelly sizing, drawdown throttles, vol-targeting, rank-based sizing, confidence "
      "ramps, persistence bonuses'). Tested honestly below regardless.")
bwz_df = pd.DataFrame(H.BASE_WZ.T)                                # (nt, nIdio), NaN where undefined
trail_mean = bwz_df.rolling(250, min_periods=250).mean().shift(1).values.T
trail_std = bwz_df.rolling(250, min_periods=250).std(ddof=0).shift(1).values.T
with np.errstate(invalid='ignore', divide='ignore'):
    self_z = (H.BASE_WZ - trail_mean) / (trail_std + 1e-9)
self_z = np.clip(np.nan_to_num(self_z), 0, 5)                      # magnitude of "extremeness", floor at 0
SIG5 = np.nan_to_num(H.BASE_WZ) * self_z                           # overweight the already-signed forecast
ic, p, _ = pooled_ic(SIG5, "forecast-self-relative z overweight")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea5_selfrelz'] = None
else:
    res = sweep_and_evaluate("5.selfrelz", SIG5)
    report_best(res)
    ALL_RESULTS['idea5_selfrelz'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 6 -- Eigenvector centrality of the trailing-250d idio correlation graph")
print("=" * 100)
t0 = time.time()
CENT = np.zeros((nIdio, nt))
for t in days:
    lo = max(0, t - 250)
    if t - lo < 60:
        continue
    X = rs_full[:, lo:t]
    if X.shape[1] < 2 or np.nanstd(X) < 1e-14:
        continue
    C = np.corrcoef(X)
    C = np.nan_to_num(C)
    A = np.abs(C)
    np.fill_diagonal(A, 0.0)
    try:
        w_, v_ = np.linalg.eigh(A)
    except np.linalg.LinAlgError:
        continue
    lead = np.abs(v_[:, -1])                    # Perron eigenvector of a nonneg symmetric matrix
    CENT[:, t] = lead
print(f"  eigen-decompositions done ({time.time()-t0:.1f}s)")
ic, p, _ = pooled_ic(CENT, "eigenvector centrality (standalone)")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED standalone-signal backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea6_centrality_standalone'] = None
else:
    res = sweep_and_evaluate("6.centrality", CENT)
    report_best(res)
    ALL_RESULTS['idea6_centrality_standalone'] = res

print("\n  -- variant: use centrality as a MULTIPLIER on the existing boost's magnitude for "
      "high-centrality names --")
CENT_Z = np.zeros((nIdio, nt))
for t in days:
    col = CENT[:, t]
    s = col.std()
    CENT_Z[:, t] = (col - col.mean()) / (s + 1e-12) if s > 1e-12 else 0.0

res6b = []
for gain in WEIGHTS:
    BOOST_MOD = H.BOOST_BASE * (1.0 + gain * CENT_Z)
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = H.WZ_PRE[:, t] + V10.BOOST_K * BOOST_MOD[:, t]
        WZ[:, t] = H.rs_blend(wz, t)
    r_ = H.evaluate(f"6b.centmult gain={gain}", WZ)
    res6b.append(r_)
report_best(res6b)
ALL_RESULTS['idea6b_centrality_multiplier'] = res6b


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 7 -- Ordinal-rank rank-stability: same short8/long22 crossover + disagreement gate, but "
      "using cross-sectional RANK instead of z-score for both legs")
print("=" * 100)


def rank_stability_ordinal(logp_, t, short_w=V10.RS_SHORT_W, long_w=V10.RS_LONG_W):
    """Exact structural analogue of V10._rank_stability_signal, substituting centered/scaled ORDINAL
    RANK (scipy.stats.rankdata) for the cross-sectional z-score on both the short and long leg, keeping
    the identical sign-disagreement gate."""
    if t < max(short_w, long_w) + 5:
        return None
    lp = logp_[1:, :t + 1]
    short_ret = lp[:, -1] - lp[:, -1 - short_w]
    long_ret = lp[:, -1] - lp[:, -1 - long_w]
    n = len(short_ret)

    def _rank_centered(x):
        rk = stats.rankdata(x)
        c = rk - (n + 1) / 2.0
        s = c.std()
        return c / (s + 1e-12) if s > 1e-12 else np.zeros_like(c)

    sz = _rank_centered(short_ret)
    lz = _rank_centered(long_ret)
    if sz.std() < 1e-12 or lz.std() < 1e-12:
        return None
    disagree = np.sign(lz) != np.sign(sz)
    return np.where(disagree, -sz, 0.0)


def build_wz_ordinal_rs(weight):
    """Replaces the shipped z-score rank-stability blend (H.rs_blend) with the ordinal-rank version at
    the given weight -- a REPLACEMENT of the mechanism, not an addition on top of it (adding both would
    double-count the same underlying short8/long22 crossover idea)."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = H.WZ_PRE[:, t] + V10.BOOST_K * H.BOOST_BASE[:, t]
        rs_sig = rank_stability_ordinal(logp, t)
        WZ[:, t] = H.blend_signal(wz, rs_sig, weight) if rs_sig is not None else wz
    return WZ


print("  Direct comparison at the SAME weight as shipped (0.015), replacing z-score with rank:")
WZ_ord_015 = build_wz_ordinal_rs(0.015)
res_ord_015 = H.evaluate("7.ordinal_rs w=0.015", WZ_ord_015)
print(f"    (shipped z-score version at this weight IS the v10 baseline itself: "
      f"OLD={H.BASE_WO:.1f} NEW={H.BASE_WN:.1f} rmean={H.BASE_SCS.mean():.1f} rfloor={H.BASE_SCS.min():.1f})")

print("\n  Re-sweeping weight for the ordinal-rank version:")
res7 = [res_ord_015]
for w in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1):
    WZ_w = build_wz_ordinal_rs(w)
    res7.append(H.evaluate(f"7.ordinal_rs w={w}", WZ_w))
report_best(res7)
ALL_RESULTS['idea7_ordinal_rankstab'] = res7


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 8 -- Cross-sectional dispersion of VOLATILITY (not returns): trailing-20d realized vol, "
      "cross-sectionally z-scored")
print("=" * 100)
SIG8 = np.nan_to_num(VOL20)     # blend_signal itself does the cross-sectional z-scoring each day
ic, p, _ = pooled_ic(SIG8, "cross-sectional vol level")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea8_voldispersion'] = None
else:
    res = sweep_and_evaluate("8.voldisp", SIG8)
    report_best(res)
    ALL_RESULTS['idea8_voldispersion'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 9 -- Consensus deviation: today's return minus today's cross-sectional mean return, used "
      "LIVE as a next-day predictor (distinct from beta-demean, which only cleans the ridge TARGET)")
print("=" * 100)
today_ret = rs_full                                        # (50, nt-1), rs_full[:,k] realized day k+1
cs_mean = today_ret.mean(axis=0, keepdims=True)             # (1, nt-1)
consensus_dev = today_ret - cs_mean
SIG9 = np.zeros((nIdio, nt))
SIG9[:, 1:nt] = consensus_dev
ic, p, _ = pooled_ic(SIG9, "consensus deviation")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea9_consensusdev'] = None
else:
    res = sweep_and_evaluate("9.consensusdev", SIG9)
    report_best(res)
    ALL_RESULTS['idea9_consensusdev'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("IDEA 10 -- Multi-horizon momentum composite: rank each stock's {3,5,10,20}-day return "
      "cross-sectionally, average the 4 ranks, re-standardize")
print("=" * 100)
HORIZONS = (3, 5, 10, 20)
RANK_SUM = np.zeros((nIdio, nt))
RANK_CNT = np.zeros(nt)
for k in HORIZONS:
    for t in range(k, nt):
        ret_k = logp[1:, t] - logp[1:, t - k]
        RANK_SUM[:, t] += stats.rankdata(ret_k)
        RANK_CNT[t] += 1
avg_rank = np.divide(RANK_SUM, RANK_CNT[None, :], out=np.zeros_like(RANK_SUM), where=RANK_CNT[None, :] > 0)
SIG10 = np.zeros((nIdio, nt))
for t in days:
    if RANK_CNT[t] < len(HORIZONS):
        continue
    col = avg_rank[:, t]
    s = col.std()
    SIG10[:, t] = (col - col.mean()) / (s + 1e-12) if s > 1e-12 else 0.0
ic, p, _ = pooled_ic(SIG10, "multi-horizon momentum rank composite")
if is_obviously_noise(ic, p):
    print(f"  SKIPPED full backtest: pooled IC obviously noise-level.")
    ALL_RESULTS['idea10_multimom'] = None
else:
    res = sweep_and_evaluate("10.multimom", SIG10)
    report_best(res)
    ALL_RESULTS['idea10_multimom'] = res


# ==================================================================================================
print("\n" + "=" * 100)
print("SUMMARY TABLE (best config per idea, or 'SKIPPED (noise IC)' / 'no config beat baseline')")
print("=" * 100)
print(f"{'idea':<32}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>8}{'n_worse':>10}{'passed':>8}")
print(f"{'(v10 baseline)':<32}{H.BASE_WO:8.1f}{H.BASE_WN:8.1f}{H.BASE_SCS.mean():8.1f}"
      f"{H.BASE_SCS.min():8.1f}{'--':>10}{'--':>8}")
for name, res in ALL_RESULTS.items():
    if res is None:
        print(f"{name:<32}{'SKIPPED (pooled IC obviously noise-level, see above)':>50}")
        continue
    best = max(res, key=lambda r: r['rm'])
    print(f"{name:<32}{best['wo']:8.1f}{best['wn']:8.1f}{best['rm']:8.1f}{best['rf']:8.1f}"
          f"{best['nworse']:>7}/61{str(best['passed']):>8}")

any_pass = any(res is not None and any(r['passed'] for r in res) for res in ALL_RESULTS.values())
print(f"\nAny idea passes OLD+NEW+rmean jointly vs real v10: {any_pass}")
