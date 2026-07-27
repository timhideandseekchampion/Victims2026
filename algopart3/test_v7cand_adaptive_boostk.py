"""
test_v7cand_adaptive_boostk.py

IDEA UNDER TEST: apply the ALGO leg's own validated adaptive-gain philosophy (size a signal by
its OWN trailing realized IC -- see IC_BLEND/_ic_ew and the older VOL_GAIN*ic direct-scaling
mechanism in SAFE_llboost_v6._algo_vol_shares) to the pairwise idio boost's overall STRENGTH
(BOOST_K), which today is a FIXED constant (1.5) regardless of how well the boost mechanism has
been doing lately in aggregate, realized trading.

Genuinely different from three previously-rejected pairwise ideas (do not re-litigate these):
  - margin-scaled boost (test_h4_margin_scaled_final.py): eff_k = K * min(margin, 2.0) where
    margin = |corr|/threshold -- a STRUCTURAL measure of significance at formation time, never
    updated by realized trading outcomes.
  - partial-pooling / empirical-Bayes shrinkage (test_partial_pooling_boost.py): blends each
    PAIR's own trailing IC with a population average, weighted by that PAIR's own sample size.
  - gated-pair-boost (test_gated_pair_boost.py): a per-pair trailing-IC on/off gate.
All three of the above operate PER PAIR. This candidate is a single PORTFOLIO-WIDE scalar
multiplier on BOOST_K, driven by how the boost mechanism performed IN AGGREGATE -- POOLED across
every qualifying (leader, follower) pair active on a given day -- over a trailing window.
Structurally this is the exact same "size the signal by its own realized trailing edge" idea
already validated for the ALGO leg's vol signal, never applied to the boost's own strength before.

FORMULA (fully causal -- day K's multiplier only uses pairs (t, j) with t < K):
  Precompute, at v6's shipped boost parameters (N=39, IC_L=250, MIN_DAY=480, SCALE_W=1000, P=2.0,
  reusing SAFE_llboost_v6._pairwise_boost verbatim -- no reimplementation risk), the raw boost
  value BOOST_AT[t][j] for every day t and every qualifying stock j, plus the realized one-step
  return rs[j, t] that the resulting position earns. BOOST_AT[t][j] is built from returns only
  through t-1 (see _pairwise_boost: lead_boost[-1] uses the leader's LAST available return), so
  pairing it with rs[j, t] (the very next return) is the same non-overlapping lag-1 alignment
  used throughout this repo (see test_partial_pooling_boost.py's "must use b[t-1] not b[t]" fix).

  trailing_ic(K, L)  = pooled Pearson corr of all (x=BOOST_AT[t][j], y=rs[j,t]) with t in
                        [K-L, K)  -- the last L days.
  reference_ic(K, L) = pooled Pearson corr of the SAME quantities with t in
                        [BOOST_MIN_DAY, K-L) -- ALL prior boost-eligible history strictly BEFORE
                        the trailing window (expanding, non-overlapping with it -- itself fully
                        causal, no fixed magic constant to guess as "the normal edge size").
  ratio    = trailing_ic / max(reference_ic, REF_FLOOR)   (REF_FLOOR avoids blow-up dividing near
             zero; a negative/weak reference floors to REF_FLOOR so a genuinely positive
             trailing_ic still yields a well-defined, capped ratio)
  eff_k(K) = BOOST_K * clip(ratio, 0, CAP)

  Both windows require >= MIN_POOL_N pooled (t,j) observations; if either doesn't have enough yet
  (early in the file, before BOOST_MIN_DAY + L + slack), eff_k(K) = BOOST_K exactly -- identical
  to v6, the same "no adjustment until proven" default philosophy as BOOST_MIN_DAY's own gate.
  A negative/weak recent trailing_ic floors the multiplier at 0 (boost shuts OFF, never flips
  sign) -- same "floor-not-flip" philosophy as the ALGO leg's max(0,ic) term and the boost's own
  per-pair ic<=0 rejection gate.

Also tested a FIXED-reference variant (reference_ic = a constant, not the expanding baseline
above) as an explicit "reference scale" robustness axis (mode="fixed").

Sweep: L (lookback) in {60,90,120,180} x CAP in {1.5,2.0,2.5,3.0} x mode in
{expanding, fixed(ref in {0.02,0.05,0.08})}. MIN_POOL_N=100 fixed throughout.

Scoring convention matches validate_llboost_v6_full.py exactly: window(POS,S,E), commRate=1e-4
(inst0=2e-5), dlr=10_000 (inst0=100_000), score=mu*sr^2/(sr^2+1). OLD=window(500,750),
NEW=window(750,nt), rolling mean/floor over end_days=range(400,nt+1,10) (61 windows). The FINAL
comparison (bottom of this script) is against the REAL SAFE_llboost_v6.getMyPosition output
(exact, not a backtest approximation) -- everything above uses a fast backtest-equivalent
reconstruction (WZ + _pairwise_boost + _algo_vol_shares, identical building blocks) purely to make
the ~32-point sweep tractable; the winning config is then re-validated end-to-end on the real
module.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v6 as V6

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)          # (nInst, nt-1)
rs = r[1:]                          # (49, nt-1)  idio-only, matches V6._pairwise_boost's input

BOOST_K = V6.BOOST_K                # 1.5
BOOST_MIN_DAY = V6.BOOST_MIN_DAY    # 480
WARMUP = V6.WARMUP                  # 96


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<46}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line, flush=True)
    return scs


# ================================================================================================
# 1) shared precompute: shipped ridge+blend WZ forecast, v6 ALGO leg -- ONCE
# ================================================================================================
print("=== precompute: ridge+blend WZ forecast (v6-identical) ===")
t0 = time.time()
WZ = {}
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V6.HALF_LIVES:
        B, mx, my = V6._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V6.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V6.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - V6.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - V6.BLEND) * wz + V6.BLEND * rv
    WZ[t] = wz
print(f"  done ({time.time()-t0:.0f}s)")

print("=== precompute: v6 ALGO leg (MOM_LB_SHORT=7/LONG=12 vol-regime-adaptive) ===")
t0 = time.time()
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V6._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)")

# ================================================================================================
# 2) boost candidate map at v6's shipped parameters -- ONCE (reuse V6._pairwise_boost verbatim,
#    so the qualifying pairs are guaranteed byte-identical to the shipped mechanism -- only the
#    scaling of BOOST_K changes downstream)
# ================================================================================================
print("=== precompute: v6 boost candidate map (N=39, IC_L=250, MIN_DAY=480, SCALE_W=1000, P=2.0) ===")
t0 = time.time()
BOOST_AT = {}   # day k -> length-49 raw boost array (0.0 where no qualifying leader)
for k in range(BOOST_MIN_DAY, nt):
    BOOST_AT[k] = V6._pairwise_boost(rs[:, :k])
print(f"  done ({time.time()-t0:.0f}s, {nt-BOOST_MIN_DAY} days)")

# sanity: reproduce v6's own baseline (fixed BOOST_K) via this backtest-equivalent path
def build_pos_fixed(k_boost):
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= BOOST_MIN_DAY:
            wz = wz + k_boost * BOOST_AT[k]
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: backtest-equivalent reconstruction vs shipped v6 numbers (774.1/828.6.../857.0 etc, see README) ===")
base_scs = report("v6 (fixed BOOST_K=1.5, backtest-equiv)", build_pos_fixed(BOOST_K))

# ================================================================================================
# 3) adaptive trailing-performance multiplier on BOOST_K
# ================================================================================================
print("\n=== building per-day pooled (x=raw boost, y=realized rs[j,t]) aggregates for the multiplier ===")
t0 = time.time()
days_lo, days_hi = BOOST_MIN_DAY, nt - 1  # need rs[:,t] to exist -> t <= nt-2
n_days = days_hi - days_lo
n_t = np.zeros(n_days); sx_t = np.zeros(n_days); sy_t = np.zeros(n_days)
sxx_t = np.zeros(n_days); syy_t = np.zeros(n_days); sxy_t = np.zeros(n_days)
for idx, t in enumerate(range(days_lo, days_hi)):
    bv = BOOST_AT[t]
    active = bv != 0.0
    if not active.any():
        continue
    x = bv[active]
    y = rs[active, t]
    n_t[idx] = x.size
    sx_t[idx] = x.sum(); sy_t[idx] = y.sum()
    sxx_t[idx] = (x * x).sum(); syy_t[idx] = (y * y).sum(); sxy_t[idx] = (x * y).sum()
# prefix sums (cn[i] = cumulative up to day index i, exclusive of i itself handled via slicing)
cn = np.concatenate(([0.0], np.cumsum(n_t)))
csx = np.concatenate(([0.0], np.cumsum(sx_t)))
csy = np.concatenate(([0.0], np.cumsum(sy_t)))
csxx = np.concatenate(([0.0], np.cumsum(sxx_t)))
csyy = np.concatenate(([0.0], np.cumsum(syy_t)))
csxy = np.concatenate(([0.0], np.cumsum(sxy_t)))
print(f"  done ({time.time()-t0:.0f}s); total qualifying pair-days = {int(n_t.sum())}")


def _pooled_ic(a_day, b_day):
    """Pooled Pearson corr over all qualifying (t,j) with t in [a_day, b_day) -- causal, uses only
    the per-day aggregates above. Returns (ic, n) or (None, n) if variance is degenerate."""
    a = max(0, a_day - days_lo); b = max(0, min(b_day, days_hi) - days_lo)
    if b <= a:
        return None, 0
    N = cn[b] - cn[a]
    if N < 1:
        return None, 0
    SX = csx[b] - csx[a]; SY = csy[b] - csy[a]
    SXX = csxx[b] - csxx[a]; SYY = csyy[b] - csyy[a]; SXY = csxy[b] - csxy[a]
    mx = SX / N; my = SY / N
    vx = SXX / N - mx * mx; vy = SYY / N - my * my
    if vx <= 1e-24 or vy <= 1e-24:
        return None, int(N)
    cov = SXY / N - mx * my
    return float(cov / np.sqrt(vx * vy)), int(N)


MIN_POOL_N = 100


def make_multiplier_expanding(L, CAP, ref_floor=0.02, min_pool_n=MIN_POOL_N):
    """eff_k(k)/BOOST_K lookup array, mode='expanding' (reference = all prior history before the
    trailing window)."""
    mult = np.ones(nt)  # default: no adjustment (identical to v6) until proven
    for k in range(BOOST_MIN_DAY, nt):
        tr_ic, tr_n = _pooled_ic(k - L, k)
        ref_ic, ref_n = _pooled_ic(BOOST_MIN_DAY, k - L)
        if tr_ic is None or ref_ic is None or tr_n < min_pool_n or ref_n < min_pool_n:
            continue
        ratio = tr_ic / max(ref_ic, ref_floor)
        mult[k] = float(np.clip(ratio, 0.0, CAP))
    return mult


def make_multiplier_fixed(L, CAP, ref_const, min_pool_n=MIN_POOL_N):
    """eff_k(k)/BOOST_K lookup array, mode='fixed' (reference = a constant, not expanding)."""
    mult = np.ones(nt)
    for k in range(BOOST_MIN_DAY, nt):
        tr_ic, tr_n = _pooled_ic(k - L, k)
        if tr_ic is None or tr_n < min_pool_n:
            continue
        ratio = tr_ic / ref_const
        mult[k] = float(np.clip(ratio, 0.0, CAP))
    return mult


def build_pos_adaptive(mult):
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= BOOST_MIN_DAY:
            eff_k = BOOST_K * mult[k]
            wz = wz + eff_k * BOOST_AT[k]
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n### sweep: mode='expanding' (reference = all prior boost-eligible history, causal) ###")
results = []
for L in (60, 90, 120, 180):
    for CAP in (1.5, 2.0, 2.5, 3.0):
        mult = make_multiplier_expanding(L, CAP)
        POS = build_pos_adaptive(mult)
        wo = window(POS, *OLD); wn = window(POS, *NEW)
        nm = f"expanding L={L:<4} CAP={CAP}"
        scs = report(nm, POS, base_scs)
        results.append(("expanding", L, CAP, None, wo, wn, scs))

print("\n### sweep: mode='fixed' (reference = a constant, explicit 'reference scale' axis) ###")
for L in (60, 90, 120, 180):
    for CAP in (1.5, 2.0, 2.5, 3.0):
        for REF in (0.02, 0.05, 0.08):
            mult = make_multiplier_fixed(L, CAP, REF)
            POS = build_pos_adaptive(mult)
            wo = window(POS, *OLD); wn = window(POS, *NEW)
            nm = f"fixed L={L:<4} CAP={CAP} ref={REF}"
            scs = report(nm, POS, base_scs)
            results.append(("fixed", L, CAP, REF, wo, wn, scs))

# ================================================================================================
# pick best-looking candidate(s) on OLD+NEW+rolling-mean jointly, then check neighbor stability
# ================================================================================================
print("\n=== ranking candidates: require improvement (vs backtest-equiv v6 baseline) on ALL of OLD, NEW, rolling-mean ===")
base_wo = window(build_pos_fixed(BOOST_K), *OLD)
base_wn = window(build_pos_fixed(BOOST_K), *NEW)
base_rm = base_scs.mean()
print(f"baseline: OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_rm:.1f} rfloor={base_scs.min():.1f}")

candidates = []
for mode, L, CAP, REF, wo, wn, scs in results:
    rm = scs.mean(); rf = scs.min()
    nworse = int((scs < base_scs).sum())
    passed = (wo > base_wo) and (wn > base_wn) and (rm > base_rm)
    candidates.append(dict(mode=mode, L=L, CAP=CAP, REF=REF, wo=wo, wn=wn, rm=rm, rf=rf,
                           nworse=nworse, passed=passed))
    if passed:
        print(f"  PASS  mode={mode:<10} L={L:<4} CAP={CAP:<4} ref={REF}  "
              f"OLD={wo:.1f} NEW={wn:.1f} rmean={rm:.1f} rfloor={rf:.1f} n_worse={nworse}/61")

passing = [c for c in candidates if c["passed"]]
print(f"\n{len(passing)}/{len(candidates)} configs beat baseline on OLD+NEW+rmean jointly.")

if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"\nBest passing config by rolling mean: mode={best['mode']} L={best['L']} CAP={best['CAP']} "
          f"ref={best['REF']}  OLD={best['wo']:.1f} NEW={best['wn']:.1f} rmean={best['rm']:.1f} "
          f"rfloor={best['rf']:.1f} n_worse={best['nworse']}/61")

    print("\n=== neighbor-stability check around the best config ===")
    L0, CAP0 = best["L"], best["CAP"]
    Ls = sorted(set([60, 90, 120, 180]))
    Ls_near = [x for x in Ls if abs(Ls.index(x) - Ls.index(L0)) <= 1]
    CAPs = sorted(set([1.5, 2.0, 2.5, 3.0]))
    CAPs_near = [x for x in CAPs if abs(CAPs.index(x) - CAPs.index(CAP0)) <= 1]
    for L in Ls_near:
        for CAP in CAPs_near:
            if best["mode"] == "expanding":
                mult = make_multiplier_expanding(L, CAP)
            else:
                mult = make_multiplier_fixed(L, CAP, best["REF"])
            POS = build_pos_adaptive(mult)
            scs = scs_curve(POS)
            wo = window(POS, *OLD); wn = window(POS, *NEW)
            nworse = int((scs < base_scs).sum())
            tag = " <== best" if (L == L0 and CAP == CAP0) else ""
            print(f"  mode={best['mode']:<10} L={L:<4} CAP={CAP:<4} ref={best['REF']}  "
                  f"OLD={wo:.1f} NEW={wn:.1f} rmean={scs.mean():.1f} rfloor={scs.min():.1f} "
                  f"n_worse={nworse}/61{tag}")

    # ============================================================================================
    # 4/5) FINAL validation vs the REAL SAFE_llboost_v6.getMyPosition (exact, not approximated)
    # ============================================================================================
    print("\n=== FINAL: validate best config's positions vs the REAL SAFE_llboost_v6.getMyPosition ===")
    print("building REAL SAFE_llboost_v6 positions (exact getMyPosition, not backtest-equivalent) ...")
    t0 = time.time()
    FIRST_DAY = 148  # covers every rolling window, same convention as validate_llboost_v6_full.py
    POS_v6_real = np.zeros((nInst, nt))
    for k in range(FIRST_DAY, nt):
        POS_v6_real[:, k] = V6.getMyPosition(P_[:, :k + 1])
    print(f"  done ({time.time()-t0:.0f}s)")

    real_scs = report("REAL SAFE_llboost_v6 (getMyPosition, exact)", POS_v6_real)

    if best["mode"] == "expanding":
        mult_best = make_multiplier_expanding(L0, CAP0)
    else:
        mult_best = make_multiplier_fixed(L0, CAP0, best["REF"])
    POS_best = build_pos_adaptive(mult_best)
    report(f"BEST candidate ({best['mode']}, L={L0}, CAP={CAP0}, ref={best['REF']}) vs REAL v6",
           POS_best, real_scs)
else:
    print("\nNo config beat the (backtest-equivalent) baseline on OLD+NEW+rmean jointly -- "
          "skipping neighbor-stability check and real-getMyPosition validation "
          "(nothing here clears the bar to be worth re-validating exactly).")
    print("\n=== reference only: REAL SAFE_llboost_v6.getMyPosition score (exact) ===")
    FIRST_DAY = 148
    POS_v6_real = np.zeros((nInst, nt))
    for k in range(FIRST_DAY, nt):
        POS_v6_real[:, k] = V6.getMyPosition(P_[:, :k + 1])
    report("REAL SAFE_llboost_v6 (getMyPosition, exact)", POS_v6_real)

print("\nVERDICT to be written up from the numbers above.")
