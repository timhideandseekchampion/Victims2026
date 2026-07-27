"""
test_v7cand_joint_search.py -- genuine JOINT local search around SAFE_llboost_v6's operating
point, instead of the sequential/coordinate-wise tuning that actually produced it.

Every one of v6's parameters was tuned ONE AT A TIME, holding the others fixed at whatever they
happened to be at that point in the session's history:
  - BOOST_K=1.5, BOOST_P=2.0, BOOST_SCALE_W=1000 were tuned first, when the boost candidate pool
    was still N=49 (before N=39 existed).
  - BOOST_N_CANDIDATES=39 (v3) was chosen next, holding BOOST_K/P/SCALE_W fixed at their pre-N=39
    values.
  - BOOST_IC_L=250 and BOOST_MIN_DAY=480 (v5) were re-tuned after that, holding N=39 and the
    original K/P/SCALE_W fixed.
  - MOM_LB_SHORT=7 / MOM_LB_LONG=12 (v2's ALGO-leg regime-adaptive momentum) was tuned separately,
    on the ALGO leg ALONE, then just structurally bolted onto v5's boost (v6) -- never re-tuned
    jointly with the final boost config.
  - COMBINE_GAIN=3.5 was tuned even earlier than any of the above, before any boost existed at all.
Nobody has varied several of these SIMULTANEOUSLY now that the full v6 stack exists, to check
whether today's config is a true joint local optimum or whether the sequential/coordinate-wise
process left cross-parameter-interaction gains on the table.

Swept jointly (4*5*3*4*3 = 720 combos, FULL grid, no narrowing):
  BOOST_K       in {1.25, 1.5, 1.75, 2.0}        (v6 shipped: 1.5)
  BOOST_IC_L    in {200, 225, 250, 275, 300}     (v6 shipped: 250)
  MOM_LB_SHORT  in {6, 7, 8}                     (v6 shipped: 7)
  MOM_LB_LONG   in {11, 12, 13, 14}              (v6 shipped: 12)
  COMBINE_GAIN  in {3.0, 3.5, 4.0}               (v6 shipped: 3.5)
Everything else held at v6's shipped defaults: BOOST_N_CANDIDATES=39, BOOST_MIN_DAY=480,
BOOST_SCALE_W=1000, BOOST_P=2.0, SWITCH_GAIN=2.5, VOL_WIN=20, VOL_Z=60, IC_FAST=90,
IC_EW_HL=(20,45), IC_EW_W=200, VOL_MODE="switch", VOL_COMBINE=True, IC_BLEND=True.

Precompute strategy (share everything that does NOT depend on the swept 5D grid):
  1. WZ_ARR[:, k]  -- the ridge+blend idio forecast (identical to SAFE_llboost_v6's own idio
     ridge). Depends on NONE of the 5 swept params -- computed exactly once, using v6's own
     _ewls_ridge/HALF_LIVES/RIDGE_A/BLEND/REV_W/WARMUP (imported directly, not re-typed).
  2. BOOST_RAW[ic_l][:, k]  -- the RAW (pre-BOOST_K) pairwise-boost value per stock/day, for each
     of the 5 BOOST_IC_L values. N=39/MIN_DAY=480/SCALE_W=1000/P=2.0 are fixed at v6's shipped
     values, so BOOST_IC_L is the only thing that changes what gets computed here; BOOST_K only
     rescales the result afterward (wz += BOOST_K * BOOST_RAW[ic_l]), so the (expensive) map is
     never rebuilt per-K -- built once per IC_L value (5x, not 20x).
  3. ALGO_POS[(short, long, gain)]  -- the ALGO leg (instrument 0) integer position series, for
     each of the 3*4*3=36 (MOM_LB_SHORT, MOM_LB_LONG, COMBINE_GAIN) combos. Independent of
     BOOST_K/BOOST_IC_L entirely.
Building any one of the 720 full position matrices from these three precomputed pieces is then
pure vectorized array algebra (sign + clip), no re-fitting -- confirmed by a sanity check that the
fast-built matrix for v6's own shipped combo is BIT-IDENTICAL to positions built by calling the
real, production SAFE_llboost_v6.getMyPosition directly (max abs diff printed below).

Ranking protocol (per task spec -- full 61-window rolling report is too expensive for all 720):
  (a) score OLD=(500,750) and NEW=(750,nt) only (cheap: 2 windows) for ALL 720 combos;
  (b) rank by OLD+NEW; take the top 10;
  (c) for those top 10 ONLY, compute the full 61-window rolling mean/floor and n_worse vs the
      ACTUAL shipped SAFE_llboost_v6.getMyPosition baseline (imported directly, real positions
      built day by day -- not a backtest approximation);
  (d) neighbor-stability check around whatever wins, to distinguish a genuine plateau from an
      isolated spike (same diagnostic v2/v3/v5's own validations used).

CAUSAL ONLY: every per-day quantity (ridge fit, boost leader search, ALGO vol/momentum signal)
uses only price history up to and including that day -- identical walk-forward structure to
SAFE_llboost_v6.getMyPosition itself (this is exactly what the bit-identical sanity check
verifies).
"""
import itertools
import time

import numpy as np
import pandas as pd

import SAFE_llboost_v6 as V6

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------ data / scoring convention --
P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
n_idio = nInst - 1
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)          # (nInst, nt-1) -- ALGO included, used as ridge predictors
rs = r[1:]                          # (n_idio, nt-1) -- idio-only returns (ALGO excluded)

log(f"loaded prices.txt: nInst={nInst}, n_idio={n_idio}, nt={nt}")


def score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    """Identical convention to validate_llboost_v6_full.py: commRate=1e-4 (inst0=2e-5),
    dlr=10_000 (inst0=100_000), score=mu*sr^2/(sr^2+1)."""
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
assert len(end_days) == 61, f"expected 61 rolling windows, got {len(end_days)}"


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


# ============================================================== 1. shared precompute: WZ ridge --
log("precompute 1/3: ridge+blend idio forecast WZ_ARR (shared across ALL 720 combos) ...")
WZ_ARR = np.zeros((n_idio, nt))
for k in range(V6.WARMUP, nt):
    rr = r[:, :k]
    fs = []
    for hl in V6.HALF_LIVES:
        B, mx, my = V6._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V6.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V6.BLEND > 0:
        rv_ = logp[1:, k] - logp[1:, k - V6.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - V6.BLEND) * wz + V6.BLEND * rv
    WZ_ARR[:, k] = wz
log("  WZ_ARR done")

# per-day, per-instrument integer share limit dlr/cur (floor) -- identical clip used in production
lim_all = (dlr[:, None] / P_).astype(int)

# ================================================== 2. shared precompute: raw boost per IC_L -----
BOOST_N = 39          # BOOST_N_CANDIDATES, v6 shipped (fixed, not swept)
BOOST_MIN_DAY = 480    # v6 shipped (fixed, not swept)
BOOST_SCALE_W = 1000   # v6 shipped (fixed, not swept)
BOOST_P = 2.0          # v6 shipped (fixed, not swept)
IC_L_GRID = [200, 225, 250, 275, 300]

log("precompute 2/3: raw pairwise-boost value per BOOST_IC_L (5 values; BOOST_K applied later) ...")
BOOST_RAW = {}
for ic_l in IC_L_GRID:
    t0 = time.time()
    arr = np.zeros((n_idio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        T = k
        Xi_full = rs[:, :T - 1]; Yj = rs[:, 1:T]
        n_samples = Xi_full.shape[1]
        thr = V6._sig_threshold(n_samples)     # uses V6.BOOST_N_CANDIDATES=39 -- matches BOOST_N
        vol_causal = np.nanstd(Xi_full, axis=1)
        cand_idx = np.argsort(-vol_causal)[:BOOST_N]
        Xi = Xi_full[cand_idx]
        C = V6._corrmat(Xi, Yj)
        for j in range(n_idio):
            col = C[:, j].copy()
            cp = np.where(cand_idx == j)[0]
            if len(cp):
                col[cp[0]] = np.nan
            if np.all(np.isnan(col)):
                continue
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) <= thr:
                continue
            i = cand_idx[ci]
            lead = rs[i, :T]
            scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
            lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
            a = max(0, T - 1 - ic_l)
            xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            if ok.sum() < 60 or xs[ok].std() < 1e-12:
                continue
            ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
            if ic <= 0:
                continue
            arr[j, k] = lead_boost[-1]
    BOOST_RAW[ic_l] = arr
    log(f"  BOOST_RAW[ic_l={ic_l}] done in {time.time() - t0:.1f}s (nnz={int((arr != 0).sum())})")

# ================================================== 3. shared precompute: ALGO leg per combo -----
VOL_WIN, VOL_Z, IC_FAST, SWITCH_GAIN = V6.VOL_WIN, V6.VOL_Z, V6.IC_FAST, V6.SWITCH_GAIN
IC_EW_HL, IC_EW_W, IC_LOOKBACK = V6.IC_EW_HL, V6.IC_EW_W, V6.IC_LOOKBACK

SHORT_GRID = [6, 7, 8]
LONG_GRID = [11, 12, 13, 14]
GAIN_GRID = [3.0, 3.5, 4.0]


def algo_leg_flex(lpA, cur0, cap_dol, mom_short, mom_long, combine_gain):
    """Same logic as V6._algo_vol_shares (VOL_MODE='switch', VOL_COMBINE=True, IC_BLEND=True,
    all fixed/shipped), but with MOM_LB_SHORT/MOM_LB_LONG/COMBINE_GAIN as explicit arguments
    instead of module globals, so a single function serves the whole 36-combo precompute."""
    T = len(lpA)
    if T < VOL_WIN + VOL_Z + 60:
        return 0
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan); vol[VOL_WIN:] = V6._roll_std(r_, VOL_WIN)
    tnow = T - 1
    lo = max(VOL_WIN + VOL_Z, tnow - IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]

    def _ic(feat, L):
        a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60:
            return None
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-12:
            return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _ic_ew(feat, HL, W):
        a = max(0, tnow - W); xs = feat[a:tnow]; ys = ret1[a:tnow]
        w = (0.5 ** (1.0 / HL)) ** ((tnow - 1) - np.arange(a, tnow))
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60:
            return None
        xs, ys, w = xs[ok], ys[ok], w[ok]; sw = w.sum()
        mx = (w * xs).sum() / sw; my = (w * ys).sum() / sw
        cxy = (w * (xs - mx) * (ys - my)).sum() / sw
        vx = (w * (xs - mx) ** 2).sum() / sw; vy = (w * (ys - my) ** 2).sum() / sw
        if vx < 1e-24 or vy < 1e-24:
            return None
        return float(cxy / np.sqrt(vx * vy))

    def _side(feat, fhv):
        icf = _ic(feat, IC_FAST)
        if icf is None:
            return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, IC_EW_W) for hl in IC_EW_HL]
        if any(x is None for x in ics):
            return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        return 0
    sig = _side(volz, fh)
    if sig is None:
        return 0

    mom_lb = mom_short if fh > 0 else mom_long
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T):
        wm = mom[s - VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        av = combine_gain * (sig + msig) * 100_000.0
    else:
        av = SWITCH_GAIN * sig * 100_000.0
    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


log("precompute 3/3: ALGO leg position series for each of 36 (short,long,gain) combos ...")
ALGO_POS = {}
for mom_short, mom_long, gain in itertools.product(SHORT_GRID, LONG_GRID, GAIN_GRID):
    t0 = time.time()
    algo_pos = np.zeros(nt)
    for k in range(100, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(
            algo_leg_flex(logp[0, :k + 1], cur0, dlr[0], mom_short, mom_long, gain), -lim0, lim0)
    ALGO_POS[(mom_short, mom_long, gain)] = algo_pos
log(f"  ALGO_POS done (36 combos, {time.time() - T0:.1f}s elapsed total)")


def build_pos(k_boost, ic_l, mom_short, mom_long, gain):
    wz = WZ_ARR + k_boost * BOOST_RAW[ic_l]
    idio_sign = np.sign(wz)
    POS = np.zeros((nInst, nt))
    POS[1:, :] = idio_sign * lim_all[1:, :]
    POS[0, :] = ALGO_POS[(mom_short, mom_long, gain)]
    return POS


# ============================================================ 4. sanity: bit-identical to real v6
log("sanity check: fast-built v6-shipped-combo POS vs real SAFE_llboost_v6.getMyPosition ...")
SHIP = dict(k_boost=1.5, ic_l=250, mom_short=7, mom_long=12, gain=3.5)
POS_FAST_SHIP = build_pos(**SHIP)

FIRST_DAY = 148  # covers every rolling window (earliest need: end_day=400 -> S=150 -> POS index 149)
t0 = time.time()
POS_V6_REAL = np.zeros((nInst, nt))
for k in range(FIRST_DAY, nt):
    POS_V6_REAL[:, k] = V6.getMyPosition(P_[:, :k + 1])
log(f"  real SAFE_llboost_v6 positions built in {time.time() - t0:.1f}s")

diff = np.abs(POS_FAST_SHIP[:, FIRST_DAY:] - POS_V6_REAL[:, FIRST_DAY:])
log(f"  max abs diff (fast-build vs real getMyPosition), day>={FIRST_DAY}: {diff.max():.6g}  "
    f"(n_nonzero_diff_days={int((diff.sum(0) > 0).sum())})")

base_scs = scs_curve(POS_V6_REAL)
wo0 = window(POS_V6_REAL, *OLD); wn0 = window(POS_V6_REAL, *NEW)
log(f"  real SAFE_llboost_v6: OLD={wo0:.1f} NEW={wn0:.1f} rmean={base_scs.mean():.1f} "
    f"rfloor={base_scs.min():.1f}  (docstring: 811.4/868.9/857.0/669.5)")

fast_scs_ship = scs_curve(POS_FAST_SHIP)
wo0f = window(POS_FAST_SHIP, *OLD); wn0f = window(POS_FAST_SHIP, *NEW)
log(f"  fast-built (same combo): OLD={wo0f:.1f} NEW={wn0f:.1f} rmean={fast_scs_ship.mean():.1f} "
    f"rfloor={fast_scs_ship.min():.1f}")

# ================================================================ 5. full 720-combo joint grid --
K_GRID = [1.25, 1.5, 1.75, 2.0]
GRID = list(itertools.product(K_GRID, IC_L_GRID, SHORT_GRID, LONG_GRID, GAIN_GRID))
log(f"joint grid: {len(K_GRID)}*{len(IC_L_GRID)}*{len(SHORT_GRID)}*{len(LONG_GRID)}*{len(GAIN_GRID)}"
    f" = {len(GRID)} combos -- scoring OLD+NEW (cheap) for every one ...")

t0 = time.time()
cheap_results = []
for (k_boost, ic_l, mom_short, mom_long, gain) in GRID:
    POS = build_pos(k_boost, ic_l, mom_short, mom_long, gain)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    cheap_results.append({
        "k_boost": k_boost, "ic_l": ic_l, "mom_short": mom_short, "mom_long": mom_long,
        "gain": gain, "OLD": wo, "NEW": wn, "OLD_NEW": wo + wn,
    })
log(f"  all {len(GRID)} combos scored (OLD+NEW) in {time.time() - t0:.1f}s")

cheap_results.sort(key=lambda d: -d["OLD_NEW"])
ship_rank = next(i for i, d in enumerate(cheap_results)
                  if (d["k_boost"], d["ic_l"], d["mom_short"], d["mom_long"], d["gain"])
                  == (SHIP["k_boost"], SHIP["ic_l"], SHIP["mom_short"], SHIP["mom_long"], SHIP["gain"]))
log(f"  v6 shipped combo rank by OLD+NEW: {ship_rank + 1}/{len(GRID)} "
    f"(OLD+NEW={cheap_results[ship_rank]['OLD_NEW']:.1f}, shipped OLD={wo0f:.1f} NEW={wn0f:.1f})")

print("\ntop 15 combos by cheap OLD+NEW ranking:")
print(f"{'rank':>4} {'K':>5} {'IC_L':>5} {'MOM_S':>6} {'MOM_L':>6} {'GAIN':>5} "
      f"{'OLD':>8} {'NEW':>8} {'OLD+NEW':>9}")
for i, d in enumerate(cheap_results[:15]):
    print(f"{i+1:>4} {d['k_boost']:>5} {d['ic_l']:>5} {d['mom_short']:>6} {d['mom_long']:>6} "
          f"{d['gain']:>5} {d['OLD']:>8.1f} {d['NEW']:>8.1f} {d['OLD_NEW']:>9.1f}")

# =================================================== 6. full 61-window report for the TOP 10 -----
TOP_N = 10
log(f"\nfull 61-window rolling report for the top {TOP_N} candidates (by cheap OLD+NEW) ...")
full_results = []
for d in cheap_results[:TOP_N]:
    POS = build_pos(d["k_boost"], d["ic_l"], d["mom_short"], d["mom_long"], d["gain"])
    scs = scs_curve(POS)
    nworse = int((scs < base_scs).sum())
    full_results.append({**d, "rmean": float(scs.mean()), "rfloor": float(scs.min()),
                          "nworse": nworse, "scs": scs})

print(f"\n{'='*100}")
print(f"FULL comparison (top {TOP_N} by cheap OLD+NEW ranking) vs real shipped SAFE_llboost_v6:")
print(f"{'='*100}")
print(f"{'label':<34} {'K':>5} {'IC_L':>5} {'MOM_S':>6} {'MOM_L':>6} {'GAIN':>5} "
      f"{'OLD':>8} {'NEW':>8} {'rmean':>8} {'rfloor':>8} {'n_worse':>9}")
print(f"{'SAFE_llboost_v6 (shipped, real)':<34} {'1.5':>5} {'250':>5} {'7':>6} {'12':>6} {'3.5':>5} "
      f"{wo0:>8.1f} {wn0:>8.1f} {base_scs.mean():>8.1f} {base_scs.min():>8.1f} {'--':>9}")
for i, d in enumerate(full_results):
    label = f"cand #{i+1} (rank {cheap_results.index(d)+1 if d in cheap_results else '?'})"
    print(f"{label:<34} {d['k_boost']:>5} {d['ic_l']:>5} {d['mom_short']:>6} {d['mom_long']:>6} "
          f"{d['gain']:>5} {d['OLD']:>8.1f} {d['NEW']:>8.1f} {d['rmean']:>8.1f} {d['rfloor']:>8.1f} "
          f"{d['nworse']:>6}/61")

# beats-v6-on-all-four-metrics-simultaneously check
print("\ncandidates (of the top 10) that beat v6 shipped on OLD AND NEW AND rmean AND rfloor "
      "simultaneously:")
any_clean_beat = False
for i, d in enumerate(full_results):
    beats = (d["OLD"] > wo0) and (d["NEW"] > wn0) and (d["rmean"] > base_scs.mean()) and \
            (d["rfloor"] >= base_scs.min())
    if beats:
        any_clean_beat = True
        print(f"  cand #{i+1}: K={d['k_boost']} IC_L={d['ic_l']} MOM_S={d['mom_short']} "
              f"MOM_L={d['mom_long']} GAIN={d['gain']} -> OLD={d['OLD']:.1f} NEW={d['NEW']:.1f} "
              f"rmean={d['rmean']:.1f} rfloor={d['rfloor']:.1f} n_worse={d['nworse']}/61")
if not any_clean_beat:
    print("  none -- no top-10 candidate clears all four headline metrics simultaneously.")

# ================================================== 7. neighbor-stability check on the winner -----
best = max(full_results, key=lambda d: d["rmean"])
best_key = (best["k_boost"], best["ic_l"], best["mom_short"], best["mom_long"], best["gain"])
ship_key = (SHIP["k_boost"], SHIP["ic_l"], SHIP["mom_short"], SHIP["mom_long"], SHIP["gain"])

print(f"\n{'='*100}")
if best_key == ship_key:
    print("VERDICT PRECURSOR: the best rolling-mean candidate among the top 10 by cheap OLD+NEW "
          "ranking IS v6's own shipped combo -- v6 sits at (or ties) the top of the ranking already.")
else:
    print(f"Best-by-rolling-mean candidate differs from v6 shipped: {best_key} vs {ship_key}")
    print("Running a neighbor-stability check around this candidate (one grid-step away in each "
          "of the 5 swept dimensions, others held at the candidate's own values) -- a genuine "
          "joint improvement should look like a plateau, not an isolated spike.")

    def grid_neighbors(val, grid):
        idx = grid.index(val)
        out = []
        if idx > 0:
            out.append(grid[idx - 1])
        if idx < len(grid) - 1:
            out.append(grid[idx + 1])
        return out

    neighbor_combos = set()
    dims = [("k_boost", K_GRID), ("ic_l", IC_L_GRID), ("mom_short", SHORT_GRID),
            ("mom_long", LONG_GRID), ("gain", GAIN_GRID)]
    base_vals = dict(k_boost=best["k_boost"], ic_l=best["ic_l"], mom_short=best["mom_short"],
                      mom_long=best["mom_long"], gain=best["gain"])
    for dim_name, grid in dims:
        for nb_val in grid_neighbors(base_vals[dim_name], grid):
            combo = dict(base_vals)
            combo[dim_name] = nb_val
            neighbor_combos.add(tuple(sorted(combo.items())))

    print(f"\n{'perturbed dim':<14} {'K':>5} {'IC_L':>5} {'MOM_S':>6} {'MOM_L':>6} {'GAIN':>5} "
          f"{'OLD':>8} {'NEW':>8} {'rmean':>8} {'rfloor':>8} {'n_worse':>9}")
    # print the winner itself first
    print(f"{'(winner)':<14} {best['k_boost']:>5} {best['ic_l']:>5} {best['mom_short']:>6} "
          f"{best['mom_long']:>6} {best['gain']:>5} {best['OLD']:>8.1f} {best['NEW']:>8.1f} "
          f"{best['rmean']:>8.1f} {best['rfloor']:>8.1f} {best['nworse']:>6}/61")
    for combo_items in sorted(neighbor_combos):
        combo = dict(combo_items)
        changed_dim = next(k for k in combo if combo[k] != base_vals[k])
        POS = build_pos(combo["k_boost"], combo["ic_l"], combo["mom_short"], combo["mom_long"],
                         combo["gain"])
        wo = window(POS, *OLD); wn = window(POS, *NEW)
        scs = scs_curve(POS)
        nworse = int((scs < base_scs).sum())
        print(f"{changed_dim:<14} {combo['k_boost']:>5} {combo['ic_l']:>5} {combo['mom_short']:>6} "
              f"{combo['mom_long']:>6} {combo['gain']:>5} {wo:>8.1f} {wn:>8.1f} {scs.mean():>8.1f} "
              f"{scs.min():>8.1f} {nworse:>6}/61")

print(f"\ntotal wall time: {time.time() - T0:.1f}s")
