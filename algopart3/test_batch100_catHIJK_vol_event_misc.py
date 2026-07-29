"""
test_batch100_catHIJK_vol_event_misc.py

BATCH of 8 items tested/diagnosed against the current best (SAFE_llboost_v10, real graded score
912.64 on days 751-1000): six new-signal/mechanism candidates (static vol-tercile filter, leader-vol
split on the boost, up/down-vol asymmetry, post-2-sigma fixed fade, multi-name co-crash response,
boost/rank-stability agreement gate) plus two confirmatory diagnostics (half-life re-verification,
day-of-week check). Uses the shared, pre-built, VERIFIED harness (`_v10_harness.py` as `H`) throughout
-- every `H.evaluate()` call is scored against the SAME real v10 baseline the harness asserts at
import time (OLD=871.0 NEW=912.6 rmean=909.8 rfloor=709.7), never a self-reported number.

Explicitly SKIPPED this batch (not run -- noted honestly rather than silently dropped):
  - "Volume-spike via range" idea: INFEASIBLE. This panel (`prices.txt`) is close-price-only -- no
    volume, no high/low -- so there is no range proxy to build a volume-spike feature from.
  - Logistic/probit-target idea: DEPRIORITIZED. Reframing the idio ridge target as a classification
    problem sits in the same territory as the already-decisively-rejected quantile/GBM family (see
    README's nonlinear-target section) -- not worth re-chasing this session.

CAUSALITY: every per-day quantity below only ever reads price/return history through the column being
decided, using the same `r[:, :t]` / `returns[:, t-w:t]` slicing convention `_v10_harness.py` itself
uses for WZ_PRE -- day t's feature never touches column t or later of the raw return array.

ORIGIN of each idea/diagnostic and why it's the right first move (cheap check vs full backtest) is
noted inline at each item.
"""
import numpy as np
import _v10_harness as H
import SAFE_llboost_v10 as V10

rs_full = H.rs_full   # (nIdio, nt-1) idio returns; rs_full[:, k] = logp[1:, k+1] - logp[1:, k]
logp = H.logp
nIdio = H.nIdio
nt = H.nt
days = H.days
W = 60   # trailing realized-vol window used throughout this file, per assignment ("~60-day, causal")


def causal_vol(returns, t, w=W):
    """std of the w returns ending at column t-1 (returns[:, t-w:t]) -- 'history through today',
    same convention _v10_harness.py uses for WZ_PRE's r[:, :t]. None if not enough history."""
    if t < w:
        return None
    return returns[:, t - w:t].std(axis=1)


def tercile_split(vec):
    """indices into low/mid/high terciles of a 1D vector (ties broken by argsort order)."""
    n = len(vec)
    k = n // 3
    order = np.argsort(vec)
    return order[:k], order[k:n - k], order[-k:]


print("\nSkipped this batch (see docstring for why): volume-spike-via-range (infeasible, no volume/"
      "high-low data), logistic/probit-target (deprioritized, same territory as rejected quantile/"
      "GBM family).\n")

# ==============================================================================================
# ITEM 1: static vol-tercile filter -- trade only top and/or bottom tercile (others flat) vs all-50
# ==============================================================================================
print("=" * 100)
print("ITEM 1: static vol-tercile filter (trailing 60d realized vol, causal)")
print("=" * 100)

WZ_TOP = H.BASE_WZ.copy()
WZ_BOTTOM = H.BASE_WZ.copy()
WZ_TOPBOT = H.BASE_WZ.copy()
for t in days:
    vol = causal_vol(rs_full, t)
    if vol is None:
        continue
    low, mid, high = tercile_split(vol)
    WZ_TOP[low, t] = 0.0; WZ_TOP[mid, t] = 0.0
    WZ_BOTTOM[mid, t] = 0.0; WZ_BOTTOM[high, t] = 0.0
    WZ_TOPBOT[mid, t] = 0.0

H.evaluate("vol-tercile: TOP only (flat elsewhere)", WZ_TOP)
H.evaluate("vol-tercile: BOTTOM only (flat elsewhere)", WZ_BOTTOM)
H.evaluate("vol-tercile: TOP+BOTTOM (drop mid third)", WZ_TOPBOT)

# ==============================================================================================
# ITEM 2: leader-vol-level diagnostic on the pairwise boost's next-day hit-rate
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 2: leader-vol-level diagnostic (DIAGNOSTIC ONLY unless a clear split appears)")
print("=" * 100)


def pairwise_boost_with_leader(rs):
    """Exact clone of V10._pairwise_boost, plus tracking of which leader index was picked for each
    follower j, so results can be split by the leader's own vol tercile. Self-checked below against
    H.BOOST_BASE (the harness's real, production-identical boost array)."""
    n, T = rs.shape
    boost = np.zeros(n)
    leader = np.full(n, -1)
    if T < V10.BOOST_MIN_DAY:
        return boost, leader
    Xi_full = rs[:, :-1]; Yj = rs[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V10.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs[i]
        scale = np.nanstd(lead[max(0, T - 1 - V10.BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V10.BOOST_P
        a = max(0, T - 1 - V10.BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
        leader[j] = i
    return boost, leader


LEADER = np.full((nIdio, nt), -1)
BOOST_CHK = np.zeros((nIdio, nt))
for k in range(V10.BOOST_MIN_DAY, nt):
    b, l = pairwise_boost_with_leader(rs_full[:, :k])
    BOOST_CHK[:, k] = b
    LEADER[:, k] = l

max_diff = np.nanmax(np.abs(BOOST_CHK - H.BOOST_BASE))
print(f"  self-check: max|recomputed boost - H.BOOST_BASE| = {max_diff:.3e} (should be ~0)")

hits = {"low": [0, 0], "mid": [0, 0], "high": [0, 0]}
for t in range(V10.BOOST_MIN_DAY, nt - 1):   # need rs_full[:, t] (next-day realized return in-range)
    vol_t = causal_vol(rs_full, t)
    if vol_t is None:
        continue
    low, mid, high = tercile_split(vol_t)
    tercile_of = np.full(nIdio, 1)
    tercile_of[low] = 0; tercile_of[high] = 2
    js = np.where(H.BOOST_BASE[:, t] != 0)[0]
    for j in js:
        i = int(LEADER[j, t])
        if i < 0:
            continue
        tname = ["low", "mid", "high"][tercile_of[i]]
        hit = int(np.sign(H.BOOST_BASE[j, t]) == np.sign(rs_full[j, t]))
        hits[tname][0] += hit
        hits[tname][1] += 1

print("  boost next-day hit-rate, split by LEADER's own trailing-vol tercile that day:")
for tname in ["low", "mid", "high"]:
    h, n_ = hits[tname]
    rate = h / n_ if n_ else float("nan")
    print(f"    leader-vol {tname:<5} tercile: hit-rate = {rate:.3f}  (n={n_})")

# ==============================================================================================
# ITEM 3: upside/downside volatility asymmetry as a standalone cross-sectional predictor
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 3: upside/downside volatility asymmetry (trailing ~60d, causal)")
print("=" * 100)


def updown_vol(returns, t, w=W, min_count=10):
    if t < w:
        return None
    win = returns[:, t - w:t]
    n = win.shape[0]
    ratio = np.full(n, np.nan); diff = np.full(n, np.nan)
    for i in range(n):
        x = win[i]
        up = x[x > 0]; dn = x[x < 0]
        if len(up) < min_count or len(dn) < min_count:
            continue
        vu = up.std(); vd = dn.std()
        ratio[i] = vu / (vd + 1e-12)
        diff[i] = vu - vd
    return ratio, diff


SIG_RATIO = np.full((nIdio, nt), np.nan)
SIG_DIFF = np.full((nIdio, nt), np.nan)
n_nan_days = 0
for t in days:
    out = updown_vol(rs_full, t)
    if out is None:
        n_nan_days += 1
        continue
    ratio, diff = out
    if np.isnan(ratio).any():
        n_nan_days += 1
        continue
    SIG_RATIO[:, t] = ratio
    SIG_DIFF[:, t] = diff

print(f"  coverage: {len(days) - n_nan_days}/{len(days)} graded-eligible days have a full 50-name "
      f"up/down-vol read (min_count=10 up AND down obs in the trailing {W}d window)")

for label, SIG in [("up/down-vol RATIO", SIG_RATIO), ("up/down-vol DIFF", SIG_DIFF)]:
    for weight in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        WZ_v = H.BASE_WZ.copy()
        for t in days:
            sv = SIG[:, t]
            if np.isnan(sv).any():
                continue
            WZ_v[:, t] = H.blend_signal(H.BASE_WZ[:, t], sv, weight)
        H.evaluate(f"{label} weight={weight}", WZ_v)

# ==============================================================================================
# ITEM 4: post-2-sigma FIXED-size fade (not magnitude-scaled, unlike the shipped continuous REV leg)
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 4: post-2-sigma fixed-size fade")
print("=" * 100)


def jump_flags(returns, t, w=W, k_sigma=2.0):
    """Flag names whose most-recent return (returns[:, t-1], the 'jump') exceeds k_sigma * trailing
    w-day stdev computed on the w returns strictly BEFORE the jump (returns[:, t-1-w:t-1]) -- so the
    jump itself never inflates its own threshold."""
    if t < w + 1:
        return None
    sigma = returns[:, t - 1 - w:t - 1].std(axis=1)
    jump = returns[:, t - 1]
    flagged = np.abs(jump) > k_sigma * (sigma + 1e-12)
    return flagged, jump


n_flagged_total = 0
for t in days:
    out = jump_flags(rs_full, t)
    if out is not None:
        n_flagged_total += int(out[0].sum())
print(f"  incidence: {n_flagged_total} flagged (name, day) pairs over {len(days)} graded-eligible "
      f"days ({n_flagged_total / (len(days) * nIdio) * 100:.2f}% of all name-days)")

for extra_w in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    WZ_v = H.BASE_WZ.copy()
    for t in days:
        out = jump_flags(rs_full, t)
        if out is None:
            continue
        flagged, jump = out
        if not flagged.any():
            continue
        scale = np.abs(H.BASE_WZ[:, t]).mean() + 1e-12
        fade_dir = -np.sign(jump)
        WZ_v[flagged, t] = H.BASE_WZ[flagged, t] + extra_w * fade_dir[flagged] * scale
    H.evaluate(f"post-2sigma fixed fade extra_w={extra_w}", WZ_v)

# ==============================================================================================
# ITEM 5: multi-name co-crash trigger
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 5: multi-name co-crash trigger (>=5/50 idio names move >2 sigma same day)")
print("=" * 100)

co_crash = []   # (return-index k, count of flagged names)
for k in range(W, rs_full.shape[1]):
    sigma = causal_vol(rs_full, k)
    jump = rs_full[:, k]
    flagged = np.abs(jump) > 2.0 * (sigma + 1e-12)
    c = int(flagged.sum())
    if c >= 5:
        co_crash.append((k, c))

print(f"  total co-crash days over the whole history (return-index k, response day = k+1): "
      f"{len(co_crash)}")
graded_ks = []
if co_crash:
    ks = np.array([k for k, _ in co_crash])
    print(f"  return-index range: {ks.min()} - {ks.max()}; histogram by 200-day bucket:")
    for lo in range(0, rs_full.shape[1], 200):
        cnt = int(((ks >= lo) & (ks < lo + 200)).sum())
        print(f"    k in [{lo:4d},{lo + 200:4d}): {cnt}")
    graded_ks = [k for k in ks if H.NEW[0] <= (k + 1) <= H.NEW[1]]
    print(f"  co-crash days whose response day (k+1) falls in the graded window {H.NEW}: "
          f"{len(graded_ks)}")

if len(graded_ks) < 10:
    print(f"  -> only {len(graded_ks)} in the graded window (<10) -- too few to matter; SKIPPING a "
          f"full backtest variant per assignment instructions.")
else:
    response_days = set(int(k) + 1 for k in ks)
    WZ_v = np.full((nIdio, nt), np.nan)
    for t in days:
        boost_k_eff = 0.5 * V10.BOOST_K if t in response_days else V10.BOOST_K
        wz = H.WZ_PRE[:, t] + boost_k_eff * H.BOOST_BASE[:, t]
        WZ_v[:, t] = H.rs_blend(wz, t)
    H.evaluate("co-crash response: halve BOOST_K on flagged days", WZ_v)

# ==============================================================================================
# ITEM 6: boost / rank-stability agreement gate -- damp (not zero) on sign disagreement
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 6: boost / rank-stability agreement gate")
print("=" * 100)

RS_RAW = np.full((nIdio, nt), np.nan)
for t in range(V10.BOOST_MIN_DAY, nt):
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        RS_RAW[:, t] = rs_sig

boost_win = H.BOOST_BASE[:, V10.BOOST_MIN_DAY:]
rs_win = RS_RAW[:, V10.BOOST_MIN_DAY:]
boost_nz = boost_win != 0
rs_nz = ~np.isnan(rs_win) & (rs_win != 0)
both_nz = boost_nz & rs_nz
with np.errstate(invalid="ignore"):
    opp_sign = both_nz & (np.sign(boost_win) != np.sign(np.nan_to_num(rs_win)))

total_eligible = boost_win.size
n_boost_nz = int(boost_nz.sum())
n_both_nz = int(both_nz.sum())
n_opp = int(opp_sign.sum())

print(f"  eligible stock-days (boost-active window, t>={V10.BOOST_MIN_DAY}): {total_eligible}")
print(f"  boost nonzero: {n_boost_nz} ({n_boost_nz / total_eligible * 100:.1f}% of eligible)")
print(f"  both boost & rank-stability nonzero: {n_both_nz} ({n_both_nz / total_eligible * 100:.1f}% "
      f"of eligible)")
print(f"  both nonzero AND opposite-signed: {n_opp} "
      f"({n_opp / total_eligible * 100:.1f}% of all eligible stock-days; "
      f"{(n_opp / n_both_nz * 100 if n_both_nz else float('nan')):.1f}% of both-nonzero cases; "
      f"{(n_opp / n_boost_nz * 100 if n_boost_nz else float('nan')):.1f}% of boost-active cases)")

WZ_v = np.full((nIdio, nt), np.nan)
for t in days:
    boost_t = H.BOOST_BASE[:, t].copy()
    rs_sig_full = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig_full is None:
        flagged = np.zeros(nIdio, dtype=bool)
    else:
        flagged = (boost_t != 0) & (rs_sig_full != 0) & (np.sign(boost_t) != np.sign(rs_sig_full))
    boost_eff = boost_t.copy()
    boost_eff[flagged] *= 0.5
    wz_boosted = H.WZ_PRE[:, t] + V10.BOOST_K * boost_eff
    if rs_sig_full is None:
        WZ_v[:, t] = wz_boosted
        continue
    s_std = rs_sig_full.std()
    s_z = (rs_sig_full - rs_sig_full.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig_full)
    s_z_eff = s_z.copy()
    s_z_eff[flagged] *= 0.5
    WZ_v[:, t] = (1 - V10.RS_WEIGHT) * wz_boosted + V10.RS_WEIGHT * s_z_eff * (np.abs(wz_boosted).mean() + 1e-12)

H.evaluate("boost/RS damped-on-disagreement (halve both terms)", WZ_v)

# ==============================================================================================
# ITEM 7: half-life re-verification (CONFIRMATORY, not a new signal)
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 7: half-life re-verification")
print("=" * 100)


def build_wz_pre(half_lives):
    """Exact template of _v10_harness.py's own WZ_PRE precompute loop (lines ~78-97), parameterized
    on HALF_LIVES so each nudge only recomputes the ridge ensemble, reusing the shipped boost
    (H.BOOST_BASE) and rank-stability blend (H.rs_blend) unchanged on top."""
    wz_pre = np.full((nIdio, nt), np.nan)
    for t in days:
        rr_ = H.r[:, :t]
        X = rr_[:, :-1].T
        Y = V10._beta_adjusted_target(rr_)
        xq = rr_[:, -1]
        fs = []
        for hl in half_lives:
            B, mx, my = V10._ewls_ridge(X, Y, hl, V10.RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if V10.BLEND > 0:
            rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
            rv_ = rv_ - rv_.mean()
            rv_ = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - V10.BLEND) * wz + V10.BLEND * rv_
        wz_pre[:, t] = wz
    return wz_pre


_chk = build_wz_pre(V10.HALF_LIVES)
print(f"  self-check: max|rebuilt WZ_PRE - H.WZ_PRE| at shipped HALF_LIVES = "
      f"{np.nanmax(np.abs(_chk - H.WZ_PRE)):.3e} (should be ~0)")

nudges = {250: (200, 300), 500: (400, 600), 1000: (800, 1200), 2000: (1600, 2400)}
any_improved = False
for hl_orig, alts in nudges.items():
    for alt in alts:
        hls = tuple(alt if h == hl_orig else h for h in V10.HALF_LIVES)
        wz_pre_n = build_wz_pre(hls)
        WZ_v = np.full((nIdio, nt), np.nan)
        for t in days:
            wz = wz_pre_n[:, t] + V10.BOOST_K * H.BOOST_BASE[:, t]
            WZ_v[:, t] = H.rs_blend(wz, t)
        res = H.evaluate(f"HALF_LIVES {hl_orig}->{alt}  {hls}", WZ_v)
        if res["passed"]:
            any_improved = True

print(f"\n  ITEM 7 verdict: {'something MOVED -- a nudge passed the strict bar, investigate' if any_improved else 'still optimal -- every nudge remains at or below shipped v10 on the joint bar, nothing drifted'}")

# ==============================================================================================
# ITEM 8: day-of-week check (cheap, likely a non-finding -- synthetic panel)
# ==============================================================================================
print("\n" + "=" * 100)
print("ITEM 8: day-of-week check")
print("=" * 100)

day_idx = np.arange(rs_full.shape[1])
dow = day_idx % 5
avg_idio_ret = rs_full.mean(axis=0)

print("  cross-sectional average idio return by day_index % 5:")
for d in range(5):
    m = avg_idio_ret[dow == d]
    print(f"    dow={d}: mean={m.mean() * 1e4:8.3f}bp  std={m.std() * 1e4:7.2f}bp  n={len(m)}")

next_day_ret = avg_idio_ret[1:]
dow_today = dow[:-1]
ic_ordinal = np.corrcoef(dow_today, next_day_ret)[0, 1]
print(f"\n  IC (ordinal day-of-week 0-4, today) vs NEXT-day cross-sectional avg idio return: "
      f"{ic_ordinal:.4f}")
print("  per-dummy IC (today's dow==d indicator) vs next-day avg idio return:")
for d in range(5):
    dummy = (dow_today == d).astype(float)
    ic_d = np.corrcoef(dummy, next_day_ret)[0, 1]
    print(f"    dow=={d}: IC={ic_d:+.4f}")

print("\n" + "=" * 100)
print("DONE -- see per-item output above for pass/fail against the strict H.evaluate() bar.")
print("=" * 100)
