"""
test_v20cand_idio_deadband.py

QUESTION: SAFE_llboost_v8 added an ALGO min-conviction HOLD deadband (don't resize into a small,
uncertain-sign combine target -- hold yesterday's ALGO shares instead), validated because ALGO's IC
is genuinely non-stationary and its low-magnitude days were shown to be actually loss-making (-$81/day
vs +$309/day elsewhere). Does an ANALOGOUS per-name deadband on the IDIO book help?

The "v7 budget" diagnostic (README) argues no: idio pooled IC is sign-stable ("nothing to adapt to")
and per-name IC is unresolvable (SNR 0.69, "can't resolve its own sign") -- but that's an inference
from aggregate IC statistics, not a direct test of this specific mechanism. Testing directly against
SAFE_llboost_v10 (current best), not assuming the inference holds.

MECHANISM: for each idio name i on each day, if |wz_i| falls below some fraction of that day's typical
cross-sectional |wz| level (a small, near-coin-flip combine target for that specific name), hold
yesterday's position for that name instead of resizing. Two treatments, matching the ALGO deadband
test's own convention: HOLD (yesterday's position) and FLATTEN (go to 0). Distinct from the double-IC
agreement gate already tested and rejected for idio (test_v7cand_double_ic_idio.py) -- this uses raw
|wz| magnitude, not cross-half-life sign agreement.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

print("=== precompute: full SAFE_llboost_v10 wz (ridge ensemble + BLEND + boost + rank-stability), "
      "verbatim -- and ALGO leg, unaffected by anything tested here ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_V10 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    if t >= max(RS_SHORT_W, RS_LONG_W) + 5:
        short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
        long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
        sz = short_ret - short_ret.mean(); sstd = sz.std()
        lz = long_ret - long_ret.mean(); lstd = lz.std()
        if sstd > 1e-12 and lstd > 1e-12:
            sz = sz / sstd; lz = lz / lstd
            disagree = np.sign(lz) != np.sign(sz)
            rs_sig = np.where(disagree, -sz, 0.0)
            s_std = rs_sig.std()
            s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos_baseline():
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: no gating must reproduce SAFE_llboost_v10 exactly ===")
POS_base = build_pos_baseline()
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def build_pos_deadband(thresh_frac, mode, min_day):
    POS = np.zeros((nInst, nt))
    prev_idio_pos = np.zeros(nIdio)
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        raw_pos = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
        if t >= min_day:
            day_scale = np.abs(wz).mean() + 1e-12
            low_conv = np.abs(wz) < thresh_frac * day_scale
            new_pos = np.where(low_conv, prev_idio_pos, raw_pos) if mode == 'hold' \
                else np.where(low_conv, 0.0, raw_pos)
        else:
            new_pos = raw_pos
        POS[1:, t] = new_pos
        prev_idio_pos = new_pos
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, thresh_frac, mode, min_day, verbose=True):
    Pz = build_pos_deadband(thresh_frac, mode, min_day); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, thresh=thresh_frac, mode=mode, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                nworse=nworse, passed=passed)


print("\n=== how often does the gate fire, and what fraction of days/names does it touch? "
      "(min_day=480, thresh_frac=0.25, for scale) ===")
touched, total = 0, 0
for t in range(480, nt):
    wz = WZ_V10[:, t]
    day_scale = np.abs(wz).mean() + 1e-12
    touched += int((np.abs(wz) < 0.25 * day_scale).sum())
    total += nIdio
print(f"  {touched}/{total} name-days ({100*touched/total:.1f}%) fall under the 0.25x threshold")

print("\n=== SWEEP: HOLD treatment, thresh_frac in {0.05..0.5}, min_day=480 ===")
THRESH = [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
hold_results = [evaluate(f"HOLD thresh={t}", t, 'hold', 480) for t in THRESH]

print("\n=== SWEEP: FLATTEN treatment, thresh_frac in {0.05..0.5}, min_day=480 ===")
flat_results = [evaluate(f"FLATTEN thresh={t}", t, 'flatten', 480) for t in THRESH]

all_results = hold_results + flat_results
passing = [c for c in all_results if c["passed"]]
print(f"\n{len(passing)}/{len(all_results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for c in passing:
        print(f"  {c['name']:<28} rmean={c['rm']:.1f} n_worse={c['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(all_results, key=lambda c: -c["rm"])[:8]:
        print(f"  {c['name']:<28} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print("\n=== WHY: $ PnL of 'low-conviction' name-days vs the rest (thresh_frac=0.25, day>=480), "
      "same measurement ALGO's deadband writeup used ===")
lo_pnl, lo_n, hi_pnl, hi_n = 0.0, 0, 0.0, 0
for t in range(480, nt - 1):
    wz = WZ_V10[:, t]
    day_scale = np.abs(wz).mean() + 1e-12
    low_conv = np.abs(wz) < 0.25 * day_scale
    pos_t = POS_base[1:, t]
    pnl_t = pos_t * (P_[1:, t + 1] - P_[1:, t])
    lo_pnl += pnl_t[low_conv].sum(); lo_n += int(low_conv.sum())
    hi_pnl += pnl_t[~low_conv].sum(); hi_n += int((~low_conv).sum())
print(f"  low-conviction (|wz|<0.25x day mean): {lo_n} name-days, ${lo_pnl/lo_n:.2f}/name-day")
print(f"  rest:                                 {hi_n} name-days, ${hi_pnl/hi_n:.2f}/name-day")
