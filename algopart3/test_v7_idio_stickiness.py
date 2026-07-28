"""
test_v7_idio_stickiness.py

Re-run of test_hysteresis.py's idea against the REAL shipped book. The original test only ever ran
against SAFE_llvol (no pairwise boost, N=49 idio names, a materially earlier/different book) and was
never re-checked after SAFE_llboost/v2..v7 added the significance-gated boost on top of wz. This
rebuilds the exact SAFE_llboost_v7 idio score (ridge+blend + BOOST_K*boost) and re-tests the same
hysteresis idea: hold the previous day's sign unless |wz| clears a threshold (optionally also
requiring a minimum hold period), instead of flipping every sign change regardless of conviction.

Per the v7 score-budget diagnostic (test_v7_leak_diagnostic.py): commission is $46.3/day of $726
idio gross (NEW window), so ANY turnover cut is hard-capped at recovering that -- and at Sharpe 7.5
a 1% mean giveback costs 30x more score than a 1% variance cut buys, so this only pays if it barely
touches accuracy. The ALGO leg is IDENTICAL in every variant, so any score delta is attributable
purely to this idio entry rule.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V7.WARMUP, V7.BOOST_MIN_DAY, V7.BOOST_K


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


def flips_per_name(sgn, S, E):
    ch = np.abs(np.diff(sgn[:, S:E], axis=1)) > 0
    return ch.sum(axis=1).mean()


# ==================================================================================================
print("=== rebuilding v7 idio book: WZ + BOOST_K*boost (unchanged across every variant) ===", flush=True)
t0 = time.time()
WZB = np.full((nIdio, nt), np.nan)
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V7.HALF_LIVES:
        B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    rv_ = logp[1:, t] - logp[1:, t - V7.REV_W]
    rv_ = rv_ - rv_.mean()
    WZB[:, t] = (1 - V7.BLEND) * wz + V7.BLEND * (-rv_ / (rv_.std() + 1e-12))
for k in range(BOOST_MIN_DAY, nt):
    WZB[:, k] = WZB[:, k] + BOOST_K * V7._pairwise_boost(rs[:, :k])
print(f"  done ({time.time()-t0:.0f}s)")

print("=== ALGO leg (identical in every variant) ===", flush=True)
t0 = time.time()
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V7._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)")


def build_pos(thresh, min_hold=0):
    POS = np.zeros((nInst, nt))
    prev_sign = np.zeros(nIdio)
    hold_days = np.zeros(nIdio, dtype=int)
    sgn_hist = np.zeros((nIdio, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZB[:, k]
        new_sign = np.sign(wz)
        if thresh > 0 or min_hold > 0:
            flip_ok = (np.abs(wz) >= thresh) & (hold_days >= min_hold)
            keep = ~flip_ok & (prev_sign != 0)
            new_sign = np.where(keep, prev_sign, new_sign)
        hold_days = np.where(new_sign == prev_sign, hold_days + 1, 0)
        prev_sign = new_sign
        sgn_hist[:, k] = new_sign
        POS[1:, k] = np.clip(new_sign * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS, sgn_hist


POS_base, sgn_base = build_pos(0.0, 0)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"\nv7 (thresh=0, no deadband): OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (README: 830.3/888.5/876.8/674.4)")
print(f"  flips/name over NEW window: {flips_per_name(sgn_base, *NEW):.1f} "
      f"(~1 every {NUMTEST/flips_per_name(sgn_base,*NEW):.1f} days)")

# a reasonable threshold scale: wz's cross-sectional/day spread
wz_scale = np.nanstd(WZB[:, WARMUP:])
print(f"  cross-sample stdev of wz (incl. boost): {wz_scale:.3f} -- thresholds below are in these units")

print(f"\n{'thresh':>8} {'min_hold':>9} {'OLD':>8} {'NEW':>8} {'rmean':>8} {'rfloor':>8} "
      f"{'n_worse':>9} {'flips/name(NEW)':>16}")
results = []
for thresh, min_hold in [
    (0.00, 0), (0.05, 0), (0.10, 0), (0.15, 0), (0.20, 0), (0.30, 0), (0.50, 0),
    (0.00, 1), (0.00, 2), (0.00, 3),
    (0.10, 1), (0.10, 2), (0.20, 1), (0.20, 2),
]:
    POS, sgn = build_pos(thresh * wz_scale, min_hold)
    wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
    scs = scs_curve(POS)
    fl = flips_per_name(sgn, *NEW)
    nworse = int((scs < base_scs).sum())
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    tag = "  <== PASS" if passed else ""
    print(f"{thresh:>8.2f} {min_hold:>9} {wo:>8.1f} {wn:>8.1f} {scs.mean():>8.1f} {scs.min():>8.1f} "
          f"{nworse:>9}/{len(scs)} {fl:>16.1f}{tag}")
    results.append(dict(thresh=thresh, min_hold=min_hold, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(),
                        nworse=nworse, passed=passed))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} variants beat v7 on OLD+NEW+rmean jointly.")
if not passing:
    print("Nothing passes -- consistent with the score-budget diagnostic: commission is only 5% of "
          "gross, and any accuracy given back to reduce turnover costs ~30x more score than it saves.")
