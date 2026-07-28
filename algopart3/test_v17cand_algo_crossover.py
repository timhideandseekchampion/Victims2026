"""
test_v17cand_algo_crossover.py

CANDIDATE: apply v10's rank-stability idea to the ALGO leg. The idio-side signal fades a stock's
short-term move only when it opposes its OWN medium-term trend (relative to the cross-section). ALGO
is a single instrument -- no cross-section to rank against -- so the natural analogue is a pure
TIME-SERIES version: fade ALGO's own short-term price move only when it opposes ALGO's own medium-
term trend (buy a pullback within an uptrend, sell a rebound within a downtrend), using ALGO's raw
log-price series alone.

MECHANISM: long_ret = logp[0,-1]-logp[0,-1-LONG_W], short_ret = logp[0,-1]-logp[0,-1-SHORT_W]. When
sign(long_ret) != sign(short_ret) (a genuine countermove against the trend), vote in the direction of
the trend (equivalently, -sign(short_ret)); otherwise no vote that day. This is a genuinely different
signal family from ALGO's existing vol-regime + momentum combine (`_side`/COMBINE_GAIN, both built
from realized-volatility and momentum Z-SCORES against their OWN trailing distribution, not a simple
price-level short/long crossover) -- distinct enough to test as an ADDITIONAL vote layered onto the
existing target, not a replacement.

BLEND: only touches days the crossover actually fires (sign disagreement) -- on non-firing days,
ALGO's target is completely unchanged from v10. On firing days: `av_new = (1-w)*av + w*sign(vote)*cap`
(full-conviction sign vote on the crossover, blended with the existing target at weight w) -- matches
this repo's established "full-conviction sign-based sizing beats magnitude-weighted schemes"
philosophy rather than trying to scale by the crossover's raw magnitude.

Tested against SAFE_llboost_v10 (current best) -- idio book (ridge+beta-demean+boost+rank-stability)
untouched, reused verbatim; only the ALGO leg gets this additional vote.
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
RIDGE_A = V10.RIDGE_A
HALF_LIVES = V10.HALF_LIVES


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


# ==================================================================================================
# raw ALGO target (before cap-clip and before the HOLD deadband), verbatim V10._algo_vol_shares logic
# ==================================================================================================
def algo_raw(lpA):
    T = len(lpA)
    if T < V10.VOL_WIN + V10.VOL_Z + 60:
        return 0.0
    rr = np.diff(lpA)
    vol = np.full(T, np.nan); vol[V10.VOL_WIN:] = V10._roll_std(rr, V10.VOL_WIN)
    tnow = T - 1
    lo = max(V10.VOL_WIN + V10.VOL_Z, tnow - V10.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V10.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]

    def _ic(feat, L):
        a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-12: return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _ic_ew(feat, HL, W):
        a = max(0, tnow - W); xs = feat[a:tnow]; ys = ret1[a:tnow]
        w = (0.5 ** (1.0 / HL)) ** ((tnow - 1) - np.arange(a, tnow))
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys, w = xs[ok], ys[ok], w[ok]; sw = w.sum()
        mx = (w * xs).sum() / sw; my = (w * ys).sum() / sw
        cxy = (w * (xs - mx) * (ys - my)).sum() / sw
        vx = (w * (xs - mx) ** 2).sum() / sw; vy = (w * (ys - my) ** 2).sum() / sw
        if vx < 1e-24 or vy < 1e-24: return None
        return float(cxy / np.sqrt(vx * vy))

    def _side(feat, fhv):
        icf = _ic(feat, V10.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, V10.IC_EW_W) for hl in V10.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return 0.0
    sig = _side(volz, fh)
    if sig is None: return 0.0
    mom_lb = V10.MOM_LB_SHORT if fh > 0 else V10.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V10.VOL_Z, tnow - V10.IC_EW_W), T):
        wm = mom[s - V10.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        return V10.COMBINE_GAIN * (sig + msig) * 100_000.0
    return V10.SWITCH_GAIN * sig * 100_000.0


def algo_crossover_vote(lpA, short_w, long_w):
    T = len(lpA)
    if T < max(short_w, long_w) + 5:
        return 0
    long_ret = lpA[-1] - lpA[-1 - long_w]
    short_ret = lpA[-1] - lpA[-1 - short_w]
    if long_ret == 0 or short_ret == 0:
        return 0
    if np.sign(long_ret) != np.sign(short_ret):
        return int(np.sign(long_ret))   # vote WITH the medium-term trend (fades the short-term move)
    return 0


print("=== instrumenting raw ALGO target + crossover vote (pre-clip, pre-deadband) ===", flush=True)
t0 = time.time()
AV_RAW = np.zeros(nt)
for k in range(130, nt):
    AV_RAW[k] = algo_raw(logp[0, :k + 1])
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

print("=== precompute: idio book (ridge+beta-demean+boost+rank-stability), unchanged -- reused "
      "verbatim from v10 ===", flush=True)
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
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        s_std = rs_sig.std()
        s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
        wz = (1 - V10.RS_WEIGHT) * wz + V10.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(algo_shares_arr):
    POS = np.zeros((nInst, nt))
    for t in days:
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(WZ_V10[:, t]) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_shares_arr
    return POS


def algo_baseline_shares():
    """v10's real ALGO leg (verbatim, incl. HOLD deadband state) -- the sanity-check baseline."""
    out = np.zeros(nt)
    prev = 0; prev_t = -1
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        av = AV_RAW[k]
        have_prev = (k == prev_t + 1)
        if (have_prev and k >= V10.DEADBAND_MIN_DAY
                and abs(av) < V10.DEADBAND_THRESH_FRAC * dlr[0]):
            sh = prev
        else:
            av_c = float(np.clip(av, -dlr[0], dlr[0]))
            sh = int(np.clip(av_c / cur0, -lim, lim))
        out[k] = sh; prev = sh; prev_t = k
    return out


def algo_crossover_shares(short_w, long_w, weight):
    out = np.zeros(nt)
    prev = 0; prev_t = -1
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        av = AV_RAW[k]
        have_prev = (k == prev_t + 1)
        if (have_prev and k >= V10.DEADBAND_MIN_DAY
                and abs(av) < V10.DEADBAND_THRESH_FRAC * dlr[0]):
            av_c = float(np.clip(prev * cur0, -dlr[0], dlr[0]))  # matches the shipped HOLD (in $ terms)
        else:
            av_c = float(np.clip(av, -dlr[0], dlr[0]))
        vote = algo_crossover_vote(logp[0, :k + 1], short_w, long_w)
        if vote != 0:
            av_c = (1 - weight) * av_c + weight * vote * dlr[0]
        sh = int(np.clip(av_c / cur0, -lim, lim))
        out[k] = sh; prev = sh; prev_t = k
    return out


print("\n=== sanity check: v10's own ALGO leg reconstruction must reproduce v10 exactly ===")
POS_base = build_pos(algo_baseline_shares())
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, short_w, long_w, weight, verbose=True):
    Pz = build_pos(algo_crossover_shares(short_w, long_w, weight)); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, short_w=short_w, long_w=long_w, weight=weight, wo=wo, wn=wn, rm=scs.mean(),
                rf=scs.min(), nworse=nworse, passed=passed, scs=scs)


print("\n=== initial sweep: (short_w, long_w) x weight ===")
results = []
for sw, lw in ((5, 15), (8, 22), (10, 30), (15, 40), (20, 60)):
    for w in (0.1, 0.2, 0.3, 0.5):
        results.append(evaluate(f"short{sw}_long{lw} w={w}", sw, lw, w))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: short_w={best['short_w']} long_w={best['long_w']} weight={best['weight']} "
          f"rmean={best['rm']:.1f} n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:8]:
        print(f"  short_w={c['short_w']:<3} long_w={c['long_w']:<3} weight={c['weight']:<4} "
              f"OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} "
              f"n_worse={c['nworse']}/61")
