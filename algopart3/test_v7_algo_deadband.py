"""
test_v7_algo_deadband.py

Follow-up to test_v7_algo_headroom.py's utilisation breakdown: on the 25/499 days where the ALGO
leg's raw combine target lands under 50% of the $100k cap (a near-cancellation of `sig` and `msig`,
i.e. the vol-regime and momentum sub-signals disagree in strength though not enough to flip the
combined sign), those days lose -$81/day on average, against +$309/day (50-99% bucket) and +$188/day
(>=99% bucket) everywhere else. Low |av| looks like low-conviction AND wrong-sign-prone, not just
small.

Two deadband treatments on that specific "raw target is small" condition, both fully causal (use
only the day's own signal value, no look-ahead), swept over threshold as % of the $100k cap:

  FLATTEN: |av_raw| < threshold*cap  ->  today's ALGO position is 0 instead of the small partial
           share count. Foregoes that day's edge entirely -- pays only if the bucket's edge is
           actually negative (it is, per the headroom diagnostic), not just small.
  HOLD:    |av_raw| < threshold*cap  ->  keep YESTERDAY's ALGO position instead of resizing to the
           new (small, uncertain-sign) target. Cheaper than FLATTEN (avoids the round-trip
           commission of flattening then re-entering) and only pays if yesterday's larger, more
           confident position is a better bet than today's marginal one.

The idio book (WZ + boost) and the ALGO leg's underlying vol/momentum signal construction are
IDENTICAL to SAFE_llboost_v7 in every variant -- only the deadband gate on top of `_algo_vol_shares`'s
raw target changes, so any score delta is attributable purely to this gate.
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


# ==================================================================================================
# raw ALGO target (before cap-clip), verbatim V7._algo_vol_shares logic
# ==================================================================================================
def algo_raw(lpA):
    T = len(lpA)
    if T < V7.VOL_WIN + V7.VOL_Z + 60:
        return 0.0
    rr = np.diff(lpA)
    vol = np.full(T, np.nan); vol[V7.VOL_WIN:] = V7._roll_std(rr, V7.VOL_WIN)
    tnow = T - 1
    lo = max(V7.VOL_WIN + V7.VOL_Z, tnow - V7.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V7.VOL_Z:s]
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
        icf = _ic(feat, V7.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, V7.IC_EW_W) for hl in V7.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return 0.0
    sig = _side(volz, fh)
    if sig is None: return 0.0
    mom_lb = V7.MOM_LB_SHORT if fh > 0 else V7.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V7.VOL_Z, tnow - V7.IC_EW_W), T):
        wm = mom[s - V7.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        return V7.COMBINE_GAIN * (sig + msig) * 100_000.0
    return V7.SWITCH_GAIN * sig * 100_000.0


print("=== instrumenting raw ALGO target (pre-clip) ===", flush=True)
t0 = time.time()
AV_RAW = np.zeros(nt)
for k in range(130, nt):
    AV_RAW[k] = algo_raw(logp[0, :k + 1])
print(f"  done ({time.time()-t0:.0f}s)")

util = np.abs(AV_RAW) / dlr[0]
print(f"  sanity: {int((util[500:nt-1] < 0.5).sum())} days <50% util in NEW-adjacent range "
      f"(headroom diagnostic found 25/499 over days 500+)")

# ==================================================================================================
print("=== rebuilding v7 idio book (unchanged in every variant) ===", flush=True)
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


def build(algo_shares_arr):
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        POS[1:, k] = np.clip(np.sign(WZB[:, k]) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_shares_arr
    return POS


def algo_baseline_shares():
    out = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        a = float(np.clip(AV_RAW[k], -dlr[0], dlr[0]))
        out[k] = int(np.clip(a / cur0, -lim, lim))
    return out


def algo_deadband_shares(thresh_frac, mode):
    """mode: 'flatten' -> 0 below threshold; 'hold' -> repeat yesterday's share count below threshold."""
    out = np.zeros(nt)
    prev = 0
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        a = AV_RAW[k]
        if abs(a) < thresh_frac * dlr[0]:
            sh = 0 if mode == "flatten" else prev
        else:
            sh = int(np.clip(a / cur0, -lim, lim))
        out[k] = sh
        prev = sh
    return out


POS_base = build(algo_baseline_shares())
base_scs = scs_curve(POS_base)
print(f"\nv7: OLD={wscore(POS_base,*OLD):.1f}  NEW={wscore(POS_base,*NEW):.1f}  "
      f"rmean={base_scs.mean():.1f}  rfloor={base_scs.min():.1f}   (README: 830.3/888.5/876.8/674.4)")


def report(nm, sh):
    Pz = build(sh); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > wscore(POS_base, *OLD)) and (wn > wscore(POS_base, *NEW)) and (scs.mean() > base_scs.mean())
    tag = "  <== PASS" if passed else ""
    print(f"  {nm:<24}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
          f"n_worse={int((scs<base_scs).sum())}/{len(scs)}{tag}")
    return passed


print("\n=== FLATTEN below threshold (go to 0 instead of a small partial position) ===")
any_pass = False
for thresh in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
    any_pass |= report(f"FLATTEN thr={thresh:.2f}", algo_deadband_shares(thresh, "flatten"))

print("\n=== HOLD below threshold (keep yesterday's position instead of resizing) ===")
for thresh in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
    any_pass |= report(f"HOLD thr={thresh:.2f}", algo_deadband_shares(thresh, "hold"))

print(f"\n{'Some variant passed OLD+NEW+rmean jointly.' if any_pass else 'No variant beat v7 on OLD+NEW+rmean jointly.'}")
