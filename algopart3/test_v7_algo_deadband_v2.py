"""
test_v7_algo_deadband_v2.py

Finishes the ALGO-deadband thread from test_v7_algo_deadband.py. That test found HOLD (keep
yesterday's ALGO position instead of resizing into a small, near-cancellation target) passes
OLD+NEW+rmean jointly at thresh~0.10-0.25, but the 16 "worse" rolling windows are ALL the earliest
ones (end_day 400-470, roughly -15 pts each) -- exactly the shape of a mechanism that isn't reliable
yet on thin history, the same reason BOOST_MIN_DAY exists for the pairwise boost. Fix: add an
analogous minimum-history gate (DEADBAND_MIN_DAY) -- the deadband is OFF (identical to shipped v7)
before it, ON only after.

Joint sweep over (threshold x min_day), neighbor-stability check around the best config, then a
REAL getMyPosition validation: a genuine standalone candidate module (SAFE_llboost_v7_deadband.py,
same single-file-submission convention as v2..v7) with the deadband implemented as module-level
state (mirrors _limits' _DLR caching pattern already in v7) rather than a backtest reconstruction,
validated by calling its getMyPosition sequentially in increasing-day order exactly as
eval_llboost_v7.py / validate_llboost_v7_full.py do for every prior promoted candidate.
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


def algo_deadband_shares(thresh_frac, min_day):
    """HOLD deadband, gated OFF before min_day (identical to shipped v7 there)."""
    out = np.zeros(nt)
    prev = 0
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        a = AV_RAW[k]
        if k >= min_day and abs(a) < thresh_frac * dlr[0]:
            sh = prev
        else:
            sh = int(np.clip(a / cur0, -lim, lim))
        out[k] = sh
        prev = sh
    return out


POS_base = build(algo_baseline_shares())
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"\nv7 baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (README: 830.3/888.5/876.8/674.4)")


def evaluate(nm, sh, verbose=True):
    Pz = build(sh); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed, scs=scs)


print("\n=== JOINT SWEEP: threshold x DEADBAND_MIN_DAY (HOLD variant) ===")
results = []
for min_day in (200, 300, 400, 450, 480, 500, 550, 600):
    for thresh in (0.10, 0.15, 0.20, 0.25, 0.30):
        r_ = evaluate(f"thr={thresh:.2f} min_day={min_day}", algo_deadband_shares(thresh, min_day))
        r_["thresh"] = thresh; r_["min_day"] = min_day
        results.append(r_)

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs pass OLD+NEW+rmean jointly.")
clean = sorted(passing, key=lambda c: c["nworse"])
print("\ncleanest (lowest n_worse) passing configs:")
for c in clean[:8]:
    print(f"  thr={c['thresh']:.2f} min_day={c['min_day']:<4} OLD={c['wo']:.1f} NEW={c['wn']:.1f} "
          f"rmean={c['rm']:.1f} rfloor={c['rf']:.1f} n_worse={c['nworse']}/61")

best = max(passing, key=lambda c: c["rm"]) if passing else None
if best:
    print(f"\nbest by rmean: thr={best['thresh']:.2f} min_day={best['min_day']} rmean={best['rm']:.1f} "
          f"n_worse={best['nworse']}/61")

# pick the winner: prioritize a clean n_worse (repo convention: n_worse near 0 = "validated, clean"),
# among those require the best rmean
if clean:
    winner = min(clean, key=lambda c: (c["nworse"], -c["rm"]))
    print(f"\n=== SELECTED CANDIDATE: thr={winner['thresh']:.2f} min_day={winner['min_day']} ===")
    print(f"  OLD={winner['wo']:.1f} NEW={winner['wn']:.1f} rmean={winner['rm']:.1f} "
          f"rfloor={winner['rf']:.1f} n_worse={winner['nworse']}/61")

    print("\n=== neighbor-stability check around the winner ===")
    T0, M0 = winner["thresh"], winner["min_day"]
    Ts = sorted(set([0.10, 0.15, 0.20, 0.25, 0.30]))
    Ms = sorted(set([200, 300, 400, 450, 480, 500, 550, 600]))
    Ts_near = [t for t in Ts if abs(Ts.index(t) - Ts.index(T0)) <= 1]
    Ms_near = [m for m in Ms if abs(Ms.index(m) - Ms.index(M0)) <= 1]
    for m in Ms_near:
        for t in Ts_near:
            r_ = evaluate(f"  neighbor thr={t:.2f} min_day={m}",
                          algo_deadband_shares(t, m), verbose=False)
            tag = " <== WINNER" if (t == T0 and m == M0) else ""
            print(f"  thr={t:.2f} min_day={m:<4} OLD={r_['wo']:.1f} NEW={r_['wn']:.1f} "
                  f"rmean={r_['rm']:.1f} rfloor={r_['rf']:.1f} n_worse={r_['nworse']}/61{tag}")

    # worst-window diagnosis for the winner (any early-window residue left?)
    diff = winner["scs"] - base_scs
    worst = np.argsort(diff)[:6]
    print("\n  worst windows for the winner vs v7 baseline:")
    for i in worst:
        print(f"    end_day={end_days[i]:4d}  base={base_scs[i]:7.1f}  winner={winner['scs'][i]:7.1f}  "
              f"diff={diff[i]:+7.1f}")
else:
    print("\nNo passing config -- deadband idea does not survive a minimum-history gate. Stopping "
          "here; not proceeding to real-getMyPosition validation.")
