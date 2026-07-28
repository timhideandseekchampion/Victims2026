"""
test_v7_algo_headroom.py  --  follow-up to test_v7_leak_diagnostic.py

The leak diagnostic showed the ALGO leg earns 22.5 bp/day per dollar of capital against the idio
book's 13.6 bp, at 1/100th the commission ($0.4/day vs $46.3/day) and -0.07 correlation. It is the
best capital in the book. So: how much of that $100k is actually being used, and on how many days is
it idle?

  1. UTILISATION -- distribution of |ALGO exposure| / $100k cap, and the PnL earned in each bucket.
  2. IDLE DAYS -- how often the leg sits at exactly zero, split by which mechanism zeroed it
     (the double-IC veto in `_side`, an exact cancellation `sig + msig == 0`, or no signal at all),
     and what the index actually did on those days -- i.e. is there money on the table.
  3. Two upper-bound probes on "use the cap harder", both causal and sign-only:
       - ALWAYS-CAP: whenever v7 takes any non-zero position, take the full +/-$100k instead.
         (COMBINE_GAIN -> infinity, but only for the combine path.)
       - FILL-IDLE: on days v7 sits flat, fall back to the single-signal `SWITCH_GAIN * sig` path
         it already uses when the momentum signal is unavailable, instead of standing down.
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
# instrumented copy of V7._algo_vol_shares -- returns the raw dollar target AND why it is what it is
# ==================================================================================================
def algo_instrumented(lpA, cur0, cap_dol):
    T = len(lpA)
    if T < V7.VOL_WIN + V7.VOL_Z + 60:
        return 0.0, "warmup", None, None
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
        if icf is None: return None, "no_ic"
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, V7.IC_EW_W) for hl in V7.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv, "ok_noblend"
        ice = float(np.mean(ics))
        if (ice >= 0) == (icf >= 0): return sf * fhv, "ok"
        return 0.0, "veto"

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        return 0.0, "no_volz", None, None
    sig, why_v = _side(volz, fh)
    if sig is None:
        return 0.0, "no_ic_vol", None, None
    mom_lb = V7.MOM_LB_SHORT if fh > 0 else V7.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V7.VOL_Z, tnow - V7.IC_EW_W), T):
        wm = mom[s - V7.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig, why_m = (_side(z10, fhm) if not np.isnan(fhm) else (None, "no_momz"))
    if msig is not None:
        av = V7.COMBINE_GAIN * (sig + msig) * 100_000.0
        why = f"combine[{why_v}/{why_m}]"
    else:
        av = V7.SWITCH_GAIN * sig * 100_000.0
        why = f"switch[{why_v}]"
    fallback = V7.SWITCH_GAIN * sig * 100_000.0
    return float(av), why, float(sig), float(fallback)


print("=== instrumenting the ALGO leg day by day ===", flush=True)
t0 = time.time()
AV = np.zeros(nt); WHY = [""] * nt; FALLBACK = np.zeros(nt)
for k in range(130, nt):
    av, why, sig, fb = algo_instrumented(logp[0, :k + 1], P_[0, k], dlr[0])
    AV[k] = av; WHY[k] = why; FALLBACK[k] = 0.0 if fb is None else fb
print(f"  done ({time.time()-t0:.0f}s)")


def algo_shares(av_arr):
    out = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        a = float(np.clip(av_arr[k], -dlr[0], dlr[0]))
        out[k] = int(np.clip(a / cur0, -lim, lim))
    return out


# ---- rebuild the idio book once ----
print("=== rebuilding v7 idio book ===", flush=True)
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


def build(av_arr):
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        POS[1:, k] = np.clip(np.sign(WZB[:, k]) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_shares(av_arr)
    return POS


POS = build(AV)
base_scs = scs_curve(POS)
print(f"\nv7: OLD={wscore(POS,*OLD):.1f}  NEW={wscore(POS,*NEW):.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (README: 830.3/888.5/876.8/674.4)")

# ==================================================================================================
print("\n" + "=" * 96)
print("1) UTILISATION of the $100k ALGO cap  (days 500+)")
print("=" * 96)
sh = algo_shares(AV)
expo = np.abs(sh * P_[0]) / dlr[0]
ret0 = np.concatenate((r[0], [np.nan]))         # log return earned by a position held on day k
pnl0 = sh * P_[0] * ret0
sl = np.arange(500, nt - 1)
buckets = [("flat (=0)", expo[sl] == 0),
           ("partial <50%", (expo[sl] > 0) & (expo[sl] < 0.5)),
           ("partial 50-99%", (expo[sl] >= 0.5) & (expo[sl] < 0.99)),
           ("at cap >=99%", expo[sl] >= 0.99)]
for nm, m in buckets:
    n = int(m.sum())
    pn = pnl0[sl][m]
    print(f"  {nm:<16} {n:4d} days ({100*n/len(sl):5.1f}%)   "
          f"PnL/day in bucket ${np.nanmean(pn) if n else 0:8.1f}   "
          f"total ${np.nansum(pn) if n else 0:10,.0f}")
print(f"  mean utilisation {100*expo[sl].mean():.1f}% of the $100k cap  "
      f"-> ~${(1-expo[sl].mean())*100_000:,.0f}/day of the book's best capital sits unused")

# ==================================================================================================
print("\n" + "=" * 96)
print("2) IDLE DAYS -- why is the leg flat, and what did the index do?")
print("=" * 96)
flat = np.array([expo[k] == 0 for k in range(nt)])
from collections import Counter
cnt = Counter(WHY[k] for k in range(500, nt - 1) if flat[k])
for why, n in cnt.most_common():
    idx = [k for k in range(500, nt - 1) if flat[k] and WHY[k] == why]
    mv = np.abs(ret0[idx]).mean() * 100
    print(f"  {why:<26} {n:4d} days   mean |index move| that day {mv:.2f}%")
allidx = [k for k in range(500, nt - 1) if flat[k]]
print(f"  total {len(allidx)} idle days out of {nt-1-500} ({100*len(allidx)/(nt-1-500):.1f}%); "
      f"mean |index move| on them {np.abs(ret0[allidx]).mean()*100:.2f}% vs "
      f"{np.abs(ret0[500:nt-1]).mean()*100:.2f}% on all days")

# ==================================================================================================
print("\n" + "=" * 96)
print("3) UPPER-BOUND PROBES on using the cap harder")
print("=" * 96)


def report(nm, av_arr):
    Pz = build(av_arr); scs = scs_curve(Pz)
    e = np.abs(algo_shares(av_arr) * P_[0]) / dlr[0]
    print(f"  {nm:<34}OLD={wscore(Pz,*OLD):7.1f}  NEW={wscore(Pz,*NEW):7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={int((scs<base_scs).sum())}/{len(scs)}  "
          f"util={100*e[500:nt-1].mean():.0f}%")
    return scs


report("v7 (COMBINE_GAIN=16)", AV)
report("ALWAYS-CAP (sign only)", np.sign(AV) * dlr[0])
av_fill = AV.copy()
fillmask = (AV == 0) & (FALLBACK != 0)
av_fill[fillmask] = FALLBACK[fillmask]
print(f"    (FILL-IDLE replaces {int(fillmask[500:nt-1].sum())} idle days with the switch-path signal)")
report("FILL-IDLE (switch fallback)", av_fill)
av_both = np.sign(av_fill) * dlr[0]
report("FILL-IDLE + ALWAYS-CAP", av_both)
for G in (20.0, 25.0, 40.0, 100.0):
    report(f"COMBINE_GAIN={G}", AV / V7.COMBINE_GAIN * G)
