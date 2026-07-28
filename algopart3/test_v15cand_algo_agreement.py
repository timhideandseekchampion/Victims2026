"""
test_v15cand_algo_agreement.py

CANDIDATE (external suggestion #2, reconstructed from a partially-visible description -- flagged
honestly): a "signal agreement" gate on the ALGO leg's vol-regime (`sig`) and momentum (`msig`)
sub-signals, distinct from the magnitude-based deadband already shipped in v8/v9.

RECONSTRUCTION, stated explicitly (the source screenshot was cut off before the exact mechanism):
the visible text says this "keeps the good part of v7: full conviction when ALGO signals agree" and
"avoids... forcing full-cap ALGO when the combined signal is basically noise." Read literally, this
proposes gating on SIGN AGREEMENT between `sig` and `msig` (do the vol-regime and momentum
sub-signals point the same direction?), NOT on the MAGNITUDE of their sum -- which is what v9's
existing deadband already does (`|sig+msig| < threshold -> hold`). These are genuinely different
criteria: two sub-signals can disagree in sign while still summing to something large in magnitude
(if one dominates), and can agree in sign while summing to something small (if both are weak) -- so
sign-agreement and magnitude-threshold are not the same gate, and this is worth testing as a
DISTINCT candidate rather than assuming it's redundant with the deadband.

FOUR variants tested for what to do on DISAGREEMENT (sign(sig) != sign(msig)), REPLACING the existing
magnitude deadband so the comparison isolates this specific mechanism (not stacked with it):
  A) FLATTEN: av=0 on disagreement days.
  B) FALLBACK: use SWITCH_GAIN*sig*100_000 (drop momentum entirely, vol-only sizing -- literally the
     same fallback the shipped code already uses when momentum is UNAVAILABLE, just re-purposed for
     when momentum is available but disagrees).
  C) REDUCED_GAIN: use a smaller combine gain (sweep) instead of COMBINE_GAIN=16 on disagreement days.
  D) HOLD: hold yesterday's position on disagreement (same mechanical idea as v9's deadband, but
     gated by sign-disagreement instead of |sig+msig| magnitude).

On AGREEMENT days, every variant is identical to v9 (full COMBINE_GAIN*(sig+msig), unchanged).

Tested against SAFE_llboost_v9 (current best) -- idio book (ridge+beta-demean+boost) untouched,
reused verbatim; only the ALGO leg's disagreement handling changes.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v9 as V9

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V9.WARMUP, V9.BOOST_MIN_DAY, V9.BOOST_K
RIDGE_A = V9.RIDGE_A
HALF_LIVES = V9.HALF_LIVES


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
# instrumented ALGO leg -- returns sig, msig (both None-able) alongside the raw av, so disagreement
# handling can be swapped in without touching the underlying signal construction (verbatim V9 logic)
# ==================================================================================================
def algo_sig_msig(lpA):
    T = len(lpA)
    if T < V9.VOL_WIN + V9.VOL_Z + 60:
        return None, None, None
    tnow = T - 1
    rr = np.diff(lpA)
    vol = np.full(T, np.nan); vol[V9.VOL_WIN:] = V9._roll_std(rr, V9.VOL_WIN)
    lo = max(V9.VOL_WIN + V9.VOL_Z, tnow - V9.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V9.VOL_Z:s]
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
        icf = _ic(feat, V9.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, V9.IC_EW_W) for hl in V9.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return None, None, None
    sig = _side(volz, fh)
    if sig is None: return None, None, None
    mom_lb = V9.MOM_LB_SHORT if fh > 0 else V9.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V9.VOL_Z, tnow - V9.IC_EW_W), T):
        wm = mom[s - V9.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    return sig, msig, fh


print("=== instrumenting ALGO leg: sig, msig per day (unchanged signal construction) ===", flush=True)
t0 = time.time()
SIG = np.full(nt, np.nan); MSIG = np.full(nt, np.nan)
for k in range(130, nt):
    sig, msig, fh = algo_sig_msig(logp[0, :k + 1])
    if sig is not None: SIG[k] = sig
    if msig is not None: MSIG[k] = msig
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

agree = np.isfinite(SIG) & np.isfinite(MSIG) & (np.sign(SIG) == np.sign(MSIG))
disagree = np.isfinite(SIG) & np.isfinite(MSIG) & (np.sign(SIG) != np.sign(MSIG))
only_sig = np.isfinite(SIG) & ~np.isfinite(MSIG)
print(f"  days 500+: agree={int(agree[500:].sum())}  disagree={int(disagree[500:].sum())}  "
      f"only_sig(no momentum)={int(only_sig[500:].sum())}  total={nt-500}")


def algo_shares(mode, reduced_gain=None):
    """mode: 'flatten' | 'fallback' | 'reduced' | 'hold' -- applied ONLY on disagree days;
    agreement and only-sig days always use the shipped v9 formula (COMBINE_GAIN or SWITCH_GAIN)."""
    out = np.zeros(nt)
    prev = 0
    for k in range(130, nt):
        cur0 = P_[0, k]; lim = int(dlr[0] / cur0)
        sig = SIG[k]; msig = MSIG[k]
        if not np.isfinite(sig):
            av = 0.0
        elif not np.isfinite(msig):
            av = V9.SWITCH_GAIN * sig * 100_000.0
        elif np.sign(sig) == np.sign(msig):
            av = V9.COMBINE_GAIN * (sig + msig) * 100_000.0
        else:
            if mode == "flatten":
                av = 0.0
            elif mode == "fallback":
                av = V9.SWITCH_GAIN * sig * 100_000.0
            elif mode == "reduced":
                av = reduced_gain * (sig + msig) * 100_000.0
            elif mode == "hold":
                av = None
        if mode == "hold" and av is None:
            sh = prev
        else:
            av = float(np.clip(av, -dlr[0], dlr[0]))
            sh = int(np.clip(av / cur0, -lim, lim))
        out[k] = sh
        prev = sh
    return out


print("=== precompute: idio book (ridge+beta-demean+boost), unchanged -- reused verbatim from v9 ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V9.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V9._pairwise_boost(rs[:, :k])

WZ_RIDGE = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V9._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V9._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    WZ_RIDGE[:, t] = (1 - V9.BLEND) * wz + V9.BLEND * REV[:, t]
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(algo_arr):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_RIDGE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_arr
    return POS


print("\n=== sanity check: v9's own ALGO leg (via V9._algo_vol_shares directly) as the baseline ===")
algo_v9 = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_v9[k] = np.clip(V9._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
POS_base = build_pos(algo_v9)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v9 docstring: 848.8/893.3/894.1/708.6)")
if not (abs(base_wo - 848.8) < 0.5 and abs(base_wn - 893.3) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v9 -- do not trust results below. ***")
else:
    print("  OK -- matches v9 to within rounding.")


def evaluate(nm, mode, reduced_gain=None, verbose=True):
    algo_arr = algo_shares(mode, reduced_gain)
    Pz = build_pos(algo_arr); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed, scs=scs)


print("\n=== A) FLATTEN on disagreement ===")
evaluate("A: flatten", "flatten")

print("\n=== B) FALLBACK to vol-only sizing on disagreement ===")
evaluate("B: fallback (switch_gain)", "fallback")

print("\n=== C) REDUCED_GAIN on disagreement (sweep) ===")
for g in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0):
    evaluate(f"C: reduced_gain={g}", "reduced", g)

print("\n=== D) HOLD yesterday's position on disagreement ===")
evaluate("D: hold", "hold")
