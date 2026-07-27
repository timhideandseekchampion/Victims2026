"""test_v7cand_algoresweep.py -- re-sweep the ALGO leg's own parameters against the TRUE, currently
-shipped v6 idio book, not the stale one test_algo_leg_resweep.py used.

test_algo_leg_resweep.py re-swept VOL_WIN/VOL_Z/IC_FAST/SWITCH_GAIN/COMBINE_GAIN/IC_EW_HL/MOM_LB
against a boosted idio book built with the OLD pre-v3/v5 boost config: N_CANDIDATES=49 (no
volatility-restricted pool), BOOST_IC_L=190, BOOST_MIN_DAY=500, BOOST_K=1.5. Since then the boost
pool was restricted to N=39 (SAFE_llboost_v3) and IC_L/MIN_DAY were re-tuned to 250/480
(SAFE_llboost_v5) -- v6 ships with that refined boost PLUS combines it with a fixed MOM_LB=10
resweep is nowhere near what v6 actually needs re-checked: v6 uses a vol-regime-adaptive
MOM_LB_SHORT=7/MOM_LB_LONG=12 (SAFE_llboost_v2's contribution) instead of one fixed lookback.

This script:
  1. Precomputes the TRUE v6 idio book ONCE (ridge+blend forecast, identical across every SAFE
     variant, plus the significance-gated pairwise boost at its ACTUAL shipped parameters -- sourced
     directly from the SAFE_llboost_v6 module so this can never silently go stale again).
  2. Re-sweeps the ALGO leg's parameters one dimension at a time (coordinate descent, matching
     test_algo_leg_resweep.py / test_batch80_catA_boostpool.py convention) against that FIXED idio
     book: VOL_WIN, VOL_Z, IC_FAST, SWITCH_GAIN, COMBINE_GAIN, IC_EW_HL, and separately
     MOM_LB_SHORT/MOM_LB_LONG (the dimension the original resweep predates entirely).
  3. Scores with the exact validate_llboost_v6_full.py convention: window(POS,S,E) PnL with
     commission, OLD=window(500,750), NEW=window(750,nt), rolling mean/floor over
     end_days=range(400,nt+1,10) (61 windows, S=E-250), n_worse vs the REAL shipped
     SAFE_llboost_v6.getMyPosition walk-forward (not an approximation).
  4. Every precomputed quantity at day t/k only uses data through that day -- see the sanity-check
     block below, which confirms our from-scratch reconstruction at v6's own defaults matches the
     real getMyPosition output exactly (max abs share diff == 0) before any sweep number is trusted.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE
import SAFE_llboost_v6 as V6

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
n = rs.shape[0]

# --- TRUE v6 shipped boost parameters, sourced directly from the module (never stale) ---
BOOST_MIN_DAY = V6.BOOST_MIN_DAY
ALPHA = V6.BOOST_ALPHA
N_CANDIDATES = V6.BOOST_N_CANDIDATES
BOOST_P = V6.BOOST_P
BOOST_SCALE_W = V6.BOOST_SCALE_W
BOOST_IC_L = V6.BOOST_IC_L
BOOST_K = V6.BOOST_K
print(f"true v6 boost params: N_CANDIDATES={N_CANDIDATES} BOOST_IC_L={BOOST_IC_L} "
      f"BOOST_MIN_DAY={BOOST_MIN_DAY} BOOST_SCALE_W={BOOST_SCALE_W} BOOST_K={BOOST_K} BOOST_P={BOOST_P}")
assert (BOOST_MIN_DAY, N_CANDIDATES, BOOST_IC_L, BOOST_SCALE_W, BOOST_K, BOOST_P) == \
       (480, 39, 250, 1000, 1.5, 2.0), "true v6 boost params drifted -- update this script's assumptions"


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


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("\n=== precompute (fixed idio side, ONCE): TRUE v6 ridge WZ + N=39/IC_L=250/MIN_DAY=480/"
      "SCALE_W=1000 boost ===")
t0 = time.time()
WZ = {}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]                      # causal: only returns through day t
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    WZ[t] = wz
print(f"  WZ done ({time.time()-t0:.0f}s)")

t0 = time.time()
BOOST_AT = {}
for k in range(BOOST_MIN_DAY, nt):
    T = k                              # causal: boost at day k only uses returns through index T-1
    Xi_full = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)               # trailing-vol ranking, causal (no full-sample)
    cand_idx = np.argsort(-vol_causal)[:N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)): continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr: continue
        i = cand_idx[ci]
        lead = rs[i, :T]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0: continue
        entry[j] = lead_boost[-1]           # today's boost value only, causal
    BOOST_AT[k] = entry
print(f"  significance-gated N=39 boost map done ({time.time()-t0:.0f}s)")

idio_pos = np.zeros((nInst, nt))
for k in range(SAFE.WARMUP, nt):
    cur = P_[:, k]; lim = (dlr / cur).astype(int)
    wz = WZ[k].copy()
    if k >= BOOST_MIN_DAY:
        for j, bv in BOOST_AT[k].items():
            wz[j] += BOOST_K * bv
    idio_pos[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
print("  fixed TRUE-v6 idio book (ridge+boost) done -- this array is now FROZEN for every sweep below")


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


def algo_vol_shares_v6(lpA, cur0, cap_dol, VOL_WIN, VOL_Z, IC_FAST, SWITCH_GAIN, COMBINE_GAIN,
                        IC_EW_HL, MOM_LB_SHORT, MOM_LB_LONG, IC_EW_W=200):
    """Faithful parameterized copy of SAFE_llboost_v6._algo_vol_shares. VOL_MODE='switch',
    IC_BLEND=True, VOL_COMBINE=True are all fixed at their shipped values (not swept dims for this
    task). Depends ONLY on ALGO's own price series -- entirely independent of the idio book/boost."""
    T = len(lpA)
    if T < VOL_WIN + VOL_Z + 60:
        return 0
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan); vol[VOL_WIN:] = roll_std(r_, VOL_WIN)
    tnow = T - 1
    lo = max(VOL_WIN + VOL_Z, tnow - 250)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
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
        icf = _ic(feat, IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        ics = [_ic_ew(feat, hl, IC_EW_W) for hl in IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh): return 0
    sig = _side(volz, fh)
    if sig is None: return 0
    mom_lb = MOM_LB_SHORT if fh > 0 else MOM_LB_LONG   # vol-regime-adaptive lookback (v6/v2's change)
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + VOL_Z, tnow - IC_EW_W), T):
        wm = mom[s - VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    if msig is not None:
        av = COMBINE_GAIN * (sig + msig) * 100_000.0
    else:
        av = SWITCH_GAIN * sig * 100_000.0
    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


DEFAULTS = dict(VOL_WIN=20, VOL_Z=60, IC_FAST=90, SWITCH_GAIN=2.5, COMBINE_GAIN=3.5,
                IC_EW_HL=(20, 45), MOM_LB_SHORT=7, MOM_LB_LONG=12)


def build_algo(**overrides):
    params = {**DEFAULTS, **overrides}
    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(algo_vol_shares_v6(logp[0, :k + 1], cur0, dlr[0], **params), -lim0, lim0)
    return algo_pos


def full_pos(algo_pos):
    POS = idio_pos.copy()
    POS[0, :] = algo_pos
    return POS


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<30}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


# ============================================================================================
# SANITY CHECK: build the REAL SAFE_llboost_v6.getMyPosition walk-forward (production module,
# not an approximation) and confirm our from-scratch idio+ALGO reconstruction at v6's own default
# parameters matches it exactly. If this doesn't match, the precompute pipeline has a bug and
# nothing below should be trusted.
# ============================================================================================
print("\n=== sanity check: real SAFE_llboost_v6.getMyPosition vs from-scratch reconstruction ===")
FIRST_DAY = 148  # covers every rolling window (earliest need: end_day=400 -> S=150 -> POS index 149)
t0 = time.time()
POS_SHIPPED = np.zeros((nInst, nt))
for k in range(FIRST_DAY, nt):
    POS_SHIPPED[:, k] = V6.getMyPosition(P_[:, :k + 1])
print(f"  real getMyPosition walk-forward done ({time.time()-t0:.0f}s)")

POS_recon_default = full_pos(build_algo())
diff = np.abs(POS_recon_default[:, FIRST_DAY:] - POS_SHIPPED[:, FIRST_DAY:])
print(f"  max abs share diff (shipped vs reconstruction, days {FIRST_DAY}-{nt-1}): {diff.max():.6g}")
assert diff.max() == 0, "RECONSTRUCTION MISMATCH -- precompute pipeline has a bug, stop here"
print("  MATCH EXACTLY -- precompute pipeline confirmed correct.")

base_scs = report("shipped SAFE_llboost_v6 (real, OFFICIAL baseline)", POS_SHIPPED)
recon_scs = report("reconstruction check (v6 defaults)", POS_recon_default, base_scs)
assert np.array_equal(recon_scs, base_scs), "reconstruction rolling-window scores don't match shipped exactly"
print("  rolling-window scores identical too -- n_worse=0/61 confirms exact match, as expected.\n")

# ============================================================================================
# SWEEPS: one dimension at a time (coordinate descent) against the FIXED true-v6 idio book.
# base_scs (from the REAL getMyPosition run above) is the n_worse reference throughout.
# ============================================================================================
print("=== sweep VOL_WIN (current v6 = 20) ===")
for v in (10, 15, 20, 25, 30):
    report(f"VOL_WIN={v}", full_pos(build_algo(VOL_WIN=v)), base_scs)

print("\n=== sweep VOL_Z (current v6 = 60) ===")
for v in (30, 45, 60, 75, 90):
    report(f"VOL_Z={v}", full_pos(build_algo(VOL_Z=v)), base_scs)

print("\n=== sweep IC_FAST (current v6 = 90) ===")
for v in (60, 75, 90, 105, 120):
    report(f"IC_FAST={v}", full_pos(build_algo(IC_FAST=v)), base_scs)

print("\n=== sweep SWITCH_GAIN (current v6 = 2.5) ===")
for v in (1.5, 2.0, 2.5, 3.0, 3.5):
    report(f"SWITCH_GAIN={v}", full_pos(build_algo(SWITCH_GAIN=v)), base_scs)

print("\n=== sweep COMBINE_GAIN (current v6 = 3.5) ===")
for v in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
    report(f"COMBINE_GAIN={v}", full_pos(build_algo(COMBINE_GAIN=v)), base_scs)

print("\n=== sweep IC_EW_HL (current v6 = (20, 45)) ===")
for v in [(15, 30), (20, 45), (20, 60), (30, 60), (15, 45)]:
    report(f"IC_EW_HL={v}", full_pos(build_algo(IC_EW_HL=v)), base_scs)

print("\n=== sweep MOM_LB_SHORT, MOM_LB_LONG=12 held fixed (current v6 short=7) ===")
for v in (5, 6, 7, 8, 9, 10):
    report(f"MOM_LB_SHORT={v}", full_pos(build_algo(MOM_LB_SHORT=v)), base_scs)

print("\n=== sweep MOM_LB_LONG, MOM_LB_SHORT=7 held fixed (current v6 long=12) ===")
for v in (9, 10, 11, 12, 13, 14, 15, 16):
    report(f"MOM_LB_LONG={v}", full_pos(build_algo(MOM_LB_LONG=v)), base_scs)

print("\ndone with primary one-dimension-at-a-time coordinate descent.")
