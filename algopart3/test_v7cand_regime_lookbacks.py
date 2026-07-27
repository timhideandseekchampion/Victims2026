"""test_v7cand_regime_lookbacks.py -- does the vol-regime-adaptive-lookback trick (currently applied
ONLY to MOM_LB in SAFE_llboost_v2/v6: MOM_LB_SHORT=7 in elevated vol, MOM_LB_LONG=12 in calm vol,
selected on today's ALGO volz sign `fh`) generalize to three OTHER fixed lookback windows in
SAFE_llboost_v6.py, tested ONE AT A TIME against the actual shipped v6, holding everything else
(including the already-adaptive MOM_LB) at v6's shipped values:

  (a) REV_W = 10        -- idio ridge's mean-reversion blend window (BLEND=0.3 leg, idio book)
  (b) BOOST_IC_L = 250   -- the pairwise boost's own sign-check window (idio book)
  (c) IC_EW_W = 200      -- the ALGO leg's fast recency-weighted IC lookback (in _ic_ew)

Each of the three is a genuinely separate, previously-untested hypothesis (MOM_LB is the only
lookback this trick has ever touched). (a) and (b) live in the idio book, a DIFFERENT code path
from the ALGO leg where MOM_LB's regime flag is computed -- so the regime flag is re-derived here,
independently, straight from ALGO's own price series, using the identical VOL_WIN/VOL_Z/IC_LOOKBACK
volz computation v6 itself uses (see algo_regime_series() below). No plumbing across modules, no
look-ahead: it's the same causal quantity, just computed twice.

Methodology (mirrors test_v7cand_algoresweep.py's "frozen idio/algo + reconstruction sanity check"
pattern): precompute the pieces each sub-experiment does NOT touch ONCE, reuse them everywhere, and
validate that every reconstruction pipeline reproduces the REAL SAFE_llboost_v6.getMyPosition
output EXACTLY at the shipped fixed values before trusting any swept number.

Scoring convention identical to validate_llboost_v6_full.py: window(POS,S,E) PnL with commission
(commRate=1e-4, inst0=2e-5; dlr=10_000, inst0=100_000), score=mu*sr^2/(sr^2+1), OLD=window(500,750),
NEW=window(750,nt), rolling mean/floor over end_days=range(400,nt+1,10) (61 windows), n_worse vs the
REAL shipped SAFE_llboost_v6.getMyPosition walk-forward (imported directly, not approximated).
CAUSAL ONLY throughout -- every quantity at day-index k only uses data through day k.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v6 as V6

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)
r = np.diff(logp, axis=1)
rs = r[1:]                       # idio return matrix (49, nt-1), ALGO excluded
n_idio = rs.shape[0]

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
FIRST_DAY = 148  # covers every rolling window (earliest need: end_day=400 -> S=150 -> POS index 149)


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<34}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line, flush=True)
    return wo, wn, scs


def beats_v6(wo, wn, scs, base_wo, base_wn, base_scs):
    """'beats v6 on OLD+NEW+rolling-mean together' per the task spec."""
    return (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())


# ============================================================================================
# 1) Baseline: the ACTUAL shipped SAFE_llboost_v6.getMyPosition, walked forward day by day.
# ============================================================================================
print("=== baseline: real SAFE_llboost_v6.getMyPosition (production module, not approximated) ===")
t0 = time.time()
POS_V6 = np.zeros((nInst, nt))
for k in range(FIRST_DAY, nt):
    POS_V6[:, k] = V6.getMyPosition(P[:, :k + 1])
print(f"  done in {time.time()-t0:.0f}s")
base_wo, base_wn, base_scs = report("SAFE_llboost_v6 (shipped, real)", POS_V6)

# ============================================================================================
# 2) The shared vol-regime flag: today's ALGO volz sign (`fh`), computed causally and
#    INDEPENDENTLY from ALGO's own price series, replicating v6's own computation exactly
#    (VOL_WIN/VOL_Z/IC_LOOKBACK unchanged). elevated[k] = True means fh>0 (elevated vol, the
#    regime MOM_LB_SHORT=7 already fires on); valid[k] = False before enough ALGO history exists
#    (irrelevant here -- always well before day 150, outside every scored window).
# ============================================================================================
def algo_regime_series():
    lpA = logp[0]
    T = len(lpA)
    rr = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[V6.VOL_WIN:] = V6._roll_std(rr, V6.VOL_WIN)
    elevated = np.zeros(T, dtype=bool)
    valid = np.zeros(T, dtype=bool)
    for tnow in range(V6.VOL_WIN + V6.VOL_Z, T):
        if tnow + 1 < V6.VOL_WIN + V6.VOL_Z + 60:
            continue
        wv = vol[tnow - V6.VOL_Z:tnow]
        vz = (vol[tnow] - wv.mean()) / (wv.std() + 1e-12)
        fh = np.clip(vz, -3, 3) / 3.0
        if np.isnan(fh):
            continue
        elevated[tnow] = fh > 0
        valid[tnow] = True
    return elevated, valid


elevated, regime_valid = algo_regime_series()
print(f"\nregime flag: valid from day-index {np.argmax(regime_valid)} onward, "
      f"{elevated[regime_valid].mean()*100:.1f}% of valid days elevated "
      f"(days 150-{nt-1}: {elevated[150:].mean()*100:.1f}% elevated)")

# ============================================================================================
# 3) Precompute the pieces each sub-experiment does NOT touch: the pure ridge ensemble (pre-blend,
#    identical across all three experiments and across every SAFE variant), and the shipped
#    fixed-REV_W=10 blend on top of it (WZ_FIXED -- what (b) and (c) hold the idio ridge at).
# ============================================================================================
print("\n=== precompute: pure ridge ensemble WZ_RIDGE[k] (pre-blend) ===")
t0 = time.time()
WZ_RIDGE = {}
for k in range(V6.WARMUP, nt):
    rr_ = r[:, :k]
    fs = []
    for hl in V6.HALF_LIVES:
        B, mx, my = V6._ewls_ridge(rr_[:, :-1].T, rr_[1:, 1:].T, hl, V6.RIDGE_A)
        pred = my + (rr_[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    WZ_RIDGE[k] = np.mean(fs, 0)
print(f"  done in {time.time()-t0:.0f}s")


def revw_blend(k, rev_w):
    """Mean-reversion blend leg (rv), REV_W=rev_w, matching v6's `rr = logp[1:,-1] - logp[1:,-1-REV_W]`
    exactly at day-index k."""
    rr_ = logp[1:, k] - logp[1:, k - rev_w]
    rr_ = rr_ - rr_.mean()
    return -rr_ / (rr_.std() + 1e-12)


WZ_FIXED = {}
for k in range(V6.WARMUP, nt):
    rv = revw_blend(k, V6.REV_W)
    WZ_FIXED[k] = (1 - V6.BLEND) * WZ_RIDGE[k] + V6.BLEND * rv

# shipped boost (IC_L=250 fixed), using the REAL production function directly -- no reimplementation
BOOST_FIXED = {k: V6._pairwise_boost(rs[:, :k]) for k in range(V6.WARMUP, nt)}


def idio_pos_from(wz_dict, boost_dict):
    idio_pos = np.zeros((nInst, nt))
    for k in range(V6.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        wz = wz_dict[k] + V6.BOOST_K * boost_dict[k]
        idio_pos[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return idio_pos


# ============================================================================================
# SANITY CHECK 0: WZ_FIXED + BOOST_FIXED must reproduce the real idio book EXACTLY.
# ============================================================================================
recon_idio = idio_pos_from(WZ_FIXED, BOOST_FIXED)
diff0 = np.abs(recon_idio[1:, FIRST_DAY:] - POS_V6[1:, FIRST_DAY:])
print(f"\nsanity 0 (fixed idio reconstruction vs real v6): max abs share diff = {diff0.max():.6g}")
assert diff0.max() == 0, "idio reconstruction pipeline bug -- stop, nothing below can be trusted"
print("  MATCH EXACTLY.")


# ============================================================================================
# (a) REV_W: switch between REV_W_SHORT / REV_W_LONG on the independently-derived regime flag.
#     Idio ridge (WZ_RIDGE) and boost (BOOST_FIXED, shipped N=39/IC_L=250/MIN_DAY=480/SCALE_W=1000/
#     K=1.5/P=2.0) held fixed; ALGO leg reused verbatim from the real v6 run (POS_V6[0,:]).
# ============================================================================================
def build_idio_pos_revw(short_w, long_w):
    idio_pos = np.zeros((nInst, nt))
    for k in range(V6.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        rev_w = short_w if elevated[k] else long_w
        rv = revw_blend(k, rev_w)
        wz = (1 - V6.BLEND) * WZ_RIDGE[k] + V6.BLEND * rv + V6.BOOST_K * BOOST_FIXED[k]
        idio_pos[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return idio_pos


def full_pos_a(short_w, long_w):
    POS = build_idio_pos_revw(short_w, long_w)
    POS[0, :] = POS_V6[0, :]
    return POS


# sanity: short=long=10 must reproduce the real idio book exactly (regime choice is a no-op then)
recon_a = full_pos_a(V6.REV_W, V6.REV_W)
diffa = np.abs(recon_a[:, FIRST_DAY:] - POS_V6[:, FIRST_DAY:])
print(f"\nsanity (a) (REV_W short=long=10 vs real v6): max abs share diff = {diffa.max():.6g}")
assert diffa.max() == 0, "REV_W reconstruction pipeline bug"
print("  MATCH EXACTLY.")

print("\n" + "=" * 100)
print("(a) REV_W vol-regime-adaptive: SHORT (elevated vol) x LONG (calm vol), boost/ALGO leg unchanged")
print("=" * 100)
results_a = {}
for short_w in (5, 6, 7, 8):
    for long_w in (12, 14, 16, 20):
        wo, wn, scs = report(f"REV_W short={short_w} long={long_w}", full_pos_a(short_w, long_w), base_scs)
        results_a[(short_w, long_w)] = (wo, wn, scs)

beats_a = [(k, v) for k, v in results_a.items() if beats_v6(v[0], v[1], v[2], base_wo, base_wn, base_scs)]
print(f"\n(a) combos beating v6 on OLD+NEW+rmean together: {[k for k, v in beats_a]}")
if beats_a:
    for (bs, bl), _ in beats_a:
        print(f"\n--- neighbor-stability check around REV_W short={bs} long={bl} ---")
        for s2 in range(max(3, bs - 2), bs + 3):
            for l2 in range(max(s2 + 1, bl - 3), bl + 4):
                report(f"REV_W short={s2} long={l2}", full_pos_a(s2, l2), base_scs)
else:
    print("  none -- REV_W adaptive lookback does not clear v6 on the joint OLD+NEW+rmean bar.")


# ============================================================================================
# (b) BOOST_IC_L: switch the boost's own sign-check window between SHORT/LONG on the same regime
#     flag. Idio ridge+blend held at v6 shipped (WZ_FIXED, REV_W=10 fixed); N=39/MIN_DAY=480/
#     SCALE_W=1000/K=1.5/P=2.0 unchanged; ALGO leg reused verbatim from the real v6 run.
# ============================================================================================
def pairwise_boost_adaptive_icl(rs_slice, ic_l):
    """Faithful parameterized copy of V6._pairwise_boost with BOOST_IC_L replaced by `ic_l`;
    every other constant (N_CANDIDATES, MIN_DAY, SCALE_W, P, significance threshold) sourced
    directly from the V6 module so it can never silently drift from the shipped values."""
    n_, T_ = rs_slice.shape
    boost = np.zeros(n_)
    if T_ < V6.BOOST_MIN_DAY:
        return boost
    Xi_full = rs_slice[:, :-1]; Yj = rs_slice[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V6._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V6.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V6._corrmat(Xi, Yj)
    for j in range(n_):
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
        lead = rs_slice[i]
        scale = np.nanstd(lead[max(0, T_ - 1 - V6.BOOST_SCALE_W):T_ - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V6.BOOST_P
        a = max(0, T_ - 1 - ic_l)
        xs = lead_boost[a:T_ - 1]; ys = rs_slice[j, a + 1:T_]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


# sanity: pairwise_boost_adaptive_icl at fixed ic_l=250 must reproduce V6._pairwise_boost exactly
_chk = [200, 500, 800, 999]
maxdiff = max(np.abs(pairwise_boost_adaptive_icl(rs[:, :k], V6.BOOST_IC_L) - V6._pairwise_boost(rs[:, :k])).max()
              for k in _chk)
print(f"\nsanity (custom pairwise_boost @ IC_L=250 vs V6._pairwise_boost, days {_chk}): max abs diff = {maxdiff:.6g}")
assert maxdiff == 0, "custom pairwise_boost reimplementation bug"
print("  MATCH EXACTLY.")


def build_idio_pos_boosticl(short_l, long_l):
    idio_pos = np.zeros((nInst, nt))
    boost_at = {}
    for k in range(V6.BOOST_MIN_DAY, nt):
        ic_l = short_l if elevated[k] else long_l
        boost_at[k] = pairwise_boost_adaptive_icl(rs[:, :k], ic_l)
    for k in range(V6.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ_FIXED[k].copy()
        if k >= V6.BOOST_MIN_DAY:
            wz = wz + V6.BOOST_K * boost_at[k]
        idio_pos[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return idio_pos


def full_pos_b(short_l, long_l):
    POS = build_idio_pos_boosticl(short_l, long_l)
    POS[0, :] = POS_V6[0, :]
    return POS


# sanity: short=long=250 must reproduce the real idio book exactly
t0 = time.time()
recon_b = full_pos_b(V6.BOOST_IC_L, V6.BOOST_IC_L)
diffb = np.abs(recon_b[:, FIRST_DAY:] - POS_V6[:, FIRST_DAY:])
print(f"\nsanity (b) (BOOST_IC_L short=long=250 vs real v6, {time.time()-t0:.0f}s): "
      f"max abs share diff = {diffb.max():.6g}")
assert diffb.max() == 0, "BOOST_IC_L reconstruction pipeline bug"
print("  MATCH EXACTLY.")

print("\n" + "=" * 100)
print("(b) BOOST_IC_L vol-regime-adaptive: SHORT (elevated vol) x LONG (calm vol), ridge/blend/ALGO unchanged")
print("=" * 100)
t0 = time.time()
results_b = {}
for short_l in (150, 180, 210, 230):
    for long_l in (270, 300, 350, 400):
        wo, wn, scs = report(f"BOOST_IC_L short={short_l} long={long_l}", full_pos_b(short_l, long_l), base_scs)
        results_b[(short_l, long_l)] = (wo, wn, scs)
print(f"  (b) grid done in {time.time()-t0:.0f}s")

beats_b = [(k, v) for k, v in results_b.items() if beats_v6(v[0], v[1], v[2], base_wo, base_wn, base_scs)]
print(f"\n(b) combos beating v6 on OLD+NEW+rmean together: {[k for k, v in beats_b]}")
if beats_b:
    for (bs, bl), _ in beats_b:
        print(f"\n--- neighbor-stability check around BOOST_IC_L short={bs} long={bl} ---")
        for s2 in (bs - 30, bs - 15, bs, bs + 15, bs + 30):
            for l2 in (bl - 30, bl - 15, bl, bl + 15, bl + 30):
                if s2 < 60 or l2 <= s2:
                    continue
                report(f"BOOST_IC_L short={s2} long={l2}", full_pos_b(s2, l2), base_scs)
else:
    print("  none -- BOOST_IC_L adaptive lookback does not clear v6 on the joint OLD+NEW+rmean bar.")


# ============================================================================================
# (c) IC_EW_W: switch the ALGO leg's fast recency-weighted IC lookback between SHORT/LONG on the
#     SAME regime flag `fh` that is computed WITHIN the same function, one line above where IC_EW_W
#     is used -- this is the self-reference case flagged in the task: fh is derived purely from
#     volz (independent of IC_EW_W), so choosing IC_EW_W *after* fh is known is causal and not
#     circular in the lookahead sense -- but it DOES mean the same regime bit now drives TWO
#     adaptive mechanisms at once (the already-shipped MOM_LB switch, and this one), both acting
#     inside the same _side() veto gate. Flagged explicitly below; reported honestly if degenerate.
#     Idio book held at v6 shipped (reuse POS_V6[1:,:] verbatim, untouched by this experiment).
# ============================================================================================
def algo_vol_shares_icew_adaptive(lpA, cur0, cap_dol, icew_short, icew_long):
    """Faithful parameterized copy of V6._algo_vol_shares with IC_EW_W replaced by a regime-adaptive
    choice (icew_short if today's fh>0 else icew_long), used everywhere the constant was used:
    inside _side()'s ics=[...] ensemble-IC veto check, AND as the trailing bound for building z10.
    Every other constant sourced directly from V6 (VOL_MODE, IC_BLEND, VOL_COMBINE, MOM_LB_SHORT/
    LONG, etc. all untouched -- v6's shipped adaptive MOM_LB stays exactly as shipped)."""
    T = len(lpA)
    if T < V6.VOL_WIN + V6.VOL_Z + 60:
        return 0
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan); vol[V6.VOL_WIN:] = V6._roll_std(r_, V6.VOL_WIN)
    tnow = T - 1
    lo = max(V6.VOL_WIN + V6.VOL_Z, tnow - V6.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V6.VOL_Z:s]
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

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        return 0
    icew = icew_short if fh > 0 else icew_long   # <-- regime-adaptive choice, THE one change under test

    def _side(feat, fhv):
        icf = _ic(feat, V6.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        if not V6.IC_BLEND: return sf * fhv
        ics = [_ic_ew(feat, hl, icew) for hl in V6.IC_EW_HL]
        if any(x is None for x in ics): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    if V6.VOL_MODE == "switch":
        sig = _side(volz, fh)
        if sig is None: return 0
        if V6.VOL_COMBINE:
            mom_lb = V6.MOM_LB_SHORT if fh > 0 else V6.MOM_LB_LONG
            mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
            z10 = np.full(T, np.nan)
            for s in range(max(mom_lb + V6.VOL_Z, tnow - icew), T):
                wm = mom[s - V6.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
            fhm = np.clip(z10[tnow], -3, 3) / 3.0
            msig = _side(z10, fhm) if not np.isnan(fhm) else None
            if msig is not None:
                av = V6.COMBINE_GAIN * (sig + msig) * 100_000.0
            else:
                av = V6.SWITCH_GAIN * sig * 100_000.0
        else:
            av = V6.SWITCH_GAIN * sig * 100_000.0
    else:
        ic = _ic(volz, V6.IC_LOOKBACK)
        if ic is None: return 0
        av = V6.VOL_GAIN * max(0.0, ic) * fh * 100_000.0
    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    return int(np.clip(av / cur0, -lim, lim))


def build_algo_pos_icew(short_w, long_w):
    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(algo_vol_shares_icew_adaptive(logp[0, :k + 1], cur0, dlr[0], short_w, long_w),
                               -lim0, lim0)
    return algo_pos


def full_pos_c(short_w, long_w):
    POS = POS_V6.copy()
    POS[0, :] = build_algo_pos_icew(short_w, long_w)
    return POS


# sanity: short=long=200 must reproduce the real ALGO leg (and hence full v6) exactly
recon_c = full_pos_c(V6.IC_EW_W, V6.IC_EW_W)
diffc = np.abs(recon_c[:, FIRST_DAY:] - POS_V6[:, FIRST_DAY:])
print(f"\nsanity (c) (IC_EW_W short=long=200 vs real v6): max abs share diff = {diffc.max():.6g}")
assert diffc.max() == 0, "IC_EW_W reconstruction pipeline bug"
print("  MATCH EXACTLY.")

print("\n" + "=" * 100)
print("(c) IC_EW_W vol-regime-adaptive: SHORT (elevated vol) x LONG (calm vol), idio book unchanged")
print("(NOTE: same regime bit as the already-shipped MOM_LB switch -- watch for degeneracy)")
print("=" * 100)
results_c = {}
for short_w in (100, 130, 150, 180):
    for long_w in (220, 260, 300, 350):
        wo, wn, scs = report(f"IC_EW_W short={short_w} long={long_w}", full_pos_c(short_w, long_w), base_scs)
        results_c[(short_w, long_w)] = (wo, wn, scs)

# degeneracy diagnostic: how often does the veto gate (icf vs ice sign agreement) actually flip
# between the short and long IC_EW_W choice, vs. producing the identical decision either way?
n_flip = 0
n_total = 0
for k in range(400, nt, 25):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    a_short = algo_vol_shares_icew_adaptive(logp[0, :k + 1], cur0, dlr[0], 100, 100)
    a_long = algo_vol_shares_icew_adaptive(logp[0, :k + 1], cur0, dlr[0], 350, 350)
    n_total += 1
    if a_short != a_long:
        n_flip += 1
print(f"\ndegeneracy check: of {n_total} sampled days, IC_EW_W=100(fixed) vs IC_EW_W=350(fixed) "
      f"produce a DIFFERENT ALGO position on {n_flip} days ({100*n_flip/n_total:.0f}%) "
      "-- if this is low, the sweep below is mostly noise around an inactive knob.")

beats_c = [(k, v) for k, v in results_c.items() if beats_v6(v[0], v[1], v[2], base_wo, base_wn, base_scs)]
print(f"\n(c) combos beating v6 on OLD+NEW+rmean together: {[k for k, v in beats_c]}")
if beats_c:
    for (bs, bl), _ in beats_c:
        print(f"\n--- neighbor-stability check around IC_EW_W short={bs} long={bl} ---")
        for s2 in (bs - 30, bs - 15, bs, bs + 15, bs + 30):
            for l2 in (bl - 30, bl - 15, bl, bl + 15, bl + 30):
                if s2 < 60 or l2 <= s2:
                    continue
                report(f"IC_EW_W short={s2} long={l2}", full_pos_c(s2, l2), base_scs)
else:
    print("  none -- IC_EW_W adaptive lookback does not clear v6 on the joint OLD+NEW+rmean bar.")

print("\ndone: (a) REV_W, (b) BOOST_IC_L, (c) IC_EW_W all tested one at a time against real shipped v6.")
