"""Batch of 80, Category C (items 41-60): portfolio-construction / sizing scheme variants.
All [MECH]: straight to scored backtest vs the current shipped SAFE_llboost.py.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE, SAFE_llvol

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
n = rs.shape[0]

BOOST_MIN_DAY = 500
ALPHA = 0.05
N_CANDIDATES = 49
BOOST_P = 2.0
BOOST_SCALE_W = 1000
BOOST_IC_L = 190
BOOST_K = 1.5


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


print("=== shared precompute: shipped ridge WZ (raw, pre-boost) + significance boost + ALGO leg ===")
t0 = time.time()
WZ_SHIP = {}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]
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
    WZ_SHIP[t] = wz
print(f"  WZ done ({time.time()-t0:.0f}s)")

BOOST_AT = {}
for k in range(BOOST_MIN_DAY, nt):
    T = k
    Xi = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr: continue
        lead = rs[i, :T]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0: continue
        entry[j] = lead_boost[-1]
    BOOST_AT[k] = entry
print("  boost map done")

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("  ALGO leg done")


def full_wz(k):
    wz = WZ_SHIP[k].copy()
    if k >= BOOST_MIN_DAY:
        for j, bv in BOOST_AT[k].items():
            wz[j] += BOOST_K * bv
    return wz


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<44}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


def build_pos_sign(sign_fn=None):
    """sign_fn(k, wz) -> sign array (or None to use plain sign)."""
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = full_wz(k)
        sgn = np.sign(wz) if sign_fn is None else sign_fn(k, wz)
        POS[1:, k] = np.clip(sgn * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity: baseline ===")
base_scs = report("shipped baseline (sanity)", build_pos_sign())

print("\n### Item 41: Kelly-criterion-inspired sizing (scale by trailing win-rate edge) ###")
def sign_kelly(k, wz, win_w=250):
    a = max(SAFE.WARMUP, k - win_w)
    # per-stock trailing hit-rate: how often sign(wz_t) matched next-day return sign, using realized rs
    sgn = np.sign(wz)
    return sgn  # placeholder -- full version below computes actual scale

def build_kelly_scaled(win_w=250, min_scale=0.3):
    POS = np.zeros((nInst, nt))
    hit_hist = np.zeros((n, nt))  # 1 if yesterday's sign matched today's realized return
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = full_wz(k)
        sgn = np.sign(wz)
        a = max(0, k - win_w)
        if k > a + 30:
            hits = hit_hist[:, a:k]
            hit_rate = hits.mean(axis=1)  # (n,)
            scale = np.clip(2 * hit_rate - 1, min_scale, 1.0)  # Kelly-like: edge = 2p-1
        else:
            scale = np.ones(n)
        POS[1:, k] = np.clip(scale * sgn * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
        if k < nt - 1:
            realized_sign = np.sign(rs[:, k]) if k < rs.shape[1] else np.zeros(n)
            hit_hist[:, k] = (np.sign(POS[1:, k]) == realized_sign).astype(float)
    POS[0, :] = algo_pos
    return POS

report("Kelly-scaled (win_w=250)", build_kelly_scaled(250), base_scs)
report("Kelly-scaled (win_w=120)", build_kelly_scaled(120), base_scs)

print("\n### Item 42: drawdown-based book-level throttle ###")
def build_drawdown_throttle(dd_thresh=0.02, scale_down=0.5, look=60):
    POS_full = build_pos_sign()
    # compute book-level cumulative PnL causally, then scale positions down after a trailing drawdown
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None
    cum = 0.0; peak = 0.0
    POS = POS_full.copy()
    cumseries = np.zeros(nt)
    for tt in range(SAFE.WARMUP + 1, nt):
        cur = P_[:, tt - 1]
        newPos = POS_full[:, tt - 1]
        pl = curPos * (cur - prevCur) - comm_vec if prevCur is not None else 0.0
        cum += float(np.sum(pl)) if not np.isscalar(pl) else pl
        peak = max(peak, cum)
        cumseries[tt-1] = cum
        dd = (peak - cum) / (abs(peak) + 1e-9) if peak > 0 else 0.0
        if dd > dd_thresh:
            POS[:, tt-1] = np.round(POS_full[:, tt-1] * scale_down)
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    return POS

report("drawdown throttle (2%, x0.5)", build_drawdown_throttle(0.02, 0.5), base_scs)
report("drawdown throttle (5%, x0.5)", build_drawdown_throttle(0.05, 0.5), base_scs)

print("\n### Item 59: rank-based position sizing (size by cross-sectional rank of |wz|, bounded) ###")
from scipy.stats import rankdata
def build_rank_sized():
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = full_wz(k)
        rank = (rankdata(np.abs(wz)) - 1) / (n - 1)  # 0..1
        scale = 0.5 + 0.5 * rank  # bounded 0.5x-1.0x, never fully zero
        POS[1:, k] = np.clip(scale * np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS

report("rank-based sizing (0.5-1.0x)", build_rank_sized(), base_scs)

print("\n### Item 60: confidence-threshold partial sizing (saturating ramp above flip point) ###")
def build_confidence_sized(ramp=0.5):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = full_wz(k)
        scale = np.clip(np.abs(wz) / ramp, 0.3, 1.0)
        POS[1:, k] = np.clip(scale * np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS

for ramp in (0.25, 0.5, 1.0):
    report(f"confidence-ramp sizing (ramp={ramp})", build_confidence_sized(ramp), base_scs)

print("\nCategory C batch 1 (sizing schemes) complete.")

print("\n### Item 43: portfolio vol-targeting (scale whole idio book to hit a constant trailing vol) ###")
def build_voltarget(target_vol=0.01, look=60):
    POS_full = build_pos_sign()
    POS = POS_full.copy()
    booked_ret = np.zeros(nt)
    for tt in range(1, nt):
        booked_ret[tt] = float((POS_full[:, tt-1] * (P_[:, tt] - P_[:, tt-1])).sum())
    for k in range(SAFE.WARMUP + look, nt):
        trail = booked_ret[k-look:k]
        cur_vol = trail.std() + 1e-6
        scale = np.clip(target_vol * 100000 / cur_vol, 0.3, 2.0)
        POS[:, k] = np.round(POS_full[:, k] * scale)
    return POS

report("vol-target (target=0.01x100k)", build_voltarget(0.01), base_scs)

print("\n### Item 53: vol-regime-scaled BOOST_K (scale boost strength by market-wide vol) ###")
def build_volscaled_boost(k_lo=1.0, k_hi=2.0):
    POS = np.zeros((nInst, nt))
    algo_vol = np.full(nt, np.nan)
    r0 = r[0]
    for t in range(20, nt-1):
        algo_vol[t] = r0[max(0,t-20):t].std()
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ_SHIP[k].copy()
        if k >= BOOST_MIN_DAY:
            a = max(20, k-250)
            vhist = algo_vol[a:k]
            vhist = vhist[~np.isnan(vhist)]
            cur_v = algo_vol[k-1] if not np.isnan(algo_vol[k-1]) else np.nanmean(vhist)
            pct = (vhist < cur_v).mean() if len(vhist) else 0.5
            k_use = k_lo + (k_hi - k_lo) * pct
            for j, bv in BOOST_AT[k].items():
                wz[j] += k_use * bv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS

report("vol-scaled BOOST_K (1.0-2.0x)", build_volscaled_boost(1.0, 2.0), base_scs)

print("\n### Item 55: same-sign persistence bonus (larger size the longer a position holds its sign) ###")
def build_persistence_bonus(max_bonus=1.5, ramp_days=20):
    POS = np.zeros((nInst, nt))
    streak = np.zeros(n)
    prev_sign = np.zeros(n)
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = full_wz(k)
        sgn = np.sign(wz)
        streak = np.where(sgn == prev_sign, streak + 1, 0)
        bonus = 1.0 + (max_bonus - 1.0) * np.clip(streak / ramp_days, 0, 1)
        POS[1:, k] = np.clip(bonus * sgn * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
        prev_sign = sgn
    POS[0, :] = algo_pos
    return POS

report("persistence bonus (1.5x over 20d)", build_persistence_bonus(1.5, 20), base_scs)

print("\n### Item 56: newly-flipped position discount (ramp size UP over a few days after a flip) ###")
def build_flip_discount(min_scale=0.4, ramp_days=5):
    POS = np.zeros((nInst, nt))
    streak = np.zeros(n)
    prev_sign = np.zeros(n)
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = full_wz(k)
        sgn = np.sign(wz)
        streak = np.where(sgn == prev_sign, streak + 1, 0)
        scale = min_scale + (1.0 - min_scale) * np.clip(streak / ramp_days, 0, 1)
        POS[1:, k] = np.clip(scale * sgn * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
        prev_sign = sgn
    POS[0, :] = algo_pos
    return POS

report("flip discount (0.4x ramping over 5d)", build_flip_discount(0.4, 5), base_scs)

print("\nCategory C batch 2 complete.")
