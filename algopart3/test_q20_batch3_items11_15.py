"""Batch 3, items 11-15 [MECH]: portfolio-construction and ensemble/meta mechanics variants.
All validated against the current shipped SAFE_llboost.py on the full five-metric bar + n_worse/61.
11. Basket-level cluster trading: trade the 12-stock cluster as one equal-weighted basket sized by
    its own aggregate ridge signal, instead of 49 independent per-stock signals for those 12 names.
12. Rebalancing-threshold rule: only flip a stock's position sign if signal and yesterday's signal
    disagree by more than a threshold magnitude (hysteresis on the flip TIMING, not size).
13. Performance-weighted HALF_LIVES ensemble: weight each half-life's forecast by its own recent
    trailing IC instead of simple-averaging.
14. Reverse-direction boost check: for each stock J's identified leader I, does J's return ALSO
    predict I's future return (bidirectional, much lower bar than "reciprocal pairs").
15. Two-model turnover rule: only rebalance when ridge and boost's own implied direction agree on
    the CHANGE, not just the level.
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

CLUSTER = [1, 3, 11, 14, 20, 27, 28, 33, 34, 42, 44, 46]

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


print("=== precompute (fixed): shipped ridge WZ + significance-gated boost + ALGO leg ===")
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

n = rs.shape[0]
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
        if abs(col[i]) <= thr:
            continue
        lead = rs[i, :T]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        entry[j] = (i, lead_boost[-1])
    BOOST_AT[k] = entry
print("  boost map done")

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("  ALGO leg done")

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<32}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


def build_pos(wz_override=None, flip_thresh=0.0, turnover_rule=False):
    """wz_override(k, wz) -> wz (modifies in place or returns new); flip_thresh: hysteresis on
    sign flips; turnover_rule: only rebalance when ridge-only sign and boosted sign agree on change."""
    POS = np.zeros((nInst, nt))
    prev_sign = np.zeros(nInst - 1)
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ_SHIP[k].copy()
        wz_ridge_only_sign = np.sign(wz)
        if k >= BOOST_MIN_DAY:
            for j, (i, bv) in BOOST_AT[k].items():
                wz[j] += BOOST_K * bv
        if wz_override is not None:
            wz = wz_override(k, wz)
        newsign = np.sign(wz)
        if flip_thresh > 0:
            flip = (newsign != prev_sign) & (np.abs(wz) < flip_thresh)
            newsign = np.where(flip, prev_sign, newsign)
        if turnover_rule:
            would_change = newsign != prev_sign
            disagree = would_change & (np.sign(wz_ridge_only_sign) != newsign) & (prev_sign != 0)
            newsign = np.where(disagree, prev_sign, newsign)
        prev_sign = newsign
        POS[1:, k] = np.clip(newsign * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== baseline: current shipped SAFE_llboost (reconstructed) ===")
base_scs = report("shipped (reconstructed)", build_pos())

print("\n### Item 12: rebalancing-threshold hysteresis (only flip if |signal| exceeds threshold) ###")
for th in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
    report(f"flip_thresh={th}", build_pos(flip_thresh=th), base_scs)

print("\n### Item 15: two-model turnover rule (only rebalance when ridge-only and boosted agree on flip) ###")
report("turnover_rule=True", build_pos(turnover_rule=True), base_scs)

print("\n### Item 11: basket-level cluster trading (trade the 12-stock cluster as one signal) ###")
CLUSTER_IDIO = [c - 1 for c in CLUSTER]  # index into wz (0..49, idio-only)


def wz_basket(k, wz):
    wz2 = wz.copy()
    avg = wz2[CLUSTER_IDIO].mean()
    wz2[CLUSTER_IDIO] = avg  # uniform signal (same sign+magnitude) for the whole basket
    return wz2


report("basket cluster (uniform avg signal)", build_pos(wz_override=wz_basket), base_scs)


def wz_basket_sign_only(k, wz):
    wz2 = wz.copy()
    avg_sign = np.sign(wz2[CLUSTER_IDIO].mean())
    mag = np.abs(wz2[CLUSTER_IDIO]).mean()
    wz2[CLUSTER_IDIO] = avg_sign * mag  # same direction, keep typical magnitude
    return wz2


report("basket cluster (sign of avg, avg magnitude)", build_pos(wz_override=wz_basket_sign_only), base_scs)

print("\n### Item 13: performance-weighted HALF_LIVES ensemble (weight by trailing pooled IC) ###")
t0 = time.time()
FI_ARR = {hl: np.full((nt, 50), np.nan) for hl in SAFE.HALF_LIVES}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        FI_ARR[hl][t] = fi / (fi.std() + 1e-12)
print(f"  per-half-life forecasts done ({time.time()-t0:.0f}s)")

IC_LOOKBACK = 120


def build_wz_perf_weighted():
    WZ = {}
    for t in range(SAFE.WARMUP, nt):
        a = max(SAFE.WARMUP, t - IC_LOOKBACK)
        ics = []
        for hl in SAFE.HALF_LIVES:
            if t - a < 20:
                ics.append(0.0); continue
            X = FI_ARR[hl][a:t].ravel()
            Y = rs[:, a:t].T.ravel()
            ok = ~np.isnan(X) & ~np.isnan(Y)
            if ok.sum() < 100 or X[ok].std() < 1e-12:
                ics.append(0.0)
            else:
                ics.append(float(np.corrcoef(X[ok], Y[ok])[0, 1]))
        w = np.clip(np.array(ics), 0, None)
        if w.sum() < 1e-9:
            w = np.full(len(SAFE.HALF_LIVES), 1.0 / len(SAFE.HALF_LIVES))
        else:
            w = w / w.sum()
        wz = sum(w[i] * FI_ARR[hl][t] for i, hl in enumerate(SAFE.HALF_LIVES))
        if SAFE.BLEND > 0:
            rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
            rv_ = rv_ - rv_.mean()
            rv = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
        WZ[t] = wz
    return WZ


def build_pos_from_wz_dict(WZ):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= BOOST_MIN_DAY:
            for j, (i, bv) in BOOST_AT[k].items():
                wz[j] += BOOST_K * bv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


t0 = time.time()
WZ_PERF = build_wz_perf_weighted()
report(f"perf-weighted half-lives ({time.time()-t0:.0f}s)", build_pos_from_wz_dict(WZ_PERF), base_scs)
