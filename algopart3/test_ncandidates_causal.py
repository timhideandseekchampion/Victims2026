"""Redo of the item-18 candidate-pool restriction test, FIXED for look-ahead bias: the vol_level
ranking used to pick the "top-N" candidate pool must be computed causally (trailing window as of
each day), not from the full-sample return series. Re-validates whether the earlier N=29-47
'stable region' survives once this is fixed.
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


def sig_threshold(n_samples, n_candidates):
    alpha_adj = ALPHA / n_candidates
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== precompute (fixed): shipped ridge WZ + ALGO leg ===")
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


def build_boost_map_causal(N, use_full_pool_baseline=False):
    """N=49 (or use_full_pool_baseline) reproduces the shipped mechanism exactly. Otherwise,
    each day, rank candidate stocks by their CAUSAL trailing realized vol (using only rs[:, :T-1],
    i.e. data known as of 'today') and restrict the leader-search pool to the top-N."""
    BOOST_AT = {}
    for k in range(BOOST_MIN_DAY, nt):
        T = k
        Xi_full = rs[:, :T - 1]; Yj = rs[:, 1:T]
        n_samples = Xi_full.shape[1]
        if use_full_pool_baseline:
            candidate_idx = np.arange(n)
            thr = sig_threshold(n_samples, 49)
        else:
            vol_causal = np.nanstd(Xi_full, axis=1)  # causal: only data through day T-2
            candidate_idx = np.argsort(-vol_causal)[:N]
            thr = sig_threshold(n_samples, N)
        Xi = Xi_full[candidate_idx]
        C = corrmat(Xi, Yj)
        entry = {}
        for j in range(n):
            col = C[:, j].copy()
            if j in candidate_idx:
                pos_in_cand = list(candidate_idx).index(j)
                col[pos_in_cand] = np.nan
            if np.all(np.isnan(col)):
                continue
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) <= thr:
                continue
            i = candidate_idx[ci]
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
            entry[j] = lead_boost[-1]
        BOOST_AT[k] = entry
    return BOOST_AT


def build_pos(BOOST_AT):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ_SHIP[k].copy()
        if k >= BOOST_MIN_DAY:
            for j, bv in BOOST_AT[k].items():
                wz[j] += BOOST_K * bv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity: full pool (N=49) must reproduce the exact shipped baseline ===")
BOOST_BASE = build_boost_map_causal(49, use_full_pool_baseline=True)
base_scs = report("shipped (sanity check)", build_pos(BOOST_BASE), None)

print("\n=== causal top-N candidate pool sweep ===")
for N in (20, 25, 29, 30, 35, 39, 40, 45):
    t0 = time.time()
    BOOST = build_boost_map_causal(N)
    POS = build_pos(BOOST)
    report(f"causal top-{N}-vol ({time.time()-t0:.0f}s)", POS, base_scs)
