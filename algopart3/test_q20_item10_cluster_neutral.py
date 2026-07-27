"""Item 10 [MECH]: cluster-neutral construction. Uses the already-validated 12-stock correlated
cluster (test_stock_clustering.py, kmeans k=6 cluster 1, avg_within_corr=+0.13, permutation
p=0%) not for alpha but as a RISK CONSTRAINT: after computing the normal shipped positions,
neutralize the book's net dollar exposure to that cluster by spreading an equal offsetting
adjustment across its 12 members. A risk-control philosophy, not a return-seeking one.
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

CLUSTER = [1, 3, 11, 14, 20, 27, 28, 33, 34, 42, 44, 46]  # price-array row indices (validated 12-stock cluster)

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
        entry[j] = lead_boost[-1]
    BOOST_AT[k] = entry
print(f"  boost map done ({time.time()-t0:.0f}s)")

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("  ALGO leg done")

POS_BASE = np.zeros((nInst, nt))
for k in range(SAFE.WARMUP, nt):
    cur = P_[:, k]; lim = (dlr / cur).astype(int)
    wz = WZ_SHIP[k].copy()
    if k >= BOOST_MIN_DAY:
        for j, bv in BOOST_AT[k].items():
            wz[j] += BOOST_K * bv
    POS_BASE[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
POS_BASE[0, :] = algo_pos
print("  shipped baseline book done")

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<28}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


print("\n=== baseline: current shipped SAFE_llboost (reconstructed) ===")
base_scs = report("shipped (reconstructed)", POS_BASE)


def cluster_neutralize(POS, band_dollars):
    """for each day, if |net $ exposure of CLUSTER| > band, spread an equal offsetting
    share-adjustment across the 12 members to bring net exposure to +-band."""
    POS2 = POS.copy()
    for k in range(nt):
        cur = P_[CLUSTER, k]
        pos_c = POS2[CLUSTER, k]
        net = float((pos_c * cur).sum())
        if abs(net) <= band_dollars:
            continue
        excess = net - np.sign(net) * band_dollars
        adj_dollars_each = excess / len(CLUSTER)
        adj_shares = adj_dollars_each / cur
        new_pos_c = pos_c - adj_shares
        lim = (dlr[CLUSTER] / cur).astype(int)
        new_pos_c = np.clip(new_pos_c, -lim, lim)
        POS2[CLUSTER, k] = new_pos_c
    return POS2


for band in (0, 20_000, 40_000, 60_000, 80_000):
    t0 = time.time()
    POS_N = cluster_neutralize(POS_BASE, band)
    report(f"cluster-neutral band=${band:,}", POS_N, base_scs)
