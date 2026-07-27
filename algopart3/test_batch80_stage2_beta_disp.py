"""Stage 2 for Category B's two Stage-1 survivors: beta-stability (item 26, perm p=0.000) and
cross-sectional dispersion (item 25, perm p=0.003). Test whether either, blended into the idio
wz score as a small additional tilt, improves the scored backtest vs the shipped baseline.
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


print("=== shared precompute: significance boost + ALGO leg ===")
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

# causal beta_change and dispersion features (recomputed each day using only data through "today")
BETA_W = 60
print("computing causal rolling beta-to-ALGO per stock ...")
beta_roll = np.full((n, nt - 1), np.nan)
r0 = r[0]
for j in range(n):
    for t in range(BETA_W, nt - 1):
        x = r0[t - BETA_W:t]; y = rs[j, t - BETA_W:t]
        if x.std() < 1e-12: continue
        beta_roll[j, t] = np.cov(x, y)[0, 1] / (x.var() + 1e-12)
beta_change = np.full((n, nt - 1), np.nan)
beta_change[:, 1:] = np.diff(beta_roll, axis=1)
stability_feat = -np.abs(beta_change)  # higher = more stable

disp_feat = np.nanstd(rs, axis=0)  # (nt-1,) cross-sectional dispersion each day

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


def wz_shipped(t):
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
    return wz


WZ_SHIP = {t: wz_shipped(t) for t in range(SAFE.WARMUP, nt)}
print("  WZ done")


def combine_and_score(extra_tilt_fn=None, tilt_w=0.0):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ_SHIP[k].copy()
        if k >= BOOST_MIN_DAY:
            for j, bv in BOOST_AT[k].items():
                wz[j] += BOOST_K * bv
        if extra_tilt_fn is not None and tilt_w > 0:
            tilt = extra_tilt_fn(k)
            if tilt is not None:
                tz = tilt - np.nanmean(tilt)
                tz = tz / (np.nanstd(tz) + 1e-12)
                tz = np.nan_to_num(tz)
                wz = (1 - tilt_w) * wz + tilt_w * tz
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity: baseline ===")
base_scs = report("shipped (baseline)", combine_and_score())

print("\n### Stage 2: beta-stability tilt (item 26, Stage-1 p=0.000) ###")
def beta_tilt(k):
    if k - 1 >= beta_change.shape[1] or k - 1 < 0: return None
    return stability_feat[:, k - 1]

for w in (0.02, 0.05, 0.1, 0.15, 0.2):
    report(f"beta-stability tilt w={w}", combine_and_score(beta_tilt, w), base_scs)

print("\n### Stage 2: cross-sectional dispersion as a UNIFORM long-bias tilt (item 25, Stage-1 p=0.003) ###")
print("    Note: this signal predicts the AVERAGE next-day return across stocks, not which stocks")
print("    to prefer -- so it's structurally a book-level/market-timing signal, not a per-stock")
print("    differentiator. Testing it as a uniform additive tilt anyway (weak effect expected).")
disp_z = (disp_feat - np.nanmean(disp_feat)) / (np.nanstd(disp_feat) + 1e-12)


def disp_tilt(k):
    if k - 1 >= len(disp_z) or k - 1 < 0: return None
    return np.full(n, disp_z[k - 1])


for w in (0.02, 0.05, 0.1):
    report(f"dispersion uniform-tilt w={w}", combine_and_score(disp_tilt, w), base_scs)

print("\nStage 2 batch complete.")
