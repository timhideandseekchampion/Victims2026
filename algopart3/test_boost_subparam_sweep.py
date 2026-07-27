"""Sub-parameter sweep for SAFE_llboost.py's pairwise boost: BOOST_P (magnitude exponent),
BOOST_SCALE_W (leader's own return-scale window), BOOST_IC_L (sign-check window) were all
inherited from the original exploratory test script without ever being swept -- only BOOST_K and
BOOST_MIN_DAY were validated. This does that, using the SAME fresh-every-day (no stale checkpoint)
design as the shipped module, matching production exactly.

Efficiency: the expensive part (correlation matrix + Bonferroni significance test -> which leader,
if any, qualifies for each stock on each day) does NOT depend on P/scale_w/IC_L at all -- only on
the day's available history. So it's computed ONCE per day and shared across every parameter
combo swept below. Only the ic-sign gate (IC_L-dependent) and the boost magnitude (P, scale_w-
dependent) vary per combo, and those are cheap to recompute.
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
r = logp[:, 1:] - logp[:, :-1]
rs = r[1:]  # idio-stock returns only, (49, T)


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


BOOST_MIN_DAY = 500
ALPHA = 0.05
N_CANDIDATES = 49


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


print("=== precompute (shared across ALL parameter combos): shipped idio WZ, ALGO leg, and the ===")
print("    day-by-day significant-leader map (fresh every day, no stale checkpoints) ===")
t0 = time.time()
WZ = {}
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
    WZ[t] = wz
print(f"  WZ done ({time.time()-t0:.0f}s)")

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("  ALGO leg done")


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


t0 = time.time()
LEADER_AT = {}  # day k -> {j: i} for stocks whose best leader clears the significance threshold
for k in range(BOOST_MIN_DAY, nt):
    T = k  # rs[:, :T] available (idio returns through day k, matching rs used inside getMyPosition at day k)
    Xi = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(nInst - 1):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) > thr:
            entry[j] = i
    LEADER_AT[k] = entry
print(f"  significant-leader map done ({time.time()-t0:.0f}s); "
      f"avg qualifying pairs/day = {np.mean([len(v) for v in LEADER_AT.values()]):.1f}/49")

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def build_pos(K, P_exp, SCALE_W, IC_L):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= BOOST_MIN_DAY:
            T = k
            entry = LEADER_AT[k]
            for j, i in entry.items():
                lead = rs[i, :T]
                scale = np.nanstd(lead[max(0, T - 1 - SCALE_W):T - 1]) + 1e-12
                lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** P_exp
                a = max(0, T - 1 - IC_L)
                xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
                ok = ~np.isnan(xs) & ~np.isnan(ys)
                if ok.sum() < 60 or xs[ok].std() < 1e-12:
                    continue
                ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
                if ic <= 0:
                    continue
                wz[j] += K * lead_boost[-1]
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


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


print("\n=== baseline (K=0, no boost == shipped SAFE_llvol) ===")
base_scs = report("baseline (no boost)", build_pos(0.0, 2.0, 500, 220))

print("\n=== current shipped SAFE_llboost (K=1.5, P=2.0, scale_w=500, IC_L=220) ===")
ship_scs = report("shipped SAFE_llboost", build_pos(1.5, 2.0, 500, 220), base_scs)

print("\n=== 1. sweep BOOST_P (magnitude exponent), K held at 1.5 ===")
for p_exp in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    report(f"P={p_exp}", build_pos(1.5, p_exp, 500, 220), base_scs)

print("\n=== 2. sweep BOOST_SCALE_W (leader's own vol-normalisation window), P=2.0, K=1.5 ===")
for scale_w in (100, 250, 375, 500, 750, 1000):
    report(f"scale_w={scale_w}", build_pos(1.5, 2.0, scale_w, 220), base_scs)

print("\n=== 3. sweep BOOST_IC_L (sign-check window), P=2.0, scale_w=500, K=1.5 ===")
for ic_l in (100, 150, 180, 220, 300, 400):
    report(f"IC_L={ic_l}", build_pos(1.5, 2.0, 500, ic_l), base_scs)

print("\n=== 4. joint refinement: scale_w x IC_L (P=2.0, K=1.5 fixed) ===")
for scale_w in (375, 500, 750, 1000):
    for ic_l in (160, 170, 180, 190, 200):
        report(f"scale_w={scale_w} IC_L={ic_l}", build_pos(1.5, 2.0, scale_w, ic_l), base_scs)

print("\n=== 5. re-check K at the new best point (scale_w=1000, IC_L=190, P=2.0) ===")
for K in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25):
    report(f"K={K} scale_w=1000 IC_L=190", build_pos(K, 2.0, 1000, 190), base_scs)

print("\n=== 6. neighbor-stability check around the winning point (not a lucky corner) ===")
for scale_w in (900, 1000, 1100):
    for ic_l in (180, 190, 200):
        for K in (1.4, 1.5, 1.6):
            report(f"scale_w={scale_w} IC_L={ic_l} K={K}", build_pos(K, 2.0, scale_w, ic_l), base_scs)

print("\n=== 7. structural variant: TOP-2 significant leaders averaged, instead of just the single best ===")
print("    (uses the current best sub-params: P=2.0, scale_w=1000, IC_L=190)")

t0 = time.time()
LEADER_AT2 = {}  # day k -> {j: [i1, i2, ...]} -- ALL leaders clearing significance, not just the argmax
for k in range(BOOST_MIN_DAY, nt):
    T = k
    Xi = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(nInst - 1):
        col = C[:, j].copy(); col[j] = np.nan
        order = np.argsort(-np.abs(col))
        qualifying = [int(i) for i in order if abs(col[i]) > thr][:3]  # up to top-3
        if qualifying:
            entry[j] = qualifying
    LEADER_AT2[k] = entry
print(f"  done ({time.time()-t0:.0f}s); avg qualifying leaders/stock/day = "
      f"{np.mean([len(v) for e in LEADER_AT2.values() for v in e.values()]) if any(LEADER_AT2.values()) else 0:.2f}")


def build_pos_multi(K, P_exp, SCALE_W, IC_L, n_leaders):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= BOOST_MIN_DAY:
            T = k
            entry = LEADER_AT2[k]
            for j, leaders in entry.items():
                vals = []
                for i in leaders[:n_leaders]:
                    lead = rs[i, :T]
                    scale = np.nanstd(lead[max(0, T - 1 - SCALE_W):T - 1]) + 1e-12
                    lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** P_exp
                    a = max(0, T - 1 - IC_L)
                    xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
                    ok = ~np.isnan(xs) & ~np.isnan(ys)
                    if ok.sum() < 60 or xs[ok].std() < 1e-12:
                        continue
                    ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
                    if ic <= 0:
                        continue
                    vals.append(lead_boost[-1])
                if vals:
                    wz[j] += K * float(np.mean(vals))
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


for n_leaders in (1, 2, 3):
    report(f"top-{n_leaders} leader(s) averaged", build_pos_multi(1.5, 2.0, 1000, 190, n_leaders), base_scs)
