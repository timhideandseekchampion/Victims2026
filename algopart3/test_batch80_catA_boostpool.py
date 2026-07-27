"""Batch of 80, Category A (items 1-20): refinements/extensions of the validated v3 boost-pool
restriction (N=39 highest-vol idio stocks as candidate leaders). Shared precompute: WZ_SHIP
(ridge+blend forecast) and algo_pos (ALGO leg), both unaffected by any boost-mechanism variant
tested here.
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

ALPHA = 0.05
DEF_N = 39
DEF_K = 1.5
DEF_ICL = 190
DEF_SCALEW = 1000
DEF_P = 2.0
DEF_MINDAY = 500


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


def sig_threshold(n_samples, n_candidates, alpha=ALPHA):
    if n_samples < 10: return 1.0
    alpha_adj = alpha / n_candidates
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== shared precompute: shipped ridge WZ + ALGO leg ===")
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
    line = f"{nm:<40}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


def build_boost_map(N=DEF_N, k_boost=DEF_K, ic_l=DEF_ICL, scale_w=DEF_SCALEW, p_exp=DEF_P,
                     min_day=DEF_MINDAY, alpha=ALPHA, restrict_followers=False, n_thresh=None):
    BOOST_AT = {}
    if n_thresh is None:
        n_thresh = N
    for k in range(min_day, nt):
        T = k
        Xi_full = rs[:, :T - 1]; Yj = rs[:, 1:T]
        n_samples = Xi_full.shape[1]
        vol_causal = np.nanstd(Xi_full, axis=1)
        cand_idx = np.argsort(-vol_causal)[:N]
        thr = sig_threshold(n_samples, n_thresh, alpha)
        Xi = Xi_full[cand_idx]
        C = corrmat(Xi, Yj)
        entry = {}
        follower_set = set(cand_idx) if restrict_followers else set(range(n))
        for j in range(n):
            if j not in follower_set:
                continue
            col = C[:, j].copy()
            cp = np.where(cand_idx == j)[0]
            if len(cp): col[cp[0]] = np.nan
            if np.all(np.isnan(col)): continue
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) <= thr: continue
            i = cand_idx[ci]
            lead = rs[i, :T]
            scale = np.nanstd(lead[max(0, T - 1 - scale_w):T - 1]) + 1e-12
            lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** p_exp
            a = max(0, T - 1 - ic_l)
            xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
            ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
            if ic <= 0: continue
            entry[j] = lead_boost[-1]
        BOOST_AT[k] = entry
    return BOOST_AT


def build_pos(BOOST_AT, k_boost=DEF_K, min_day=DEF_MINDAY):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ_SHIP[k].copy()
        if k >= min_day:
            for j, bv in BOOST_AT[k].items():
                wz[j] += k_boost * bv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity: baselines ===")
BOOST_SHIP = build_boost_map(N=49)
base_scs = report("shipped (N=49, sanity)", build_pos(BOOST_SHIP), None)
BOOST_V3 = build_boost_map(N=39)
v3_scs = report("v3 (N=39, sanity)", build_pos(BOOST_V3), base_scs)

print("\n### items 1-4: re-tune BOOST_K, IC_L, SCALE_W, P at N=39 ###")
for k_b in (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5):
    report(f"N=39, BOOST_K={k_b}", build_pos(build_boost_map(N=39, k_boost=k_b), k_boost=k_b), base_scs)
for icl in (130, 160, 190, 220, 250, 300):
    report(f"N=39, BOOST_IC_L={icl}", build_pos(build_boost_map(N=39, ic_l=icl)), base_scs)
for sw in (500, 750, 1000, 1250, 1500):
    report(f"N=39, BOOST_SCALE_W={sw}", build_pos(build_boost_map(N=39, scale_w=sw)), base_scs)
for p_e in (1.0, 1.5, 2.0, 2.5, 3.0):
    report(f"N=39, BOOST_P={p_e}", build_pos(build_boost_map(N=39, p_exp=p_e)), base_scs)

print("\n### item 5: re-tune BOOST_MIN_DAY at N=39 ###")
for md in (450, 480, 500, 520, 550):
    report(f"N=39, BOOST_MIN_DAY={md}", build_pos(build_boost_map(N=39, min_day=md), min_day=md), base_scs)

print("\n### item 9: restrict followers to top-39 pool too ###")
report("N=39, followers restricted too", build_pos(build_boost_map(N=39, restrict_followers=True)), base_scs)

print("\n### item 18: fixed strict N=49 threshold, but N=39 search pool (pool-only effect, re-verify) ###")
report("N=39 pool, thr uses n=49 (strict)", build_pos(build_boost_map(N=39, n_thresh=49)), base_scs)

print("\ndone with core parameter re-tuning at N=39.")
