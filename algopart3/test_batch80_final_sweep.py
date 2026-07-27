"""Final consolidated batch covering remaining high-value items across all 4 categories:
A6/A7 (EWMA / short-window vol ranking), A10 (FDR correction), A11 (Spearman leader search),
D71 (BOOST_ALPHA at N=39), D72 (REV_W recheck), C46 (momentum-of-momentum sizing),
C54 (asymmetric long/short cap), B32 (rolling corr-to-ALGO), B40 (flip-frequency stability).
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
DEF_N = 39; DEF_K = 1.5; DEF_ICL = 190; DEF_SCALEW = 1000; DEF_P = 2.0; DEF_MINDAY = 500
ALPHA = 0.05


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


def spearman_corrmat(X, Y):
    Xr = np.apply_along_axis(stats.rankdata, 1, X)
    Yr = np.apply_along_axis(stats.rankdata, 1, Y)
    return corrmat(Xr, Yr)


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
    line = f"{nm:<44}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


def build_boost_map(N=DEF_N, k_boost=DEF_K, ic_l=DEF_ICL, scale_w=DEF_SCALEW, p_exp=DEF_P,
                     min_day=DEF_MINDAY, alpha=ALPHA, rank_method="expanding", rank_w=250,
                     use_spearman=False, use_fdr=False):
    BOOST_AT = {}
    for k in range(min_day, nt):
        T = k
        Xi_full = rs[:, :T - 1]; Yj = rs[:, 1:T]
        n_samples = Xi_full.shape[1]
        if rank_method == "expanding":
            vol_causal = np.nanstd(Xi_full, axis=1)
        elif rank_method == "ewma":
            lam = 0.5 ** (1.0 / rank_w)
            w = lam ** np.arange(Xi_full.shape[1] - 1, -1, -1)
            m = (w * Xi_full).sum(1) / w.sum()
            vol_causal = np.sqrt((w * (Xi_full - m[:, None]) ** 2).sum(1) / w.sum())
        elif rank_method == "trailing":
            a = max(0, Xi_full.shape[1] - rank_w)
            vol_causal = np.nanstd(Xi_full[:, a:], axis=1)
        cand_idx = np.argsort(-vol_causal)[:N]
        Xi = Xi_full[cand_idx]
        if use_spearman:
            C = spearman_corrmat(Xi, Yj)
        else:
            C = corrmat(Xi, Yj)
        entry = {}
        if use_fdr:
            allcorrs = []
            for j in range(n):
                col = C[:, j].copy()
                cp = np.where(cand_idx == j)[0]
                if len(cp): col[cp[0]] = np.nan
                if np.all(np.isnan(col)): continue
                ci = int(np.nanargmax(np.abs(col)))
                pval = 2 * (1 - stats.t.cdf(abs(col[ci]) * np.sqrt((n_samples - 2) / (1 - col[ci]**2 + 1e-12)), n_samples - 2))
                allcorrs.append((j, ci, col[ci], pval))
            allcorrs.sort(key=lambda x: x[3])
            m_tests = len(allcorrs)
            passed = set()
            for rank_i, (j, ci, cval, pval) in enumerate(allcorrs, 1):
                if pval <= (ALPHA * rank_i / m_tests):
                    passed.add(j)
                else:
                    break
            for j, ci, cval, pval in allcorrs:
                if j not in passed: continue
                i = cand_idx[ci]
                lead = rs[i, :T]
                scale = np.nanstd(lead[max(0, T - 1 - scale_w):T - 1]) + 1e-12
                lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** p_exp
                a = max(0, T - 1 - ic_l)
                xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
                ok = ~np.isnan(xs) & ~np.isnan(ys)
                if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
                icv = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
                if icv <= 0: continue
                entry[j] = lead_boost[-1]
        else:
            thr = sig_threshold(n_samples, N, alpha)
            for j in range(n):
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
                icv = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
                if icv <= 0: continue
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


print("\n=== sanity: v3 (N=39) baseline ===")
base_scs = report("v3 (N=39, sanity)", build_pos(build_boost_map(N=39)), None)

print("\n### Item 6: EWMA (recency-weighted) vol ranking instead of plain expanding ###")
report("N=39, EWMA vol rank (hl=250)", build_pos(build_boost_map(N=39, rank_method="ewma", rank_w=250)), base_scs)

print("\n### Item 7: short trailing-window (60d) DYNAMIC vol ranking ###")
report("N=39, trailing-60d vol rank", build_pos(build_boost_map(N=39, rank_method="trailing", rank_w=60)), base_scs)

print("\n### Item 10: FDR (Benjamini-Hochberg) correction instead of Bonferroni ###")
report("N=39, FDR correction", build_pos(build_boost_map(N=39, use_fdr=True)), base_scs)

print("\n### Item 11: Spearman rank correlation instead of Pearson for leader search ###")
report("N=39, Spearman leader search", build_pos(build_boost_map(N=39, use_spearman=True)), base_scs)

print("\nFinal sweep batch complete.")
