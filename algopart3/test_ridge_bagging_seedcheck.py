"""Critical robustness check on test_ridge_bagging.py's apparent K_bags=5, drop_frac=0.05 result:
the K_bags sweep showed a highly erratic, non-monotonic pattern (K=5 much better than its
neighbors K=3 and K=8), which is a red flag for a lucky-seed artifact rather than a real effect.
This reruns the SAME (K_bags=5, drop_frac=0.05) configuration with THREE different, unrelated seed
formulas to check if the result survives changing which specific random masks get drawn.
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


def ewls_ridge_masked(X, Y, hl, a, mask):
    n_, p = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n_ - 1, -1, -1)
    w = w * mask
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


print("=== precompute (shared): significance-gated boost + ALGO leg ===")
t0 = time.time()
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

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def build_pos(bagging=False, K_bags=5, drop_frac=0.2, seed_fn=None):
    POS = np.zeros((nInst, nt))
    for t in range(SAFE.WARMUP, nt):
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        rr = r[:, :t]
        X = rr[:, :-1].T; Y = rr[1:, 1:].T
        xin = rr[:, -1]
        fs = []
        for hl in SAFE.HALF_LIVES:
            if not bagging:
                B, mx, my = SAFE._ewls_ridge(X, Y, hl, SAFE.RIDGE_A)
                pred = my + (xin - mx) @ B
                fi = pred - pred.mean()
                fs.append(fi / (fi.std() + 1e-12))
            else:
                seed = seed_fn(t, hl)
                rng = np.random.default_rng(seed)
                preds = []
                for b in range(K_bags):
                    mask = (rng.random(X.shape[0]) >= drop_frac).astype(float)
                    if mask.sum() < 20:
                        mask[:] = 1.0
                    B, mx, my = ewls_ridge_masked(X, Y, hl, SAFE.RIDGE_A, mask)
                    pred = my + (xin - mx) @ B
                    preds.append(pred)
                pred_avg = np.mean(preds, axis=0)
                fi = pred_avg - pred_avg.mean()
                fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if SAFE.BLEND > 0:
            rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
            rv_ = rv_ - rv_.mean()
            rv = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
        if t >= BOOST_MIN_DAY:
            for j, bv in BOOST_AT[t].items():
                wz[j] += BOOST_K * bv
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


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


print("\n=== baseline ===")
base_scs = report("shipped (no bagging)", build_pos(bagging=False))

print("\n=== SEED SENSITIVITY CHECK: same K_bags=5, drop_frac=0.05, 4 different unrelated seed formulas ===")
seed_formulas = {
    "original (t*1000+hl)": lambda t, hl: t * 1000 + hl,
    "seedB (t*7919+hl*31)": lambda t, hl: t * 7919 + hl * 31,
    "seedC (t*13+hl*997)": lambda t, hl: t * 13 + hl * 997,
    "seedD (hash-like t^2+hl)": lambda t, hl: (t * t + hl * 17) % 2_000_000_000,
}
for name, fn in seed_formulas.items():
    t0 = time.time()
    report(f"K=5 drop=0.05 {name}", build_pos(bagging=True, K_bags=5, drop_frac=0.05, seed_fn=fn), base_scs)
    print(f"  (took {time.time()-t0:.0f}s)")
