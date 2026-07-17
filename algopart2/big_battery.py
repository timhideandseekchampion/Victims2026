"""
big_battery.py — exhaustive multi-domain test battery on days 400-750, hunting for ANY signal
that beats the ridge+blend IC of 0.079. Each block names its MATH AREA and reports mean daily
cross-sectional IC (+ t) vs next-day return, all causal. Baseline to beat: IC 0.0791 (t~8.4).

Prior (why most exotic families should be NULL): the DGP is linear-Gaussian, one-factor, with a
directed linear lead-lag + linear OU reversion. Nonlinear / tail / spectral / entropy methods have
no structure to grip. The one family with real upside is BETTER LINEAR ESTIMATION of the 51x50
lead-lag matrix (RMT denoising, shrinkage, reduced-rank) — tested first.
"""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
S, E = 400, 749

def ic_stats(sig_fn, step=1):
    ics = []
    for d in range(S, E, step):
        s = sig_fn(d)
        if s is None: continue
        fwd = RET[1:, d]
        if s.std() < 1e-12 or fwd.std() < 1e-12: continue
        ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics)
    if len(ics) < 5: return np.nan, np.nan
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))

def wmats(d, hl):
    """weighted centered predictor/target design for day d (causal)."""
    X = RET[:, :d - 1].T; Y = RET[1:, 1:d].T; xin = RET[:, d - 1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    return Xc, Yc, w, mx, my, xin

def base_ridge(d, hl=1000, a=0.3):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = wmats(d, hl)
    p = Xc.shape[1]
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    f = my + (xin - mx) @ B; return f - f.mean()

print(f"battery on days {S}-{E}. baseline ridge+blend IC=0.0791.\n")
results = {}
def report(name, area, fn, step=1):
    ic, t = ic_stats(fn, step)
    results[name] = ic
    flag = "  <-- beats baseline" if ic > 0.0791 else ""
    print(f"  {name:<34}{area:<22} IC={ic:7.4f}  t={t:5.2f}{flag}")

print("[AREA 1] LINEAR ESTIMATION of the lead-lag matrix (the measured bottleneck)")
report("ridge baseline (hl1000,a0.3)", "ridge regression", base_ridge)

# 1a. Random Matrix Theory — Marchenko-Pastur eigenvalue clipping of predictor covariance
def ridge_rmt(d, hl=1000, a=0.05):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = wmats(d, hl)
    n, p = Xc.shape
    Xw = np.sqrt(w)[:, None] * Xc
    C = (Xw.T @ Xw) / w.sum()                                # weighted cov (p x p)
    ev, V = np.linalg.eigh(C)
    q = p / n; lam_plus = (1 + np.sqrt(q)) ** 2 * np.median(ev) / (np.median(ev) or 1)
    # MP upper edge with sigma^2 ~ mean of bulk; clip sub-edge eigenvalues to their average
    sig2 = ev.mean(); edge = sig2 * (1 + np.sqrt(q)) ** 2
    bulk = ev < edge
    if bulk.any(): ev[bulk] = ev[bulk].mean()
    Cd = (V * ev) @ V.T
    XtWY = Xc.T @ (w[:, None] * Yc)
    B = np.linalg.solve(Cd * w.sum() + a * np.trace(Cd) / p * np.eye(p), XtWY)
    f = my + (xin - mx) @ B; return f - f.mean()
report("RMT eigenvalue-clip ridge", "random matrix theory", ridge_rmt)

# 1b. Ledoit-Wolf shrinkage of predictor covariance toward scaled identity
def ridge_lw(d, hl=1000):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = wmats(d, hl)
    n, p = Xc.shape
    Xw = np.sqrt(w)[:, None] * Xc; sw = w.sum()
    C = (Xw.T @ Xw) / sw
    mu = np.trace(C) / p; F = mu * np.eye(p)
    d2 = ((C - F) ** 2).sum()
    b2 = 0.0
    for i in range(n):
        xi = Xw[i][:, None]; ci = xi @ xi.T / (w[i] + 1e-12)
        b2 += ((ci - C) ** 2).sum()
    b2 = min(b2 / n**2, d2)
    shr = b2 / (d2 + 1e-12)
    Cs = shr * F + (1 - shr) * C
    XtWY = Xc.T @ (w[:, None] * Yc)
    B = np.linalg.solve(Cs * sw + 1e-6 * np.eye(p), XtWY)
    f = my + (xin - mx) @ B; return f - f.mean()
report("Ledoit-Wolf shrinkage ridge", "shrinkage estimation", ridge_lw, step=2)

# 1c. Reduced-rank regression (truncate the coefficient matrix's rank)
def ridge_rr(d, hl=1000, a=0.3, rank=10):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = wmats(d, hl); p = Xc.shape[1]
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    B = np.linalg.solve(XtWX + a * np.eye(p), XtWY)
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    s[rank:] = 0; Br = (U * s) @ Vt
    f = my + (xin - mx) @ Br; return f - f.mean()
for rk in (5, 10, 20):
    report(f"reduced-rank ridge (r={rk})", "reduced-rank regression", lambda d, rk=rk: ridge_rr(d, rank=rk), step=2)

# 1d. sparse lead-lag: keep only top-k predictor coefs per name (LASSO-ish hard threshold)
def ridge_sparse(d, hl=1000, a=0.3, keep=8):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = wmats(d, hl); p = Xc.shape[1]
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    B = np.linalg.solve(XtWX + a * np.eye(p), XtWY)
    for j in range(B.shape[1]):
        col = B[:, j]; thr = np.sort(np.abs(col))[::-1][min(keep, p - 1)]
        col[np.abs(col) < thr] = 0
    f = my + (xin - mx) @ B; return f - f.mean()
report("sparse top-8 lead-lag", "sparse regression", ridge_sparse, step=2)

print("\n[AREA 2] STAT-ARB / mean-reversion variants")
# 2a. Avellaneda-Lee: PCA factor residual OU s-score
def avellaneda_lee(d, lookback=60, npc=1):
    if d < lookback + 5: return None
    R = RET[1:, d - lookback:d].T                           # (lookback, 50)
    Rc = R - R.mean(0)
    U, s, Vt = np.linalg.svd(np.cov(Rc.T))
    F = Rc @ U[:, :npc]                                     # factor returns
    # regress each name on factors, take residual, OU s-score of cumulative residual
    scores = np.zeros(50)
    for i in range(50):
        beta = np.linalg.lstsq(np.c_[np.ones(lookback), F], Rc[:, i], rcond=None)[0]
        resid = Rc[:, i] - (np.c_[np.ones(lookback), F] @ beta)
        x = np.cumsum(resid)
        if x[:-1].std() < 1e-9: continue
        b = np.polyfit(x[:-1], x[1:], 1)[0]
        m = x.mean()
        scores[i] = -(x[-1] - m) / (x.std() + 1e-9)         # negative s-score = revert
    return scores - scores.mean()
report("Avellaneda-Lee PCA-OU s-score", "stat-arb / OU", avellaneda_lee, step=2)

# 2b. vol-scaled reversion (risk-normalized)
def volscaled_rev(d, w=10):
    if d < w + 21: return None
    r = lp[1:, d] - lp[1:, d - w]; r = r - r.mean()
    vol = RET[1:, d - 20:d].std(1) + 1e-9
    s = -(r / vol); return s - s.mean()
report("vol-scaled reversion", "risk normalization", volscaled_rev)

print("\n[AREA 3] NONLINEAR / RANK / higher-moment (expected NULL on Gaussian data)")
# 3a. rank-transform ridge (Spearman-style, robust to tails)
def ridge_rank(d, hl=1000, a=0.3):
    if d < 95: return None
    from scipy.stats import rankdata
    Xc, Yc, w, mx, my, xin = wmats(d, hl); p = Xc.shape[1]
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    B = np.linalg.solve(XtWX + a * np.eye(p), XtWY)
    f = my + (xin - mx) @ B
    return None if f.std() < 1e-12 else (rankdata(f) - 25.5)
report("rank-IC of ridge (Spearman)", "rank statistics", ridge_rank, step=2)

# 3b. squared-return (nonlinear) predictor augmentation
def ridge_nl(d, hl=1000, a=0.3):
    if d < 95: return None
    X = RET[:, :d - 1].T; Y = RET[1:, 1:d].T; xin = RET[:, d - 1]
    X2 = np.c_[X, X ** 2]; xin2 = np.r_[xin, xin ** 2]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X2).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X2 - mx; Yc = Y - my; p = Xc.shape[1]
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(p), Xc.T @ (w[:, None] * Yc))
    f = my + (xin2 - mx) @ B; return f - f.mean()
report("nonlinear (+squared) ridge", "polynomial features", ridge_nl, step=2)

print("\n[AREA 4] COMBINATION of the two best (ridge estimation variants)")
def best_combo(d):
    a = base_ridge(d); b = ridge_rmt(d)
    if a is None or b is None: return None
    az = a / (a.std() + 1e-12); bz = b / (b.std() + 1e-12)
    r = lp[1:, d] - lp[1:, d - 5]; r = r - r.mean(); z = -r / (r.std() + 1e-12)
    return 0.5 * az + 0.3 * bz + 0.2 * z
report("ridge + RMT + revz5 combo", "signal blending", best_combo, step=1)

print("\nSUMMARY (sorted):")
for k, v in sorted(results.items(), key=lambda x: -(x[1] if x[1] == x[1] else -9)):
    print(f"  {v:7.4f}  {k}")
best = max((v, k) for k, v in results.items() if v == v)
print(f"\nBEST: {best[1]} @ IC {best[0]:.4f}  (baseline 0.0791)")
print("beats baseline" if best[0] > 0.0791 else "-> nothing beat the ridge+blend baseline of 0.0791")
