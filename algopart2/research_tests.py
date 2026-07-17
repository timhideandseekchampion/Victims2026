"""
research_tests.py — implement the literature's top untested ideas and measure them on 400-750.
  (A) proper reduced-rank RIDGE (Mukherjee-Zhu: project fitted values onto top-k)
  (B) optimal singular-value cleaning of the CROSS-covariance (numerator), index removed
  (C) single-index Ledoit-Wolf target for the denominator
  (D) Lo-MacKinlay accumulated cross-sectional reversion
  (E) SIZING test: w ~ Sigma^-1 alpha (Grinold) vs sign vs inverse-vol  -> effect on SCORE
Baseline IC to beat: 0.0791. Baseline SCORE on 500-750: 604 (sign sizing).
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
S, E = 400, 749
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def ic_stats(sig_fn, step=1):
    ics = []
    for d in range(S, E, step):
        s = sig_fn(d)
        if s is None: continue
        fwd = RET[1:, d]
        if s.std() < 1e-12 or fwd.std() < 1e-12: continue
        ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics)
    return (np.nan, np.nan) if len(ics) < 5 else (ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))))

def design(d, hl):
    X = RET[:, :d - 1].T; Y = RET[1:, 1:d].T; xin = RET[:, d - 1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    return (X - mx), (Y - my), w, mx, my, xin

# ---------- (A) reduced-rank ridge, Mukherjee-Zhu ----------
def rrr_mz(d, hl=1000, a=0.3, k=5):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = design(d, hl); p = Xc.shape[1]
    Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + a * np.eye(p), Xc.T @ (Wm * Yc))
    Yhat = Xc @ B
    M = Yhat.T @ (Wm * Yhat)                       # 50x50
    ev, V = np.linalg.eigh(M); Pk = V[:, -k:]
    Brr = B @ Pk @ Pk.T
    f = my + (xin - mx) @ Brr; return f - f.mean()

# ---------- (B) optimal SV cleaning of cross-covariance (index removed) ----------
def crosscov_clean(d, hl=1000, keep_frac=0.3):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = design(d, hl); n, p = Xc.shape
    # remove common mode: regress predictors & targets on ALGO col 0
    a0 = Xc[:, [0]]; den = (w * a0[:, 0] ** 2).sum() + 1e-12
    bX = (Xc.T @ (w[:, None] * a0)).ravel() / den; Xr = Xc - np.outer(a0[:, 0], bX)
    bY = (Yc.T @ (w[:, None] * a0)).ravel() / den; Yr = Yc - np.outer(a0[:, 0], bY)
    Wm = w[:, None]; sw = w.sum()
    Ccross = (Xr.T @ (Wm * Yr)) / sw               # p x 50
    Clag = (Xr.T @ (Wm * Xr)) / sw
    U, sv, Vt = np.linalg.svd(Ccross, full_matrices=False)
    q = 50.0 / n; thr = np.median(sv) * (1 + np.sqrt(q))   # crude MP-style edge for singular values
    svc = np.where(sv > thr, sv, 0.0)              # kill sub-edge (noise) singular values
    Cc = (U * svc) @ Vt
    B = np.linalg.solve(Clag + 0.1 * np.trace(Clag) / p * np.eye(p), Cc)
    xin_r = xin - a0[-1, 0] * 0  # xin already includes index; use residual predictor
    f = my + (xin - mx) @ B; return f - f.mean()

# ---------- (C) single-index Ledoit-Wolf target ----------
def si_lw_ridge(d, hl=1000, delta=0.3):
    if d < 95: return None
    Xc, Yc, w, mx, my, xin = design(d, hl); n, p = Xc.shape
    Wm = w[:, None]; sw = w.sum()
    S_ = (Xc.T @ (Wm * Xc)) / sw
    # single-index target: beta to col0, F = var_m*bb' + diag(resid var)
    a0 = Xc[:, 0]; vm = (w * a0 ** 2).sum() / sw
    b = (Xc.T @ (w * a0)) / sw / (vm + 1e-12)
    F = vm * np.outer(b, b); resid = np.diag(S_) - vm * b ** 2
    F[np.diag_indices(p)] = np.maximum(resid, 1e-8) + vm * b ** 2
    Sh = delta * F + (1 - delta) * S_
    B = np.linalg.solve(Sh + 1e-8 * np.eye(p), (Xc.T @ (Wm * Yc)) / sw)
    f = my + (xin - mx) @ B; return f - f.mean()

# ---------- (D) Lo-MacKinlay accumulated reversion ----------
def lomackinlay(d, w=8):
    if d < w + 2: return None
    acc = np.zeros(50)
    for h in range(1, w + 1):
        rr = RET[1:, d - h]; acc += (rr - rr.mean())
    return -acc / (acc.std() + 1e-12)

print(f"IC tests on {S}-{E} (baseline ridge+blend 0.0791):\n")
res = {}
for name, fn, st in [
    ("(A) RRR-ridge MZ k=3", lambda d: rrr_mz(d, k=3), 2),
    ("(A) RRR-ridge MZ k=5", lambda d: rrr_mz(d, k=5), 2),
    ("(A) RRR-ridge MZ k=8", lambda d: rrr_mz(d, k=8), 2),
    ("(B) cross-cov SV clean", crosscov_clean, 2),
    ("(C) single-index LW ridge", si_lw_ridge, 2),
    ("(D) Lo-MacKinlay acc rev", lomackinlay, 1),
]:
    ic, t = ic_stats(fn, st); res[name] = ic
    print(f"  {name:<30} IC={ic:7.4f}  t={t:5.2f}" + ("  <-- BEATS 0.0791" if ic > 0.0791 else ""))

# blend best RRR with reversion
def rrr_blend(d, lam=0.2):
    r = rrr_mz(d, k=5); z = lomackinlay(d, 5)
    if r is None or z is None: return None
    return (1 - lam) * r / (r.std() + 1e-12) + lam * z
ic, t = ic_stats(rrr_blend, 1); res["(A+D) RRR + LoMac blend"] = ic
print(f"  {'(A+D) RRR + LoMac blend':<30} IC={ic:7.4f}  t={t:5.2f}" + ("  <-- BEATS 0.0791" if ic > 0.0791 else ""))

# ---------- (E) SIZING test on SCORE (does Sigma^-1 alpha beat sign?) ----------
print("\n(E) SIZING test — effect on SCORE (500-750), forecast=ridge hl1000 + 0.2 revz5:")
def ridge_fc(d, hl=1000, a=0.3):
    Xc, Yc, w, mx, my, xin = design(d, hl); p = Xc.shape[1]; Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + a * np.eye(p), Xc.T @ (Wm * Yc))
    f = my + (xin - mx) @ B; return f - f.mean()

def backtest_size(mode, Sd=500, Ed=750):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            f = ridge_fc(t); wz = f / (f.std() + 1e-12)
            rr = lp[1:, t - 1] - lp[1:, t - 6]; rr = rr - rr.mean(); z = -rr / (rr.std() + 1e-12)
            wz = 0.8 * wz + 0.2 * z
            if mode == "sign":
                raw = np.sign(wz)
            elif mode == "invvol":
                vol = RET[1:, t - 21:t - 1].std(1) + 1e-9
                raw = wz / vol; raw = raw / (np.abs(raw).max() + 1e-12)
            elif mode == "siginv_alpha":
                r = RET[:, max(0, t - 121):t - 1]; rA = r[0] - r[0].mean(); vm = rA @ rA / len(rA)
                bet = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / (rA @ rA + 1e-12)
                D = RET[1:, t - 61:t - 1].var(1) + 1e-9
                alpha = wz * np.sqrt(D)                    # alpha = IC*sigma*z (IC const drops out)
                # Sigma^-1 alpha via Sherman-Morrison, Sigma = vm*bb' + diag(D)
                Dinv_a = alpha / D; Dinv_b = bet / D
                w_ = Dinv_a - Dinv_b * (vm * (bet @ Dinv_a)) / (1 + vm * (bet @ Dinv_b))
                raw = w_ / (np.abs(w_).max() + 1e-12)
            pos[1:] = raw * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = np.log(prc[0, :t]); mv = lpA[30:] - lpA[:-30]
            zz = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            pos[0] = float(np.clip(-np.clip(zz, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu, 0
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1), sr

for mode in ("sign", "invvol", "siginv_alpha"):
    sc, sr = backtest_size(mode)
    print(f"  {mode:<14} Score={sc:7.1f}  Sharpe={sr:5.2f}  (gross deployed varies)")

print("\nSUMMARY IC:")
for k, v in sorted(res.items(), key=lambda x: -(x[1] if x[1] == x[1] else -9)):
    print(f"  {v:7.4f}  {k}")
