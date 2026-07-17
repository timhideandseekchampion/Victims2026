"""Reverse-engineer the Algothon 2026 data-generating process to the parameter level.

Hypothesis (from all prior research): stationary VAR(1) on log-returns,
    r_t = A @ r_{t-1} + eps_t,   eps_t ~ N(0, Sigma),   drifts = 0,
with Sigma = one dominant market factor + idiosyncratic, and ALGO a deterministic
(weighted) index of the 50 constituents. This script estimates A and Sigma, then
interrogates their STRUCTURE (rank/sparsity/symmetry) because that structure dictates
the optimal predictor. Everything is reported with both-halves splits so we separate
'true parameter' from 'sample noise'.
"""
import numpy as np, pandas as pd
np.set_printoptions(suppress=True, linewidth=140)

P = pd.read_csv("prices.txt", sep=r"\s+").values.T          # (51, 500) incl ALGO at row 0
names = list(pd.read_csv("prices.txt", sep=r"\s+").columns)
lp = np.log(P)
ret = lp[:, 1:] - lp[:, :-1]                                # (51, 499)
A_ret = ret[1:]                                             # 50 tradeable assets
ALGO = ret[0]
nA, T = A_ret.shape
print(f"{P.shape[1]} days, {nA} tradeable assets + ALGO.  return matrix {A_ret.shape}\n")

# ================= 0. DRIFTS / MEANS =================
print("="*70, "\n[0] DRIFTS  (H0: per-asset mean daily return = 0)")
mu = A_ret.mean(1); se = A_ret.std(1)/np.sqrt(T); tt = mu/se
print(f"  |t|>2 assets: {(np.abs(tt)>2).sum()}/50   max|t| {np.abs(tt).max():.2f}   mean t {tt.mean():+.3f}")
print(f"  ALGO mean {ALGO.mean()*252*100:+.2f}%/yr  t={ALGO.mean()/(ALGO.std()/np.sqrt(T)):+.2f}")

# ================= 1. VAR(1) FIT (the transition matrix A) =================
print("="*70, "\n[1] VAR(1) TRANSITION MATRIX A   (r_t = A r_{t-1}, 50x50, OLS)")
X = A_ret[:, :-1].T                                         # (498,50) predictors r_{t-1}
Y = A_ret[:, 1:].T                                          # (498,50) targets   r_t
Xc = X - X.mean(0); Yc = Y - Y.mean(0)
A = np.linalg.lstsq(Xc, Yc, rcond=None)[0].T               # (50,50): row i = loadings predicting asset i
resid = Yc - Xc @ A.T
R2 = 1 - resid.var(0).sum()/Yc.var(0).sum()
print(f"  in-sample R^2 (pooled): {R2:.4f}   (per-day cross-sec IC ~ sqrt-ish of this)")
ev = np.linalg.eigvals(A); sr = np.abs(ev).max()
print(f"  spectral radius max|eig(A)| = {sr:.3f}  (<1 => stationary: {'YES' if sr<1 else 'NO'})")
print(f"  ||diag(A)|| vs ||offdiag(A)||: {np.linalg.norm(np.diag(A)):.3f} vs {np.linalg.norm(A-np.diag(np.diag(A))):.3f}  "
      f"(off/diag ratio {np.linalg.norm(A-np.diag(np.diag(A)))/ (np.linalg.norm(np.diag(A))+1e-9):.1f})")
mean_diag = np.diag(A).mean()
print(f"  mean own-lag coef (diagonal): {mean_diag:+.4f}   (own-return autocorrelation term)")

# ================= 2. A: SYMMETRY (directed vs undirected lead-lag) =================
print("="*70, "\n[2] IS THE LEAD-LAG DIRECTED?  (A antisymmetric => directed network)")
offi, offj = np.where(~np.eye(nA, dtype=bool))
aij = A[offi, offj]; aji = A[offj, offi]
print(f"  corr( A_ij , A_ji ) over off-diagonal = {np.corrcoef(aij, aji)[0,1]:+.3f}   "
      f"(-1 pure antisymmetric/directed, +1 symmetric, 0 none)")
S = 0.5*(A+A.T); K = 0.5*(A-A.T)
print(f"  symmetric-part energy {np.linalg.norm(S)**2/np.linalg.norm(A)**2:.2%}   "
      f"antisymmetric-part energy {np.linalg.norm(K)**2/np.linalg.norm(A)**2:.2%}")

# ================= 3. A: RANK  (the big one for the optimal estimator) =================
print("="*70, "\n[3] RANK OF A   (low rank => reduced-rank regression beats plain ridge)")
U, s, Vt = np.linalg.svd(A)
print(f"  top-15 singular values: {np.round(s[:15],3)}")
cum = np.cumsum(s**2)/np.sum(s**2)
for r in (1,2,3,5,8,10,15,20):
    print(f"    rank {r:>2}: captures {cum[r-1]:.1%} of ||A||_F^2")

# OOS reduced-rank test: fit A on days 1..250, TRUNCATE to rank r, predict 251..500, measure IC
def crosssec_ic(Bmat, Xte, Yte):
    pred = Xte @ Bmat.T
    ics = [np.corrcoef(pred[k], Yte[k])[0,1] for k in range(len(Yte)) if Yte[k].std()>0]
    return np.mean(ics)
half = T//2
Xtr, Ytr = Xc[:half], Yc[:half]; Xte, Yte = Xc[half:], Yc[half:]
# ridge-fit A on train, then rank-truncate via SVD, evaluate OOS
lamb = 0.1*np.trace(Xtr.T@Xtr)/nA
Atr = np.linalg.solve(Xtr.T@Xtr + lamb*np.eye(nA), Xtr.T@Ytr).T
Ur,sr_,Vtr = np.linalg.svd(Atr)
print("\n  OOS (fit days 1-250 -> predict 251-500) cross-sectional IC by rank truncation:")
best=(0,-9)
for r in (1,2,3,4,5,6,8,10,15,25,50):
    Ar = (Ur[:,:r]*sr_[:r]) @ Vtr[:r]
    ic = crosssec_ic(Ar, Xte, Yte)
    if ic>best[1]: best=(r,ic)
    print(f"    rank {r:>2}:  OOS IC {ic:+.4f}")
print(f"  => OOS-optimal rank ~ {best[0]} (IC {best[1]:+.4f}); full-rank ridge IC {crosssec_ic(Atr,Xte,Yte):+.4f}")

# ================= 4. A: SPARSITY =================
print("="*70, "\n[4] SPARSITY OF A   (are most cross-terms exactly ~0?)")
# standard error per coef from OLS: se_ij ~ sigma_i / sqrt(N * var(x_j))
sig = resid.std(0)                                          # (50,) residual sd per target
sx = Xc.std(0)                                              # (50,) predictor sd
SE = sig[:,None] / (sx[None,:]*np.sqrt(len(Xc)))
tA = A/SE
off_t = tA[offi,offj]
for thr in (2,3,4):
    print(f"  |t|>{thr}: {(np.abs(off_t)>thr).sum():4d}/{len(off_t)} off-diagonal entries "
          f"({(np.abs(off_t)>thr).mean():.1%})")
print(f"  median |t| off-diagonal: {np.median(np.abs(off_t)):.2f}")

# ================= 5. INNOVATION COVARIANCE Sigma  (factor structure) =================
print("="*70, "\n[5] INNOVATION COVARIANCE Sigma = cov(eps)   (one-factor?)")
Sig = np.cov(resid.T)
evS = np.linalg.eigvalsh(Sig)[::-1]
print(f"  top-8 eigenvalues: {np.round(evS[:8]/evS.sum()*100,2)} (% of total var)")
print(f"  factor-1 share {evS[0]/evS.sum():.1%}   ratio eig1/eig2 = {evS[0]/evS[1]:.1f}")
w1 = np.linalg.eigh(Sig)[1][:,-1]; w1 = w1/np.sign(w1.mean())
print(f"  leading eigenvector (market factor) loadings: mean {w1.mean():.3f} sd {w1.std():.3f} "
      f"min {w1.min():.3f} max {w1.max():.3f}  (flat => equal-weight market factor)")

# ================= 6. ALGO CONSTRUCTION =================
print("="*70, "\n[6] HOW IS ALGO BUILT?   (regress ALGO ret on 50 constituent rets, same day)")
G = np.linalg.lstsq(np.c_[np.ones(T), A_ret.T], ALGO, rcond=None)[0]
w = G[1:]; fit = np.c_[np.ones(T),A_ret.T]@G
r2A = 1-((ALGO-fit)**2).sum()/((ALGO-ALGO.mean())**2).sum()
print(f"  R^2 of ALGO ~ basket: {r2A:.5f}  (==1 => ALGO is a DETERMINISTIC index of the 50)")
print(f"  intercept {G[0]*1e5:+.3f}e-5   weights: mean {w.mean():.4f} sd {w.std():.4f} "
      f"min {w.min():.4f} max {w.max():.4f}   (1/50={1/50:.4f})")
# is it equal-weight, or price-driven (equal-DOLLAR basket => weights track price share)?
print(f"  weight dispersion sd/mean = {w.std()/w.mean():.3f}  (0 => exactly equal-weight)")
print(f"  corr(weights, price0 share) = {np.corrcoef(w, P[1:,0]/P[1:,0].sum())[0,1]:+.3f}   "
      f"(+high => fixed-share/price-weighted index)")

# ================= 7. INNOVATIONS: NORMAL? HETEROSKEDASTIC? =================
print("="*70, "\n[7] INNOVATION DISTRIBUTION eps")
from scipy import stats
flat = (resid/resid.std(0)).ravel()
print(f"  pooled standardized eps: skew {stats.skew(flat):+.3f}  excess-kurt {stats.kurtosis(flat):+.3f}  "
      f"(0,0 => Gaussian)")
jb = stats.jarque_bera(flat)
print(f"  Jarque-Bera p={jb.pvalue:.3g}  (large => can't reject Normal)")
# ARCH / vol clustering: autocorr of squared residuals
sq = (resid**2)
ac1 = np.mean([np.corrcoef(sq[:-1,k], sq[1:,k])[0,1] for k in range(nA)])
print(f"  mean lag-1 autocorr of squared eps = {ac1:+.4f}  (>0 => GARCH/vol-clustering; ~0 => homoskedastic)")
r_ac1 = np.mean([np.corrcoef(resid[:-1,k], resid[1:,k])[0,1] for k in range(nA)])
print(f"  mean lag-1 autocorr of eps (should be ~0 if VAR(1) is complete) = {r_ac1:+.4f}")

# ================= 8. DOES VAR(2) ADD ANYTHING? (order selection) =================
print("="*70, "\n[8] MODEL ORDER   (does lag-2 improve OOS? VAR(1) vs VAR(2))")
def oos_ic_lags(nlag):
    Xl = np.hstack([A_ret[:, (2-nlag)+k:T-nlag+k].T for k in range(nlag)]) if False else None
    # build design with nlag lags predicting r_t
    rows=[]; tgt=[]
    for tt_ in range(nlag, T):
        rows.append(np.concatenate([A_ret[:,tt_-l-1] for l in range(nlag)])); tgt.append(A_ret[:,tt_])
    Xl=np.array(rows); Yl=np.array(tgt)
    h=len(Xl)//2
    lam=0.1*np.trace(Xl[:h].T@Xl[:h])/Xl.shape[1]
    B=np.linalg.solve(Xl[:h].T@Xl[:h]+lam*np.eye(Xl.shape[1]), Xl[:h].T@Yl[:h]).T
    return crosssec_ic(B, Xl[h:], Yl[h:])
print(f"  OOS IC  VAR(1): {oos_ic_lags(1):+.4f}    VAR(2): {oos_ic_lags(2):+.4f}    VAR(3): {oos_ic_lags(3):+.4f}")
print("="*70)
