"""FRESH exhaustive statistical analysis of days 400-750 (no strategy assumptions).
Characterize the data: distribution, stationarity, autocorrelation, variance ratio, vol
clustering, factor structure, lead-lag predictability, cointegration, cross-sectional signals,
and within-window stability. Report a clean findings summary."""
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
import warnings; warnings.filterwarnings("ignore")

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
A0, A1 = 400, min(750, nt)                                  # window days 400-750
P = prc[:, A0:A1]                                            # (51, ~349)
lp = np.log(P); R = np.diff(lp, axis=1)                      # returns in-window (51, T-1)
algo = R[0]; STK = R[1:]; T = STK.shape[1]
print(f"=== DATA ANALYSIS: days {A0}-{A1} ({nInst} instruments, {T} return-days) ===\n")

# 1. DRIFT
mu = STK.mean(1); se = STK.std(1)/np.sqrt(T); tt = mu/se
print("[1] DRIFT (per-name mean return)")
print(f"    |t|>2 names: {(np.abs(tt)>2).sum()}/50   max|t| {np.abs(tt).max():.2f}   "
      f"ALGO drift {algo.mean()*252*100:+.1f}%/yr (t={algo.mean()/(algo.std()/np.sqrt(T)):+.2f})")

# 2. DISTRIBUTION
flat = ((STK - STK.mean(1, keepdims=True))/(STK.std(1, keepdims=True)+1e-12)).ravel()
jb = [stats.jarque_bera(STK[i])[1] for i in range(50)]
print("\n[2] DISTRIBUTION")
print(f"    pooled skew {stats.skew(flat):+.3f}  excess-kurt {stats.kurtosis(flat):+.3f}   "
      f"JB rejects normal: {sum(np.array(jb)<0.05)}/50  (fat tails? {'yes' if stats.kurtosis(flat)>0.5 else 'no'})")

# 3. STATIONARITY (ADF): prices unit-root? returns stationary?
adf_p_price = [adfuller(P[i], maxlag=5)[1] for i in range(51)]
adf_p_ret = [adfuller(R[i], maxlag=5)[1] for i in range(51)]
print("\n[3] STATIONARITY (ADF p<0.05 = stationary)")
print(f"    prices stationary: {sum(np.array(adf_p_price)<0.05)}/51   "
      f"returns stationary: {sum(np.array(adf_p_ret)<0.05)}/51")

# 4. AUTOCORRELATION (white noise?)
ac1 = np.array([np.corrcoef(STK[i, :-1], STK[i, 1:])[0, 1] for i in range(50)])
lb_sig = 0
for i in range(50):
    a = acf(STK[i], nlags=5, fft=True)[1:]
    Q = T*(T+2)*np.sum(a**2/(T-np.arange(1, 6)))
    if Q > 11.07: lb_sig += 1                               # chi2(5) 0.05 crit
print("\n[4] AUTOCORRELATION")
print(f"    mean lag-1 AC {ac1.mean():+.4f} (t {ac1.mean()*np.sqrt(T):+.2f})   "
      f"|AC1|>2/sqrt(T): {(np.abs(ac1)>2/np.sqrt(T)).sum()}/50   Ljung-Box sig: {lb_sig}/50")

# 5. VARIANCE RATIO
def vr(x, q):
    v1 = x.var(); xq = np.add.reduceat(x, np.arange(0, len(x)-len(x) % q, q)); return (xq.var()/q)/v1
vr5 = np.mean([vr(STK[i], 5) for i in range(50)]); vr10 = np.mean([vr(STK[i], 10) for i in range(50)])
print(f"\n[5] VARIANCE RATIO   VR(5) {vr5:.3f}   VR(10) {vr10:.3f}   (<1 mean-revert, >1 trend, =1 RW)")

# 6. VOL CLUSTERING (ARCH: autocorr of squared returns)
arch = np.array([np.corrcoef((STK[i]**2)[:-1], (STK[i]**2)[1:])[0, 1] for i in range(50)])
print(f"\n[6] VOL CLUSTERING   mean ARCH-AC1 {arch.mean():+.4f}   sig names: {(np.abs(arch)>2/np.sqrt(T)).sum()}/50")

# 7. FACTOR STRUCTURE
C = np.corrcoef(STK); ev = np.linalg.eigvalsh(C)[::-1]
mp_hi = (1+np.sqrt(50/T))**2                                # Marchenko-Pastur upper edge
nfac = int((ev > mp_hi).sum())
pc1 = np.linalg.eigh(np.cov(STK))[1][:, -1]
print(f"\n[7] FACTOR STRUCTURE   PC1 var-share {ev[0]/ev.sum():.1%}   #factors>MP-noise {nfac}   "
      f"corr(PC1,ALGO) {abs(np.corrcoef((STK.T@pc1), algo)[0,1]):.3f}   mean pair-corr {C[np.triu_indices(50,1)].mean():+.3f}")

# 8. LEAD-LAG PREDICTABILITY (walk-forward VAR IC within window)
def wf_ic():
    ics = []
    for d in range(60, T-1):
        X = STK[:, :d].T; Y = STK[:, 1:d+1].T; Xc = X-X.mean(0); Yc = Y-Y.mean(0)
        lam = 0.1*np.trace(Xc.T@Xc)/50
        B = np.linalg.solve(Xc.T@Xc+lam*np.eye(50), Xc.T@Yc)
        pr = (STK[:, d]-X.mean(0))@B; fwd = STK[:, d]  # careful: predict d+1 from d
        # proper: fit on [:d], predict d+1
    return None
# simpler: split-half OOS lead-lag IC
h = T//2
Xtr = STK[:, :h-1].T; Ytr = STK[:, 1:h].T; Xc = Xtr-Xtr.mean(0); Yc = Ytr-Ytr.mean(0)
lam = 0.1*np.trace(Xc.T@Xc)/50
B = np.linalg.solve(Xc.T@Xc+lam*np.eye(50), Xc.T@Yc)
Xte = STK[:, h:-1].T; Yte = STK[:, h+1:].T
pred = (Xte-Xtr.mean(0))@B
ics = [np.corrcoef(pred[k]-pred[k].mean(), Yte[k])[0, 1] for k in range(len(pred)) if Yte[k].std() > 0]
print(f"\n[8] LEAD-LAG (VAR(1) OOS, fit 1st half predict 2nd)   cross-sec IC {np.mean(ics):+.4f} (t {np.mean(ics)/(np.std(ics)/np.sqrt(len(ics))):+.1f})")

# 9. COINTEGRATION (Engle-Granger on corr-prefiltered pairs)
from statsmodels.tsa.stattools import coint
Lp = lp; cnt = 0; tested = 0
Cc = np.corrcoef(np.diff(Lp[1:], axis=1))
for i in range(50):
    for j in range(i+1, 50):
        if abs(Cc[i, j]) < 0.4: continue
        tested += 1
        try:
            if coint(Lp[i+1], Lp[j+1])[1] < 0.02: cnt += 1
        except Exception: pass
print(f"\n[9] COINTEGRATION   pairs p<0.02: {cnt} (of {tested} corr-prefiltered; ~{0.02*tested:.1f} expected by chance)")

# 10. CROSS-SECTIONAL SIGNAL IC (reversion & momentum horizons)
def xs_ic(hfn):
    ics = []
    for t in range(25, T-1):
        s = hfn(t); s = s-s.mean(); f = STK[:, t+1]
        if s.std() > 0 and f.std() > 0: ics.append(np.corrcoef(s, f)[0, 1])
    return np.mean(ics), np.mean(ics)/(np.std(ics)/np.sqrt(len(ics))+1e-12)
print("\n[10] CROSS-SECTIONAL SIGNAL IC (predict next-day return)")
for lbl, fn in [("reversion 1d", lambda t: -STK[:, t]), ("reversion 5d", lambda t: -STK[:, t-4:t+1].sum(1)),
                ("reversion 20d", lambda t: -STK[:, t-19:t+1].sum(1)), ("momentum 5d", lambda t: STK[:, t-4:t+1].sum(1))]:
    ic, tv = xs_ic(fn); print(f"     {lbl:14} IC {ic:+.4f} (t {tv:+.1f})")

# 11. WITHIN-WINDOW STABILITY (first vs second half)
print("\n[11] WITHIN-WINDOW STABILITY (400-575 vs 575-750)")
for lbl, a, b in [("1st half", 0, h), ("2nd half", h, T)]:
    s = STK[:, a:b]; ac = np.mean([np.corrcoef(s[i, :-1], s[i, 1:])[0, 1] for i in range(50)])
    print(f"     {lbl}: mean|ret| {np.abs(s).mean()*100:.2f}%  x-sec disp {s.std(0).mean():.4f}  ac1(t) {ac*np.sqrt(b-a):+.2f}")
print("\nDONE")
