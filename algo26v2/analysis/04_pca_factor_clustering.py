"""Module 4: Factor structure (PCA), clustering, and the KEY signal test:
cross-sectional / residual mean-reversion.

Libraries: scikit-learn (PCA, FactorAnalysis, KMeans, AgglomerativeClustering,
StandardScaler), scipy, statsmodels.

Logic: module 3 showed ALGO correlates with everything -> a common factor.
If we regress each name's return on the market factor, the RESIDUAL is the
idiosyncratic move. If residuals mean-revert day-to-day, we have a market-
neutral stat-arb signal (short winners / long losers of the residual).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from common import load, log_returns, section, stars

df, tickers = load()
rets = log_returns(df)
N = len(tickers)
R = rets.values                       # (T-1) x N
Rz = StandardScaler().fit_transform(R)

section("4A. PCA ON STANDARDISED RETURNS (factor structure)")
pca = PCA().fit(Rz)
ev = pca.explained_variance_ratio_
print("Variance explained by top components:")
for k in range(8):
    print(f"  PC{k+1}: {ev[k]*100:5.2f}%   cumulative: {ev[:k+1].sum()*100:5.2f}%")
print(f"\nPC1 explains {ev[0]*100:.1f}% -> {'STRONG' if ev[0]>0.25 else 'moderate' if ev[0]>0.15 else 'weak'} common (market) factor")
# correlation of PC1 score with ALGO return
pc1_scores = pca.transform(Rz)[:, 0]
algo_r = R[:, 0]
corr_algo = np.corrcoef(pc1_scores, algo_r)[0, 1]
print(f"corr(PC1 score, ALGO return) = {corr_algo:+.3f}  "
      f"=> ALGO {'IS essentially the market factor' if abs(corr_algo)>0.7 else 'partially proxies the factor'}")
# PC1 loadings
load1 = pd.Series(pca.components_[0], index=tickers)
print(f"\nPC1 loadings: all same sign? {'YES (level factor)' if (load1>0).all() or (load1<0).all() else 'no'}")
print(f"  ALGO loading: {load1['ALGO']:+.3f} (rank {load1.abs().rank(ascending=False)['ALGO']:.0f}/{N})")

section("4B. FACTOR ANALYSIS (sklearn) - how many latent factors")
for nf in (1, 2, 3, 5):
    fa = FactorAnalysis(n_components=nf, random_state=0).fit(Rz)
    print(f"  {nf} factors: avg log-likelihood/sample = {fa.score(Rz):.4f}")

section("4C. CLUSTERING INSTRUMENTS (KMeans & hierarchical on corr distance)")
corr = rets.corr().values
dist = np.sqrt(2 * (1 - corr))
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(dist)
agg = AgglomerativeClustering(n_clusters=5, metric="precomputed",
                              linkage="average").fit(dist)
cl = pd.DataFrame({"tkr": tickers, "kmeans": km.labels_, "hier": agg.labels_})
for c in sorted(set(km.labels_)):
    print(f"  KMeans cluster {c}: {' '.join(cl[cl.kmeans==c].tkr.tolist())}")

section("4D. *** KEY TEST: RESIDUAL MEAN-REVERSION (market-neutral signal) ***")
# Market factor return = cross-sectional mean of returns each day (equal-weight index)
mkt = R.mean(axis=1)
print("Regress each name on the equal-weight market return; test residual AC1.\n")
betas, ac1s, lbps = [], [], []
resid_mat = np.zeros_like(R)
for i in range(N):
    y = R[:, i]
    b1, b0 = np.polyfit(mkt, y, 1)
    resid = y - (b1 * mkt + b0)
    resid_mat[:, i] = resid
    betas.append(b1)
    a1 = np.corrcoef(resid[1:], resid[:-1])[0, 1]
    lb = acorr_ljungbox(resid, lags=[1], return_df=True)["lb_pvalue"].iloc[0]
    ac1s.append(a1); lbps.append(lb)
rr = pd.DataFrame({"tkr": tickers, "beta_mkt": betas,
                   "resid_AC1": ac1s, "LB1_p": lbps})
rr["sig"] = rr.LB1_p.apply(stars)
print(rr.round(4).to_string(index=False))
n_neg = (rr.resid_AC1 < 0).sum()
n_sig = (rr.LB1_p < 0.05).sum()
print(f"\nResidual AC1: mean {rr.resid_AC1.mean():+.4f}")
print(f"  negative (mean-reverting) in {n_neg}/{N} names; significant (LB p<.05) in {n_sig}/{N}")

section("4E. CROSS-SECTIONAL REVERSAL PORTFOLIO TEST")
# Each day rank residual returns; go short today's winners / long losers; measure next-day pnl
resid = resid_mat
demeaned = resid - resid.mean(axis=1, keepdims=True)
# signal = -yesterday residual (bet on reversal); weight normalised
sig = -demeaned[:-1]
fwd = resid[1:]                         # next-day residual return
sig = sig / np.abs(sig).sum(axis=1, keepdims=True)
pnl = (sig * fwd).sum(axis=1)
sr = np.sqrt(250) * pnl.mean() / pnl.std()
t_stat, t_p = stats.ttest_1samp(pnl, 0)
print("Market-neutral 1-day residual-reversal strategy (in return space):")
print(f"  mean daily pnl: {pnl.mean():.6e}")
print(f"  annualised Sharpe: {sr:.3f}")
print(f"  t-stat vs 0: {t_stat:.2f} (p={t_p:.4g}) {stars(t_p)}")
print(f"  hit rate (days pnl>0): {(pnl>0).mean()*100:.1f}%")

# also raw (total-return) cross-sectional reversal
dm_tot = R - R.mean(axis=1, keepdims=True)
sig2 = -dm_tot[:-1]; sig2 = sig2/np.abs(sig2).sum(axis=1, keepdims=True)
pnl2 = (sig2 * R[1:]).sum(axis=1)
sr2 = np.sqrt(250)*pnl2.mean()/pnl2.std()
t2, p2 = stats.ttest_1samp(pnl2, 0)
print(f"\nTotal-return cross-sectional reversal: Sharpe {sr2:.3f}, t={t2:.2f} (p={p2:.4g}) {stars(p2)}")

section("4F. VERDICT")
print(f"PC1 variance: {ev[0]*100:.1f}%; corr(PC1,ALGO)={corr_algo:+.2f}")
print(f"Residual reversal Sharpe {sr:.2f} (t={t_stat:.1f}); total-ret reversal Sharpe {sr2:.2f} (t={t2:.1f})")
print("If reversal Sharpe is high & t-stat strong, THIS is the exploitable edge.")
np.save("results/resid_matrix.npy", resid_mat)
