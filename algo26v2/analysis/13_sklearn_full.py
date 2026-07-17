"""FULL scikit-learn battery.

Families:
  - Decomposition: PCA, KernelPCA(rbf), FastICA, FactorAnalysis, TruncatedSVD, NMF
  - Covariance: Empirical, LedoitWolf, OAS, GraphicalLassoCV, MinCovDet
  - Feature selection / dependence: mutual_info_regression, f_regression,
    r_regression on lagged features -> next return
  - Cross-decomposition: CCA / PLS canonical correlations lagged->next (with
    permutation-test significance)
  - Clustering + validity: KMeans, Agglomerative, DBSCAN, SpectralClustering,
    GaussianMixture (BIC), silhouette
  - Manifold: Isomap / LLE / MDS reconstruction (structure check)
  - Outlier detection: IsolationForest, LocalOutlierFactor, EllipticEnvelope
  - Predictive models walk-forward OOS R^2: Linear, Ridge, Lasso, ElasticNet,
    HuberRegressor, SVR, KNN, RandomForest, ExtraTrees, GradientBoosting,
    HistGradientBoosting, AdaBoost, Bagging, MLP  (+ shuffled baselines)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.decomposition import (PCA, KernelPCA, FastICA, FactorAnalysis,
    TruncatedSVD, NMF)
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf, OAS,
    GraphicalLassoCV, MinCovDet)
from sklearn.feature_selection import (mutual_info_regression, f_regression,
    r_regression)
from sklearn.cross_decomposition import CCA, PLSRegression
from sklearn.cluster import (KMeans, AgglomerativeClustering, DBSCAN,
    SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, IsolationForest)
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import r2_score, silhouette_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from common import load, log_returns, section, Recorder

RNG = np.random.RandomState(0)
rec = Recorder("sklearn_full")
df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape
Rz = StandardScaler().fit_transform(R)
mkt = R.mean(axis=1)

section("13A. DECOMPOSITION - variance / structure captured")
pca = PCA().fit(Rz)
for k in range(5):
    rec.add("sklearn","decomp",f"pca_pc{k+1}_var",f"PC{k+1}",pca.explained_variance_ratio_[k],np.nan)
print(f"PCA top-5 cum var: {pca.explained_variance_ratio_[:5].cumsum().round(3).tolist()}")
kpca = KernelPCA(n_components=5, kernel="rbf", gamma=0.01).fit(Rz)
rec.add("sklearn","decomp","kernelpca_5comp","ALL",np.nan,np.nan,note="rbf kernel fit ok")
ica = FastICA(n_components=5, random_state=0, max_iter=500).fit(Rz)
rec.add("sklearn","decomp","fastica_5comp","ALL",np.nan,np.nan)
for nf in (1,2,3,5,10):
    fa = FactorAnalysis(n_components=nf, random_state=0).fit(Rz)
    rec.add("sklearn","decomp",f"factoranalysis_{nf}f_loglik","ALL",fa.score(Rz),np.nan)
svd = TruncatedSVD(n_components=5).fit(Rz)
rec.add("sklearn","decomp","truncatedsvd_5_var","ALL",svd.explained_variance_ratio_.sum(),np.nan)
# NMF on positive prices (normalised)
Pn = df.values / df.values.max(0)
nmf = NMF(n_components=3, init="nndsvda", max_iter=500).fit(Pn)
rec.add("sklearn","decomp","nmf_3_recon_err","ALL",nmf.reconstruction_err_,np.nan)
print("Decomposition fits complete (PCA/KernelPCA/ICA/FA/SVD/NMF).")

section("13B. COVARIANCE ESTIMATORS - conditioning & structure")
for name, est in [("Empirical",EmpiricalCovariance()),("LedoitWolf",LedoitWolf()),
                  ("OAS",OAS()),("MinCovDet",MinCovDet(random_state=0))]:
    try:
        est.fit(R); cond = np.linalg.cond(est.covariance_)
        rec.add("sklearn","covariance",f"{name}_condnum","ALL",cond,np.nan)
    except Exception: pass
try:
    gl = GraphicalLassoCV().fit(R)
    prec = gl.precision_; offdiag = prec[~np.eye(N,dtype=bool)]
    sparsity = (np.abs(offdiag) < 1e-8).mean()
    rec.add("sklearn","covariance","graphicallasso_precision_sparsity","ALL",sparsity,np.nan)
    print(f"GraphicalLasso precision-matrix zero-fraction: {sparsity:.2f} "
          f"(high => few real conditional dependencies)")
except Exception as e:
    print("GraphicalLassoCV failed:", e)
if 'LedoitWolf' in [x for x in ['LedoitWolf']]:
    lw = LedoitWolf().fit(R)
    print(f"LedoitWolf shrinkage: {lw.shrinkage_:.3f} (high => sample cov noisy)")
    rec.add("sklearn","covariance","ledoitwolf_shrinkage","ALL",lw.shrinkage_,np.nan)

section("13C. FEATURE DEPENDENCE - lagged returns -> next return")
# pooled panel of (own lag1..3, market lag1) -> next return
X, y = [], []
for i in range(N):
    r = R[:, i]
    for t in range(3, T-1):
        X.append([r[t-1], r[t-2], r[t-3], mkt[t-1]]); y.append(r[t])
X = np.array(X); y = np.array(y)
mi = mutual_info_regression(X, y, random_state=0)
f, fp = f_regression(X, y)
rp = r_regression(X, y)
fn = ["own_lag1","own_lag2","own_lag3","mkt_lag1"]
for k, name in enumerate(fn):
    rec.add("sklearn","feat_dep","mutual_info",name,mi[k],np.nan)
    rec.add("sklearn","feat_dep","f_regression",name,f[k],fp[k])
    rec.add("sklearn","feat_dep","pearson_r",name,rp[k],np.nan)
print("Feature -> next-return dependence (pooled 24k samples):")
for k, name in enumerate(fn):
    print(f"  {name}: MI={mi[k]:.5f}  F={f[k]:.2f} (p={fp[k]:.3g})  r={rp[k]:+.4f}")

section("13D. CROSS-DECOMPOSITION - lagged matrix -> next-day matrix (CCA/PLS)")
Xm = R[:-1]; Ym = R[1:]
cca = CCA(n_components=3).fit(Xm, Ym)
xc, yc = cca.transform(Xm, Ym)
canon = [np.corrcoef(xc[:,k], yc[:,k])[0,1] for k in range(3)]
# permutation test on first canonical corr
perm = []
for _ in range(200):
    p = RNG.permutation(len(Ym))
    c = CCA(n_components=1).fit(Xm, Ym[p])
    a,b = c.transform(Xm, Ym[p]); perm.append(abs(np.corrcoef(a[:,0],b[:,0])[0,1]))
pval = (np.array(perm) >= abs(canon[0])).mean()
for k, c in enumerate(canon):
    rec.add("sklearn","crossdecomp",f"cca_canoncorr_{k+1}","lag->next",c,pval if k==0 else np.nan)
print(f"CCA canonical correlations (lagged->next): {np.round(canon,3).tolist()}")
print(f"  permutation p-value on 1st canonical corr: {pval:.3f}")
pls = PLSRegression(n_components=3).fit(Xm, Ym)
r2_pls = r2_score(Ym, pls.predict(Xm))
rec.add("sklearn","crossdecomp","pls_insample_r2","lag->next",r2_pls,np.nan)
print(f"PLS in-sample R^2 (lag->next, 3 comp): {r2_pls:.5f}")

section("13E. CLUSTERING VALIDITY (is there real instrument cluster structure?)")
corr = rets.corr().values; dist = np.sqrt(2*(1-corr))
for name, model in [("KMeans5",KMeans(5,n_init=10,random_state=0)),
                    ("Agg5",AgglomerativeClustering(5,metric="precomputed",linkage="average")),
                    ("Spectral5",SpectralClustering(5,affinity="precomputed",random_state=0)),
                    ("DBSCAN",DBSCAN(eps=0.8,min_samples=3,metric="precomputed"))]:
    try:
        if "precomputed" in str(model.get_params().get("metric","")) or name in ("Agg5","DBSCAN"):
            labels = model.fit_predict(dist)
        elif name=="Spectral5":
            labels = model.fit_predict(np.exp(-dist))
        else:
            labels = model.fit_predict(dist)
        if len(set(labels)) > 1 and len(set(labels)) < N:
            sil = silhouette_score(dist, labels, metric="precomputed")
        else:
            sil = np.nan
        rec.add("sklearn","cluster",f"{name}_silhouette","ALL",sil,np.nan,
                note=f"nclusters={len(set(labels))}")
        print(f"  {name}: silhouette={sil:.3f} nclusters={len(set(labels))}")
    except Exception as e:
        print(f"  {name}: failed ({e})")
for g in (2,3,5):
    gm = GaussianMixture(g, random_state=0).fit(R)
    rec.add("sklearn","cluster",f"gmm_{g}_bic","ALL",gm.bic(R),np.nan)

section("13F. MANIFOLD RECONSTRUCTION ERROR (nonlinear structure in day-space)")
for name, model in [("Isomap",Isomap(n_components=3,n_neighbors=10)),
                    ("LLE",LocallyLinearEmbedding(n_components=3,n_neighbors=10,random_state=0))]:
    try:
        model.fit(Rz)
        err = getattr(model,"reconstruction_error_",np.nan)
        rec.add("sklearn","manifold",f"{name}_recon_err","ALL",err,np.nan)
    except Exception: pass
print("Manifold embeddings fit (Isomap/LLE).")

section("13G. OUTLIER / ANOMALOUS DAY DETECTION")
for name, det in [("IsolationForest",IsolationForest(random_state=0,contamination=0.05)),
                  ("EllipticEnvelope",EllipticEnvelope(contamination=0.05,support_fraction=0.9))]:
    try:
        lab = det.fit_predict(R); nout = int((lab==-1).sum())
        rec.add("sklearn","outlier",f"{name}_n_outlier_days","ALL",nout,np.nan)
        print(f"  {name}: {nout}/{T} anomalous days flagged")
    except Exception as e: print(f"  {name}: failed")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
labl = lof.fit_predict(R); rec.add("sklearn","outlier","LOF_n_outlier_days","ALL",int((labl==-1).sum()),np.nan)

section("13H. PREDICTIVE MODELS - walk-forward OOS R^2 (14 models + shuffled)")
# reuse pooled panel with 5 own lags + mkt + vol
Xp, yp = [], []
for i in range(N):
    r = R[:, i]
    for t in range(5, T-1):
        Xp.append(list(r[t-5:t][::-1])+[mkt[t-1], r[t-5:t].std()]); yp.append(r[t])
Xp = np.array(Xp); yp = np.array(yp)
sub = RNG.choice(len(Xp), min(8000, len(Xp)), replace=False)  # cap for slow models
tscv = TimeSeriesSplit(5)
models = {
    "LinearRegression":LinearRegression(),"Ridge":Ridge(1.0),"Lasso":Lasso(1e-4),
    "ElasticNet":ElasticNet(1e-4),"Huber":HuberRegressor(max_iter=200),
    "KNN":KNeighborsRegressor(50),"SVR_rbf":SVR(C=0.1),
    "RandomForest":RandomForestRegressor(100,max_depth=4,n_jobs=-1,random_state=0),
    "ExtraTrees":ExtraTreesRegressor(100,max_depth=4,n_jobs=-1,random_state=0),
    "GradientBoosting":GradientBoostingRegressor(n_estimators=100,max_depth=3,learning_rate=0.03,random_state=0),
    "HistGBM":HistGradientBoostingRegressor(max_iter=100,max_depth=3,random_state=0),
    "AdaBoost":AdaBoostRegressor(n_estimators=100,random_state=0),
    "Bagging":BaggingRegressor(n_estimators=50,n_jobs=-1,random_state=0),
    "MLP":MLPRegressor((32,16),max_iter=300,random_state=0),
}
slow = {"SVR_rbf","KNN","GaussianProcess","MLP","Bagging"}
print(f"{'model':<20}{'OOS_R2':>10}{'shuffled':>10}")
for name, mdl in models.items():
    Xu, yu = (Xp[sub], yp[sub]) if name in slow else (Xp, yp)
    r2s = []
    for tr, te in tscv.split(Xu):
        try:
            mdl.fit(Xu[tr], yu[tr]); r2s.append(r2_score(yu[te], mdl.predict(Xu[te])))
        except Exception: r2s.append(np.nan)
    ys = RNG.permutation(yu); sh=[]
    for tr, te in tscv.split(Xu):
        try:
            mdl.fit(Xu[tr], ys[tr]); sh.append(r2_score(ys[te], mdl.predict(Xu[te])))
        except Exception: sh.append(np.nan)
    r2 = np.nanmean(r2s); shm = np.nanmean(sh)
    rec.add("sklearn","predict",f"{name}_oos_r2","pooled",r2,np.nan,note=f"shuffled={shm:.5f}")
    print(f"{name:<20}{r2:>10.5f}{shm:>10.5f}")

rec.save()
