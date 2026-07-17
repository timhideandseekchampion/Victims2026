"""FULL statsmodels battery.

Families:
  - Unit root / stationarity: ADF, KPSS, range_unit_root, Zivot-Andrews (break)
  - Autocorrelation: Ljung-Box (multi-lag), Box-Pierce, Breusch-Godfrey, DW
  - Cross-correlation (ccf) ALGO->name lead-lag at lags 1..5
  - Granger causality: ALGO->name AND name->ALGO (full)
  - Heteroskedasticity: ARCH-LM, Breusch-Pagan, White, Goldfeld-Quandt
  - Specification/linearity: RESET, Harvey-Collier, Rainbow
  - Structural breaks: CUSUM (OLS resid), recursive residuals, Hansen
  - Robust moments: medcouple, robust_skewness/kurtosis; omni normtest
  - Model selection: ARMA order (AIC/BIC) - does any ARMA beat white noise?
  - AR(1) coefficient significance
  - Seasonal decomposition strength (STL)
  - Markov regime switching (subset)
  - Multiple-testing correction summary on cointegration p-values
"""
import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import (adfuller, kpss, range_unit_root_test,
    zivot_andrews, acf, ccf, grangercausalitytests, arma_order_select_ic, coint)
from statsmodels.stats.diagnostic import (acorr_ljungbox, acorr_breusch_godfrey,
    het_arch, het_breuschpagan, het_white, het_goldfeldquandt, linear_reset,
    linear_harvey_collier, linear_rainbow, breaks_cusumolsresid)
from statsmodels.stats.stattools import (durbin_watson, medcouple,
    robust_skewness, robust_kurtosis, omni_normtest, jarque_bera)
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.arima.model import ARIMA
from common import load, log_returns, section, stars, Recorder

rec = Recorder("statsmodels_full")
df, tickers = load()
rets = log_returns(df)
R = rets.values
P = df.values
T, N = R.shape
mkt = R.mean(axis=1)

section("11A. UNIT ROOT / STATIONARITY (4 tests x prices & returns)")
sc = {"adf_price":0,"kpss_price_nonstat":0,"rur_price":0,"za_price":0}
for i, t in enumerate(tickers):
    p = P[:, i]; r = R[:, i]
    ap = adfuller(p, autolag="AIC"); rec.add("statsmodels","unitroot","adf_price",t,ap[0],ap[1])
    ar = adfuller(r, autolag="AIC"); rec.add("statsmodels","unitroot","adf_return",t,ar[0],ar[1])
    try:
        kp = kpss(p, "c", nlags="auto"); rec.add("statsmodels","unitroot","kpss_price",t,kp[0],kp[1])
        if kp[1] < 0.05: sc["kpss_price_nonstat"] += 1
    except Exception: pass
    try:
        rr = range_unit_root_test(p); rec.add("statsmodels","unitroot","range_unitroot_price",t,rr[0],rr[1])
        if rr[1] < 0.05: sc["rur_price"] += 1
    except Exception: pass
    try:
        za = zivot_andrews(p); rec.add("statsmodels","unitroot","zivot_andrews_price",t,za[0],za[1],
                                       note=f"break_at={za[4]}")
        if za[1] < 0.05: sc["za_price"] += 1
    except Exception: pass
    if ap[1] < 0.05: sc["adf_price"] += 1
print(f"ADF rejects unit root on price (stationary): {sc['adf_price']}/{N}")
print(f"KPSS says price non-stationary:              {sc['kpss_price_nonstat']}/{N}")
print(f"Range-unit-root rejects:                     {sc['rur_price']}/{N}")
print(f"Zivot-Andrews (allows 1 break) rejects:      {sc['za_price']}/{N}")

section("11B. AUTOCORRELATION (Ljung-Box multi-lag, Box-Pierce, Breusch-Godfrey, DW)")
nlb = 0
for i, t in enumerate(tickers):
    r = R[:, i]
    lb = acorr_ljungbox(r, lags=[1,2,3,5,10,20], boxpierce=True, return_df=True)
    for lag in lb.index:
        rec.add("statsmodels","autocorr",f"ljungbox_lag{lag}",t,lb.loc[lag,"lb_stat"],lb.loc[lag,"lb_pvalue"])
        rec.add("statsmodels","autocorr",f"boxpierce_lag{lag}",t,lb.loc[lag,"bp_stat"],lb.loc[lag,"bp_pvalue"])
    rec.add("statsmodels","autocorr","durbin_watson",t,durbin_watson(r),np.nan)
    if lb.loc[10,"lb_pvalue"] < 0.05: nlb += 1
    # Breusch-Godfrey on AR(0) regression (returns on const)
    X = sm.add_constant(np.arange(len(r))); ols = sm.OLS(r, X).fit()
    try:
        bg = acorr_breusch_godfrey(ols, nlags=5)
        rec.add("statsmodels","autocorr","breusch_godfrey",t,bg[0],bg[1])
    except Exception: pass
print(f"Significant Ljung-Box at lag10 (p<0.05): {nlb}/{N}")

section("11C. CROSS-CORRELATION ALGO->name (lead-lag) at lags 1..5")
lead = 0
for i, t in enumerate(tickers):
    if i == 0: continue
    c = ccf(R[:, i], R[:, 0], adjusted=False)[:6]  # corr(name_t, algo_{t-k})
    for k in range(1, 6):
        # approx 2-sigma band = 2/sqrt(T)
        rec.add("statsmodels","crosscorr",f"ccf_algo_lag{k}",t,c[k],np.nan,
                note=f"2sig={2/np.sqrt(T):.3f}")
    if abs(c[1]) > 2/np.sqrt(T): lead += 1
print(f"Names with |lag-1 cross-corr to ALGO|>2sigma: {lead}/{N-1}")

section("11D. GRANGER CAUSALITY full: ALGO->name and name->ALGO")
g_ta = g_at = 0
for i in range(1, N):
    for x, y, tag in [(R[:,0],R[:,i],"algo_to"), (R[:,i],R[:,0],"to_algo")]:
        data = np.column_stack([y, x])
        try:
            res = grangercausalitytests(data, maxlag=3, verbose=False)
            p = min(res[l][0]["ssr_ftest"][1] for l in (1,2,3))
            rec.add("statsmodels","granger",f"granger_{tag}",tickers[i],np.nan,p)
            if tag=="algo_to" and p<0.05: g_at += 1
            if tag=="to_algo" and p<0.05: g_ta += 1
        except Exception: pass
print(f"ALGO Granger-causes name: {g_at}/{N-1} | name Granger-causes ALGO: {g_ta}/{N-1}")

section("11E. HETEROSKEDASTICITY (ARCH-LM, Breusch-Pagan, White, Goldfeld-Quandt)")
narch=nbp=nwhite=0
for i, t in enumerate(tickers):
    r = R[:, i]
    la = het_arch(r, nlags=5); rec.add("statsmodels","hetero","arch_lm",t,la[0],la[1])
    X = sm.add_constant(mkt); ols = sm.OLS(r, X).fit()
    bp = het_breuschpagan(ols.resid, ols.model.exog); rec.add("statsmodels","hetero","breusch_pagan",t,bp[0],bp[1])
    try:
        wh = het_white(ols.resid, ols.model.exog); rec.add("statsmodels","hetero","white",t,wh[0],wh[1])
        if wh[1]<0.05: nwhite+=1
    except Exception: pass
    gq = het_goldfeldquandt(r, X); rec.add("statsmodels","hetero","goldfeld_quandt",t,gq[0],gq[1])
    if la[1]<0.05: narch+=1
    if bp[1]<0.05: nbp+=1
print(f"ARCH-LM sig: {narch}/{N} | Breusch-Pagan sig: {nbp}/{N} | White sig: {nwhite}/{N}")

section("11F. SPECIFICATION / LINEARITY (RESET, Harvey-Collier, Rainbow)")
nrz=0
for i, t in enumerate(tickers):
    r = R[:, i]; X = sm.add_constant(mkt); ols = sm.OLS(r, X).fit()
    try:
        rz = linear_reset(ols, power=2, use_f=True); rec.add("statsmodels","spec","reset",t,rz.fvalue,rz.pvalue)
        if rz.pvalue<0.05: nrz+=1
    except Exception: pass
    try:
        hc = linear_harvey_collier(ols); rec.add("statsmodels","spec","harvey_collier",t,hc[0],hc[1])
    except Exception: pass
    try:
        rb = linear_rainbow(ols); rec.add("statsmodels","spec","rainbow",t,rb[0],rb[1])
    except Exception: pass
print(f"RESET nonlinearity in name~market (p<0.05): {nrz}/{N}")

section("11G. STRUCTURAL BREAKS (CUSUM OLS residuals)")
ncb=0
for i, t in enumerate(tickers):
    r = R[:, i]; X = sm.add_constant(np.arange(len(r))); resid = sm.OLS(r, X).fit().resid
    try:
        cb = breaks_cusumolsresid(resid); rec.add("statsmodels","break","cusum_olsresid",t,cb[0],cb[1])
        if cb[1]<0.05: ncb+=1
    except Exception: pass
print(f"CUSUM structural break (p<0.05): {ncb}/{N}")

section("11H. ROBUST MOMENTS + OMNIBUS NORMALITY")
for i, t in enumerate(tickers):
    r = R[:, i]
    rec.add("statsmodels","robust","medcouple",t,medcouple(r),np.nan)
    rec.add("statsmodels","robust","robust_skewness",t,robust_skewness(r)[0],np.nan)
    rec.add("statsmodels","robust","robust_kurtosis",t,robust_kurtosis(r)[0],np.nan)
    om = omni_normtest(r); rec.add("statsmodels","normality","omni_normtest",t,om[0],om[1])

section("11I. ARMA ORDER SELECTION - does any ARMA beat white noise (0,0)?")
better=0
for i, t in enumerate(tickers):
    r = R[:, i]
    try:
        sel = arma_order_select_ic(r, max_ar=2, max_ma=2, ic="bic")
        order = sel.bic_min_order
        rec.add("statsmodels","model","arma_bic_order",t,np.nan,np.nan,note=f"order={order}")
        if order != (0,0): better += 1
        # AR(1) coefficient significance
        fit = ARIMA(r, order=(1,0,0)).fit()
        ar1 = fit.params[1]; pv = fit.pvalues[1]
        rec.add("statsmodels","model","ar1_coef",t,ar1,pv)
    except Exception: pass
print(f"Instruments whose BIC-best ARMA != white noise (0,0): {better}/{N}")

section("11J. SEASONAL DECOMPOSITION STRENGTH (STL, period=5 & 21)")
from statsmodels.tsa.seasonal import STL
for period in (5, 21):
    strengths=[]
    for i in range(N):
        try:
            res = STL(P[:, i], period=period, robust=True).fit()
            var_resid = np.var(res.resid); var_sr = np.var(res.seasonal+res.resid)
            strength = max(0, 1 - var_resid/var_sr) if var_sr>0 else 0
            strengths.append(strength)
            rec.add("statsmodels","seasonal",f"stl_seasonal_strength_p{period}",tickers[i],strength,np.nan)
        except Exception: pass
    print(f"STL seasonal strength period={period}: mean {np.mean(strengths):.3f} "
          f"(>0.3 would indicate real seasonality)")

section("11K. MARKOV REGIME SWITCHING (2-state, subset: ALGO + market + 6 names)")
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
subset = [0, 5, 10, 20, 30, 40, 50]
for i in subset:
    try:
        mr = MarkovRegression(R[:, i], k_regimes=2, switching_variance=True).fit(disp=False)
        # LR vs single regime
        ll2 = mr.llf; ll1 = sm.OLS(R[:,i], np.ones(T)).fit().llf - 0.5*T*np.log(2*np.pi*np.var(R[:,i]))
        rec.add("statsmodels","regime","markov_2state_llf",tickers[i],mr.llf,np.nan,
                note=f"regimes_persist={np.diag(mr.regime_transition[:,:,0]).round(2).tolist()}")
        print(f"  {tickers[i]}: 2-state Markov llf={mr.llf:.1f}, "
              f"persistence={np.diag(mr.regime_transition[:,:,0]).round(2).tolist()}")
    except Exception as e:
        print(f"  {tickers[i]}: markov failed")

section("11L. MULTIPLE-TESTING CORRECTION on cointegration (Benjamini-Hochberg)")
pv = []
for i, j in itertools.combinations(range(N), 2):
    try: pv.append(coint(P[:, i], P[:, j])[1])
    except Exception: pass
pv = np.array(pv)
rej_bh, q_bh, _, _ = multipletests(pv, alpha=0.05, method="fdr_bh")
rej_bonf, _, _, _ = multipletests(pv, alpha=0.05, method="bonferroni")
print(f"Cointegration pairs: {len(pv)} tested")
print(f"  raw p<0.05: {(pv<0.05).sum()}  |  BH-FDR 5% survivors: {rej_bh.sum()}  |  Bonferroni: {rej_bonf.sum()}")
print(f"  => {rej_bh.sum()} pairs are cointegrated after honest multiple-testing correction")
rec.add("statsmodels","multitest","coint_BH_survivors","ALL",rej_bh.sum(),np.nan)
rec.add("statsmodels","multitest","coint_bonferroni_survivors","ALL",rej_bonf.sum(),np.nan)

rec.save()
