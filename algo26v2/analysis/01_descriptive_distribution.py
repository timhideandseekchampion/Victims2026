"""Module 1: Descriptive stats + distribution/normality tests.

Libraries: numpy, pandas, scipy.stats, statsmodels.
For every instrument's daily log returns we compute moments and run the full
battery of normality / fat-tail tests. Fat tails + non-normality are expected;
what we care about is HOW non-normal (t-dist dof) and which instruments differ.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import jarque_bera as sm_jb
from statsmodels.stats.diagnostic import lilliefors
from common import prices_array, log_returns, section, stars, load

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_rows", 60)

df, tickers = load()
rets = log_returns(df)
T, N = df.shape
section(f"DATA SHAPE: {T} days x {N} instruments; {len(rets)} return obs")
print("Tickers:", " ".join(tickers))
print(f"\nInstrument 0 = {tickers[0]} (SPECIAL: comm 0.2bp, poslimit $100k)")

section("1A. PRICE-LEVEL DESCRIPTIVE STATS")
pdesc = df.describe().T[["mean", "std", "min", "max"]]
pdesc["last"] = df.iloc[-1].values
pdesc["total_drift_%"] = (df.iloc[-1].values / df.iloc[0].values - 1) * 100
print(pdesc.round(2))

section("1B. LOG-RETURN MOMENTS (daily) + ANNUALISED")
rows = []
for i, t in enumerate(tickers):
    r = rets.iloc[:, i].values
    mu, sd = r.mean(), r.std(ddof=1)
    sk = stats.skew(r)
    ku = stats.kurtosis(r)  # excess
    ann_ret = mu * 250 * 100
    ann_vol = sd * np.sqrt(250) * 100
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    rows.append([t, mu * 1e4, sd * 1e4, sk, ku, ann_ret, ann_vol, sharpe])
mom = pd.DataFrame(rows, columns=["tkr", "mu_bp", "sd_bp", "skew", "exkurt",
                                  "annRet%", "annVol%", "BH_Sharpe"])
print(mom.round(3).to_string(index=False))
print("\nBuy&Hold Sharpe leaders:")
print(mom.reindex(mom.BH_Sharpe.abs().sort_values(ascending=False).index).head(10).round(3).to_string(index=False))

section("1C. NORMALITY / FAT-TAIL TEST BATTERY (per instrument)")
print("Tests: Jarque-Bera, D'Agostino K2, Shapiro-Wilk, Anderson-Darling, "
      "Kolmogorov-Smirnov(norm), Lilliefors. p<0.05 => reject normality.\n")
rows = []
for i, t in enumerate(tickers):
    r = rets.iloc[:, i].values
    jb_stat, jb_p, _, _ = sm_jb(r)
    k2_stat, k2_p = stats.normaltest(r)
    sw_stat, sw_p = stats.shapiro(r)
    ad = stats.anderson(r, dist="norm")
    ad_reject5 = ad.statistic > ad.critical_values[2]  # 5% level
    z = (r - r.mean()) / r.std(ddof=0)
    ks_stat, ks_p = stats.kstest(z, "norm")
    lf_stat, lf_p = lilliefors(r, dist="norm")
    # fit Student-t to get tail heaviness (dof)
    tdof, _, _ = stats.t.fit(r)
    rows.append([t, jb_p, k2_p, sw_p, "Y" if ad_reject5 else "n",
                 ks_p, lf_p, tdof])
nt = pd.DataFrame(rows, columns=["tkr", "JB_p", "K2_p", "Shapiro_p",
                                 "AD_rej5%", "KS_p", "Lillie_p", "t_dof"])
print(nt.round(4).to_string(index=False))
n_nonnorm = (nt.JB_p < 0.05).sum()
section("1D. SUMMARY")
print(f"Instruments rejecting normality (Jarque-Bera p<0.05): {n_nonnorm}/{N}")
print(f"Median Student-t dof: {nt.t_dof.median():.2f} "
      f"(low dof = fat tails; <30 => materially heavier than normal)")
print(f"Mean excess kurtosis: {mom.exkurt.mean():.3f} "
      f"(>0 => leptokurtic/fat-tailed)")
print(f"Mean skew: {mom.skew.mean():.3f}")
print("\nInterpretation: returns are non-Gaussian & fat-tailed (as expected for")
print("financial data). This justifies robust/rank methods downstream and warns")
print("against Sharpe-optimisation that assumes normality.")
