"""Module 2: Stationarity, autocorrelation, random-walk vs mean-reversion.

Libraries: statsmodels (adfuller, kpss, acf/pacf, Ljung-Box, Durbin-Watson,
Zivot-Andrews), arch (variance-ratio via bootstrap, BDS via statsmodels).

This is the core alpha search: if daily returns have significant serial
correlation we can trade momentum (positive AC) or mean-reversion (negative AC).
If prices are stationary (reject unit root) they mean-revert to a level.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf, bds
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from common import load, log_returns, section, stars

df, tickers = load()
rets = log_returns(df)
N = len(tickers)

section("2A. UNIT-ROOT / STATIONARITY ON PRICE LEVELS")
print("ADF H0: has unit root (non-stationary). KPSS H0: stationary.")
print("If ADF rejects (p<0.05) AND KPSS does not -> price is STATIONARY / mean-reverts.\n")
rows = []
for i, t in enumerate(tickers):
    p = df.iloc[:, i].values
    adf_stat, adf_p, *_ = adfuller(p, autolag="AIC")
    try:
        kpss_stat, kpss_p, *_ = kpss(p, regression="c", nlags="auto")
    except Exception:
        kpss_p = np.nan
    verdict = "STATIONARY(MR)" if (adf_p < 0.05 and (np.isnan(kpss_p) or kpss_p > 0.05)) else \
              "unit-root(RW)" if adf_p > 0.05 else "mixed"
    rows.append([t, adf_p, kpss_p, verdict])
lv = pd.DataFrame(rows, columns=["tkr", "ADF_p", "KPSS_p", "verdict"])
print(lv.round(4).to_string(index=False))
print(f"\nStationary (mean-reverting) price levels: {(lv.verdict=='STATIONARY(MR)').sum()}/{N}")
print("Tickers:", " ".join(lv[lv.verdict=='STATIONARY(MR)'].tkr.tolist()) or "(none)")

section("2B. STATIONARITY ON RETURNS (sanity: should be stationary)")
n_stat = sum(adfuller(rets.iloc[:, i].values, autolag="AIC")[1] < 0.05 for i in range(N))
print(f"Returns rejecting unit root (stationary) by ADF: {n_stat}/{N}")

section("2C. SERIAL CORRELATION OF RETURNS  (momentum vs mean-reversion)")
print("Ljung-Box(10 lags) H0: no autocorrelation. Durbin-Watson ~2 = none, <2 pos, >2 neg.")
print("AC1 = lag-1 autocorr: >0 momentum, <0 mean-reversion.\n")
rows = []
for i, t in enumerate(tickers):
    r = rets.iloc[:, i].values
    a = acf(r, nlags=5, fft=True)
    lb = acorr_ljungbox(r, lags=[1, 5, 10], return_df=True)
    dw = durbin_watson(r)
    rows.append([t, a[1], a[2], a[3], dw,
                 lb["lb_pvalue"].iloc[0], lb["lb_pvalue"].iloc[2]])
ac = pd.DataFrame(rows, columns=["tkr", "AC1", "AC2", "AC3", "DW",
                                 "LB1_p", "LB10_p"])
ac["sig1"] = ac.LB1_p.apply(stars)
print(ac.round(4).to_string(index=False))
n_sig1 = (ac.LB1_p < 0.05).sum()
n_sig10 = (ac.LB10_p < 0.05).sum()
print(f"\nInstruments with significant lag-1 autocorr (LB p<0.05): {n_sig1}/{N}")
print(f"Instruments with significant autocorr up to lag10:       {n_sig10}/{N}")
print(f"Mean AC1: {ac.AC1.mean():+.4f}  (sign indicates net momentum/reversion)")
print(f"Count AC1<0 (mean-revert): {(ac.AC1<0).sum()},  AC1>0 (momentum): {(ac.AC1>0).sum()}")
print("\nStrongest |AC1| instruments:")
print(ac.reindex(ac.AC1.abs().sort_values(ascending=False).index)
      [["tkr","AC1","LB1_p","sig1"]].head(12).round(4).to_string(index=False))

section("2D. VARIANCE RATIO TEST (Lo-MacKinlay random-walk test)")
print("VR(q)=Var(q-day ret)/(q*Var(1-day)). VR>1 momentum, VR<1 mean-revert, =1 RW.")
print("z-stat |>1.96| => reject random walk at 5%.\n")
def variance_ratio(x, q):
    x = np.asarray(x); n = len(x)
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / n
    xq = np.array([x[i:i+q].sum() for i in range(n - q + 1)])
    varq = np.sum((xq - q * mu) ** 2) / (n * q)
    vr = varq / var1
    # heteroskedasticity-robust z (Lo-MacKinlay)
    phi = 0.0
    for j in range(1, q):
        dj = np.sum((x[j:] - mu) ** 2 * (x[:-j] - mu) ** 2)
        dj /= (np.sum((x - mu) ** 2)) ** 2
        phi += (2 * (q - j) / q) ** 2 * dj
    z = (vr - 1) / np.sqrt(phi) if phi > 0 else np.nan
    return vr, z
rows = []
for i, t in enumerate(tickers):
    r = rets.iloc[:, i].values
    out = [t]
    for q in (2, 5, 10):
        vr, z = variance_ratio(r, q)
        out += [vr, z]
    rows.append(out)
vr = pd.DataFrame(rows, columns=["tkr","VR2","z2","VR5","z5","VR10","z10"])
vr["rej5_any"] = (vr[["z2","z5","z10"]].abs() > 1.96).any(axis=1)
print(vr.round(3).to_string(index=False))
print(f"\nInstruments rejecting random walk (|z|>1.96 at some q): {vr.rej5_any.sum()}/{N}")
print(f"Mean VR5: {vr.VR5.mean():.3f}  (<1 => aggregate mean-reversion at weekly horizon)")

section("2E. BDS TEST FOR NONLINEAR STRUCTURE (a few instruments)")
print("H0: iid. Reject => nonlinear dependence exploitable by ML.\n")
for i in [0, 20, 40]:
    r = rets.iloc[:, i].values
    try:
        stat, p = bds(r, max_dim=3)
        print(f"{tickers[i]}: BDS dim3 stat={stat[-1]:.3f} p={p[-1]:.4f} {stars(p[-1])}")
    except Exception as e:
        print(f"{tickers[i]}: BDS failed ({e})")

section("2F. VERDICT")
print(f"- Stationary price levels: {(lv.verdict=='STATIONARY(MR)').sum()}/{N}")
print(f"- Sig lag-1 autocorr:      {n_sig1}/{N}")
print(f"- Reject random walk (VR): {vr.rej5_any.sum()}/{N}")
print("If these counts are near the ~5% false-positive floor (~2-3/51), single-name")
print("time-series predictability is WEAK -> pivot to cross-sectional / pairs alpha.")
