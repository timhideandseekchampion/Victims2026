"""FULL arch battery.

Families:
  - Unit-root suite (arch.unitroot): ADF, DFGLS, PhillipsPerron, KPSS,
    ZivotAndrews, VarianceRatio  (all 51 instruments, on prices)
  - Cointegration (arch): Engle-Granger, Phillips-Ouliaris on the strongest pairs
  - Volatility models: GARCH, EGARCH, GJR-GARCH(TARCH), APARCH - AIC compare
    & test whether ARCH terms are significant
  - Stationary bootstrap CI for each name's mean daily return (is drift real?)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from arch.unitroot import (ADF, DFGLS, PhillipsPerron, KPSS, ZivotAndrews,
                           VarianceRatio)
from arch.unitroot.cointegration import engle_granger, phillips_ouliaris
from arch import arch_model
from arch.bootstrap import StationaryBootstrap
from common import load, log_returns, section, Recorder

rec = Recorder("arch_full")
df, tickers = load()
rets = log_returns(df)
R = rets.values
P = df.values
T, N = R.shape

section("12A. UNIT-ROOT SUITE (6 tests x 51 instruments, on prices)")
counts = {k: 0 for k in ["ADF","DFGLS","PP","KPSS_nonstat","ZA","VR_rejectRW"]}
for i, t in enumerate(tickers):
    p = P[:, i]
    for name, cls in [("ADF", ADF), ("DFGLS", DFGLS), ("PP", PhillipsPerron)]:
        try:
            r_ = cls(p); rec.add("arch","unitroot",name,t,r_.stat,r_.pvalue)
            if r_.pvalue < 0.05: counts[name] += 1
        except Exception: pass
    try:
        k = KPSS(p); rec.add("arch","unitroot","KPSS",t,k.stat,k.pvalue)
        if k.pvalue < 0.05: counts["KPSS_nonstat"] += 1
    except Exception: pass
    try:
        za = ZivotAndrews(p); rec.add("arch","unitroot","ZivotAndrews",t,za.stat,za.pvalue)
        if za.pvalue < 0.05: counts["ZA"] += 1
    except Exception: pass
    try:
        vr = VarianceRatio(np.log(p), lags=5); rec.add("arch","unitroot","VarianceRatio_q5",t,vr.stat,vr.pvalue)
        if vr.pvalue < 0.05: counts["VR_rejectRW"] += 1
    except Exception: pass
print("Rejections at 5% (prices unless noted):")
for k, v in counts.items():
    print(f"  {k}: {v}/{N}")

section("12B. COINTEGRATION (Engle-Granger & Phillips-Ouliaris) - strong pairs")
PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("SMAH","ILVX"),("HUXZ","ACAC"),
         ("HETT","ULXY"),("CTGI","EELT"),("ACIX","ITPA")]
idx = {t: k for k, t in enumerate(tickers)}
for a, b in PAIRS:
    ya, yb = P[:, idx[a]], P[:, idx[b]]
    try:
        eg = engle_granger(ya, yb); rec.add("arch","coint","engle_granger",f"{a}-{b}",eg.stat,eg.pvalue)
    except Exception: eg = None
    try:
        po = phillips_ouliaris(ya, yb); rec.add("arch","coint","phillips_ouliaris",f"{a}-{b}",po.stat,po.pvalue)
    except Exception: po = None
    print(f"  {a}-{b}: EG p={getattr(eg,'pvalue',float('nan')):.4f}  "
          f"PO p={getattr(po,'pvalue',float('nan')):.4f}")

section("12C. GARCH FAMILY - AIC comparison + ARCH significance (all 51)")
better_than_const = 0
aic_win = {}
for i, t in enumerate(tickers):
    r = R[:, i] * 100
    fits = {}
    specs = {
        "GARCH":  dict(vol="Garch", p=1, q=1),
        "EGARCH": dict(vol="EGARCH", p=1, q=1),
        "GJR":    dict(vol="Garch", p=1, o=1, q=1),
        "APARCH": dict(vol="APARCH", p=1, o=1, q=1),
    }
    for name, kw in specs.items():
        try:
            fit = arch_model(r, mean="Constant", dist="normal", **kw).fit(disp="off")
            fits[name] = fit.aic
            rec.add("arch","garch",f"{name}_aic",t,fit.aic,np.nan)
        except Exception: pass
    # constant-variance baseline AIC
    try:
        base = arch_model(r, mean="Constant", vol="Constant").fit(disp="off").aic
        rec.add("arch","garch","Constant_aic",t,base,np.nan)
        if fits and min(fits.values()) < base - 2:  # >2 AIC gain = meaningful
            better_than_const += 1
        if fits:
            aic_win[min(fits, key=fits.get)] = aic_win.get(min(fits, key=fits.get), 0) + 1
    except Exception: pass
print(f"Instruments where a GARCH model beats constant-vol by >2 AIC: {better_than_const}/{N}")
print(f"AIC-winning vol model counts: {aic_win}")

section("12D. STATIONARY BOOTSTRAP - is mean daily return distinguishable from 0?")
np.random.seed(0)
nsig = 0
for i, t in enumerate(tickers):
    r = R[:, i]
    bs = StationaryBootstrap(10, r)
    ci = bs.conf_int(np.mean, 1000, method="percentile")
    lo, hi = ci[0, 0], ci[1, 0]
    signif = not (lo < 0 < hi)
    rec.add("arch","bootstrap","mean_return_ci",t,r.mean(),np.nan,
            note=f"ci=[{lo:.2e},{hi:.2e}] signif={signif}")
    if signif: nsig += 1
print(f"Instruments with bootstrap mean-return CI excluding 0: {nsig}/{N}")

rec.save()
