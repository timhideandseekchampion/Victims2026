"""Module 6: Volatility modelling + remaining test batteries.

Libraries: arch (GARCH, ARCH-LM), statsmodels (Granger causality, Engle ARCH-LM,
CUSUM structural-break, ACF of squared returns), numpy (Hurst exponent, runs test).

Covers: volatility clustering (GARCH), lead-lag causality (Granger),
long-memory / mean-reversion (Hurst), structural breaks, and day-of-cycle
seasonality.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import grangercausalitytests, acf
from scipy import stats
from common import load, log_returns, section, stars

df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape
mkt = R.mean(axis=1)

section("6A. ARCH-LM TEST + GARCH(1,1) - volatility clustering")
print("Engle ARCH-LM H0: no ARCH (homoskedastic). Reject => volatility clusters.\n")
n_arch = 0
garch_rows = []
for i, t in enumerate(tickers):
    r = R[:, i] * 100
    lm, lmp, f, fp = het_arch(r, nlags=5)
    if lmp < 0.05:
        n_arch += 1
    if i < 8 or lmp < 0.01:
        try:
            am = arch_model(r, vol="Garch", p=1, q=1, mean="Constant", dist="normal")
            fit = am.fit(disp="off")
            a, b = fit.params.get("alpha[1]", np.nan), fit.params.get("beta[1]", np.nan)
            garch_rows.append([t, lmp, a, b, a + b])
        except Exception:
            pass
print(f"Instruments with significant ARCH effects (LM p<0.05): {n_arch}/{N}")
if garch_rows:
    g = pd.DataFrame(garch_rows, columns=["tkr","ARCH_LM_p","alpha","beta","persist"])
    print("\nGARCH(1,1) fits (persist=alpha+beta; ~1 => strong vol memory):")
    print(g.round(4).to_string(index=False))
print(f"\n=> {'Volatility clustering present' if n_arch>N*0.2 else 'Little/no volatility clustering'} "
      f"({n_arch}/{N}). Squared-return ACF check:")
sq_ac1 = [acf(R[:, i]**2, nlags=1, fft=True)[1] for i in range(N)]
print(f"   mean lag-1 ACF of squared returns: {np.mean(sq_ac1):+.4f} "
      f"({(np.array(sq_ac1)>0).sum()}/{N} positive)")

section("6B. GRANGER CAUSALITY - does ALGO (market) lead other names?")
print("H0: ALGO returns do NOT Granger-cause name's returns (lag up to 3).\n")
lead_hits = 0
rows = []
for i in range(1, N):
    data = np.column_stack([R[:, i], R[:, 0]])   # [y=name, x=ALGO]
    try:
        res = grangercausalitytests(data, maxlag=3, verbose=False)
        p = min(res[l][0]["ssr_ftest"][1] for l in (1, 2, 3))
        if p < 0.05:
            lead_hits += 1
            rows.append((tickers[i], p))
    except Exception:
        pass
rows.sort(key=lambda x: x[1])
print(f"Names Granger-caused by ALGO (p<0.05): {lead_hits}/{N-1} "
      f"(expected under null ~{0.05*(N-1):.0f})")
print("Strongest:", ", ".join(f"{t}({p:.3f})" for t, p in rows[:10]) or "(none)")

section("6C. HURST EXPONENT - trending vs mean-reverting")
print("H<0.5 mean-reverting, =0.5 random walk, >0.5 trending/persistent.\n")
def hurst(ts):
    lags = range(2, 20)
    tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
    return np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
h_prices = [hurst(df.iloc[:, i].values) for i in range(N)]
h_rets = [hurst(R[:, i]) for i in range(N)]
print(f"Hurst on PRICES: mean {np.mean(h_prices):.3f} (min {np.min(h_prices):.3f}, max {np.max(h_prices):.3f})")
print(f"Hurst on RETURNS: mean {np.mean(h_rets):.3f}")
print(f"  prices with H<0.45 (mean-reverting): {(np.array(h_prices)<0.45).sum()}/{N}")
print(f"  prices with H>0.55 (trending):       {(np.array(h_prices)>0.55).sum()}/{N}")

section("6D. STRUCTURAL BREAKS - CUSUM test on cumulative returns")
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm
nb = 0
for i in range(N):
    y = R[:, i]
    X = sm.add_constant(np.arange(len(y)))
    resid = sm.OLS(y, X).fit().resid
    try:
        stat, p, _ = breaks_cusumolsresid(resid)
        if p < 0.05:
            nb += 1
    except Exception:
        pass
print(f"Instruments with CUSUM structural break in mean (p<0.05): {nb}/{N}")

section("6E. RUNS TEST + SEASONALITY")
# Wald-Wolfowitz runs test for randomness of sign sequence (market)
def runs_test(x):
    s = np.sign(x - np.median(x)); s = s[s != 0]
    runs = 1 + np.sum(s[1:] != s[:-1])
    n1 = np.sum(s > 0); n2 = np.sum(s < 0); n = n1 + n2
    er = 2*n1*n2/n + 1
    vr = (2*n1*n2*(2*n1*n2-n))/(n**2*(n-1))
    z = (runs - er)/np.sqrt(vr)
    return z, 2*(1-stats.norm.cdf(abs(z)))
z, p = runs_test(mkt)
print(f"Runs test on market return signs: z={z:.2f} p={p:.3f} {stars(p)} "
      f"({'non-random' if p<0.05 else 'consistent with randomness'})")
# day-of-cycle: is there a weekly (5-day) periodicity in market return?
per = [mkt[k::5].mean() for k in range(5)]
fstat, fp = stats.f_oneway(*[mkt[k::5] for k in range(5)])
print(f"5-day-cycle mean returns: {[f'{x*1e4:.1f}bp' for x in per]}")
print(f"ANOVA across cycle positions: F={fstat:.2f} p={fp:.3f} {stars(fp)}")

section("6F. VERDICT")
print(f"ARCH/vol-clustering: {n_arch}/{N} | Granger by ALGO: {lead_hits}/{N-1} | "
      f"CUSUM breaks: {nb}/{N}")
print(f"Mean Hurst(prices): {np.mean(h_prices):.3f} "
      f"({'mildly mean-reverting' if np.mean(h_prices)<0.48 else 'random-walk-like'})")
