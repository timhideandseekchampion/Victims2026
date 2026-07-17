"""Full per-pair cointegration detail for ALL 1275 pairs.

For each pair (i<j): Engle-Granger p (both directions, take min), OLS hedge beta,
spread half-life (OU), spread ADF p, correlation, current spread z-score.
Then Benjamini-Hochberg FDR across all 1275 and full ranked CSV.
"""
import warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from common import prices_array, log_returns, section, RESULTS
import os

P, df, tickers = prices_array()
N, T = P.shape
rets = log_returns(df)
C = rets.corr().values


def half_life(spread):
    s = pd.Series(spread); ds = s.diff().dropna(); lag = s.shift(1).dropna()[1:]; ds = ds[1:]
    beta = sm.OLS(ds.values, sm.add_constant(lag.values)).fit().params[1]
    return (-np.log(2)/beta) if beta < 0 else np.inf


section("16A. ENGLE-GRANGER ON ALL 1275 PAIRS (both directions)")
rows = []
for i, j in itertools.combinations(range(N), 2):
    yi, yj = P[i], P[j]
    try:
        p_ij = coint(yi, yj)[1]; p_ji = coint(yj, yi)[1]
        p = min(p_ij, p_ji)
        # orient regression the way with lower p
        if p_ij <= p_ji:
            beta = sm.OLS(yi, sm.add_constant(yj)).fit().params[1]; spread = yi - beta*yj
        else:
            beta = sm.OLS(yj, sm.add_constant(yi)).fit().params[1]; spread = yj - beta*yi
        hl = half_life(spread)
        adfp = adfuller(spread, autolag="AIC")[1]
        z = (spread[-1]-spread.mean())/(spread.std()+1e-12)
        rows.append([tickers[i], tickers[j], p, beta, hl, adfp, C[i, j], z])
    except Exception:
        pass
cd = pd.DataFrame(rows, columns=["a","b","coint_p","hedge_beta","half_life_d",
                                 "spread_adf_p","ret_corr","z_now"])
rej_bh, q_bh, _, _ = multipletests(cd.coint_p.values, alpha=0.05, method="fdr_bh")
rej_bonf, _, _, _ = multipletests(cd.coint_p.values, alpha=0.05, method="bonferroni")
cd["q_bh"] = q_bh; cd["sig_bh"] = rej_bh; cd["sig_bonf"] = rej_bonf
cd = cd.sort_values("coint_p").reset_index(drop=True)

print(f"Pairs tested: {len(cd)}")
for thr in (0.001, 0.01, 0.05):
    print(f"  coint_p<{thr}: {(cd.coint_p<thr).sum():>4}  (expected under null ~{thr*len(cd):.1f})")
print(f"  survive BH-FDR 5%:     {cd.sig_bh.sum()}")
print(f"  survive Bonferroni 5%: {cd.sig_bonf.sum()}")

section("16B. ALL PAIRS SURVIVING BH-FDR (the real cointegrated set)")
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
surv = cd[cd.sig_bh].copy()
print(surv.round(4).to_string(index=False))

section("16C. TRADEABILITY FILTER (FDR-significant + stationary spread + half-life 2-40d)")
trade = surv[(surv.spread_adf_p < 0.05) & (surv.half_life_d.between(2, 40))]
print(f"{len(trade)} pairs are FDR-cointegrated AND have a fast, stationary spread:")
print(trade.round(4).to_string(index=False))

section("16D. HUB INSTRUMENTS (appear most in FDR-significant pairs)")
from collections import Counter
cnt = Counter()
for _, r in surv.iterrows():
    cnt[r.a] += 1; cnt[r.b] += 1
hub = pd.DataFrame(cnt.most_common(15), columns=["ticker","n_coint_pairs"])
print(hub.to_string(index=False))

cd.to_csv(os.path.join(RESULTS, "coint_all_pairs.csv"), index=False)
print(f"\nFull 1275-pair table -> results/coint_all_pairs.csv")
