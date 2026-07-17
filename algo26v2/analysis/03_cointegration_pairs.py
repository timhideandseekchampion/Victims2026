"""Module 3: Cross-sectional structure - correlation, cointegration, pairs.

Libraries: statsmodels (coint Engle-Granger, Johansen, OLS), scipy, sklearn.
The single-name search failed, so we hunt for:
  (a) correlated return blocks (common factors / sectors),
  (b) cointegrated pairs -> stationary tradeable spread (stat-arb),
  (c) half-life of spread mean-reversion (is it fast enough vs commissions?).
"""
import warnings
warnings.filterwarnings("ignore")
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
from common import load, log_returns, section, stars

df, tickers = load()
rets = log_returns(df)
N = len(tickers)

section("3A. RETURN CORRELATION STRUCTURE")
C = rets.corr()
off = C.values[np.triu_indices(N, 1)]
print(f"Off-diagonal return correlations: mean={off.mean():+.4f} "
      f"std={off.std():.4f} min={off.min():+.3f} max={off.max():+.3f}")
print(f"|corr|>0.3: {(np.abs(off)>0.3).sum()} pairs;  >0.5: {(np.abs(off)>0.5).sum()} pairs")
# top correlated pairs
pairs = []
for i, j in itertools.combinations(range(N), 2):
    pairs.append((tickers[i], tickers[j], C.iloc[i, j]))
pairs.sort(key=lambda x: -abs(x[2]))
print("\nTop 15 |return correlation| pairs:")
for a, b, c in pairs[:15]:
    print(f"  {a}-{b}: {c:+.3f}")

section("3B. ENGLE-GRANGER COINTEGRATION - ALL 1275 PAIRS")
print("H0: no cointegration. p<0.05 => prices share a stationary linear combo (spread).")
print("With 1275 tests at 5%, ~64 false positives expected. We look for EXCESS + low p.\n")
res = []
P = df.values  # T x N
for i, j in itertools.combinations(range(N), 2):
    try:
        _, pval, _ = coint(P[:, i], P[:, j])
        res.append((tickers[i], tickers[j], i, j, pval))
    except Exception:
        pass
cdf = pd.DataFrame(res, columns=["a", "b", "i", "j", "p"])
n_sig = (cdf.p < 0.05).sum()
n_sig1 = (cdf.p < 0.01).sum()
print(f"Pairs tested: {len(cdf)}")
print(f"Cointegrated p<0.05: {n_sig} (expected under null ~{0.05*len(cdf):.0f})")
print(f"Cointegrated p<0.01: {n_sig1} (expected under null ~{0.01*len(cdf):.0f})")
print(f"Cointegrated p<0.001: {(cdf.p<0.001).sum()} (expected ~{0.001*len(cdf):.1f})")


def half_life(spread):
    s = pd.Series(spread)
    ds = s.diff().dropna()
    lag = s.shift(1).dropna()[1:]
    ds = ds[1:]
    X = sm.add_constant(lag.values)
    beta = sm.OLS(ds.values, X).fit().params[1]
    if beta >= 0:
        return np.inf
    return -np.log(2) / beta


print("\nStrongest cointegrated pairs (lowest p) with hedge ratio & half-life:")
top = cdf.sort_values("p").head(25).copy()
rows = []
for _, r in top.iterrows():
    x = P[:, int(r.j)]; y = P[:, int(r.i)]
    beta = sm.OLS(y, sm.add_constant(x)).fit().params[1]
    spread = y - beta * x
    hl = half_life(spread)
    adf_p = adfuller(spread, autolag="AIC")[1]
    z_now = (spread[-1] - spread.mean()) / spread.std()
    rows.append([r.a, r.b, r.p, beta, hl, adf_p, z_now])
hp = pd.DataFrame(rows, columns=["a", "b", "coint_p", "hedge_beta",
                                 "half_life_d", "spread_adf_p", "z_now"])
print(hp.round(4).to_string(index=False))
tradeable = hp[(hp.half_life_d > 1) & (hp.half_life_d < 60) & (hp.spread_adf_p < 0.05)]
print(f"\nPairs with stationary spread AND half-life 1-60d (tradeable stat-arb): {len(tradeable)}")

section("3C. JOHANSEN TEST on the most-cointegrated basket")
# take the instruments appearing most in significant pairs
from collections import Counter
cnt = Counter()
for _, r in cdf[cdf.p < 0.02].iterrows():
    cnt[r.a] += 1; cnt[r.b] += 1
basket = [t for t, _ in cnt.most_common(6)]
if len(basket) >= 2:
    idx = [tickers.index(t) for t in basket]
    Y = P[:, idx]
    jo = coint_johansen(Y, det_order=0, k_ar_diff=1)
    print(f"Basket: {basket}")
    print("Trace stat vs 95% crit (rank r cointegrating relations):")
    for r in range(len(basket)):
        print(f"  r<={r}: trace={jo.lr1[r]:.2f} crit95={jo.cvt[r,1]:.2f} "
              f"{'REJECT->more relations' if jo.lr1[r]>jo.cvt[r,1] else 'stop'}")
else:
    print("Not enough cointegrated instruments for a basket.")

section("3D. VERDICT")
print(f"Return corr: mean {off.mean():+.3f} (near 0 => weak common factor)")
print(f"Cointegration p<0.01: {n_sig1} vs ~{0.01*len(cdf):.0f} expected under null "
      f"=> {'MATERIAL excess (real pairs)' if n_sig1 > 2*0.01*len(cdf) else 'near noise floor'}")
print(f"Tradeable stat-arb pairs (stationary + sane half-life): {len(tradeable)}")
tradeable.to_csv("results/tradeable_pairs.csv", index=False)
print("Saved candidate pairs -> results/tradeable_pairs.csv")
