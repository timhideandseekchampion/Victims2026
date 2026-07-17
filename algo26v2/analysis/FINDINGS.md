# Statistical Analysis of `prices.txt` — Algothon 2026

**Data:** 500 days × 51 instruments (log prices). Instrument 0 (**ALGO**) is
special: 5× lower commission (0.2 bp) and 10× the dollar position limit ($100k).

**Libraries exercised:** `statsmodels`, `scipy`, `scikit-learn`, `PyTorch`,
`TensorFlow/Keras`, `arch`. Every script is in `analysis/`, raw output in
`analysis/results/`.

---

## TL;DR

| What was tested | Result | Tradeable? |
| :--- | :--- | :--- |
| Normality / fat tails | Near-Gaussian, excess kurt ≈ 0 | — (clean synthetic data) |
| Single-name autocorrelation | **1/51** significant (noise floor) | ❌ |
| Random-walk (variance ratio) | 6/51 reject (weak) | ❌ |
| Price mean-reversion (ADF/KPSS/Hurst) | 0/51 stationary; Hurst 0.476 | ❌ (too weak alone) |
| Volatility clustering (ARCH/GARCH) | 1/51 — constant vol | ❌ |
| Structural breaks / seasonality | None | ❌ |
| ML predictability (linear, MLP, LSTM) | OOS R² ≈ 0 | ❌ |
| ML predictability (tree ensembles) | OOS R² +0.4%, 52.3% dir. | ⚠️ marginal |
| **Cointegrated pairs (stat-arb)** | **7 pairs p<0.001 (vs 1.3 exp.)** | ✅ **YES** |
| ALGO = market factor (PCA) | corr(PC1,ALGO)=0.98, 22.7% var | ✅ (as hedge/factor) |
| ALGO lead-lag (Granger) | leads 9/50 names (vs 2 exp.) | ⚠️ real but unprofitable net |

**The one robust, profitable edge is cointegration pairs trading.**
Out-of-sample (pairs picked on days 0–250, traded on 250–500): **Sharpe 2.53,
score 84.5** vs the starter momentum strategy which *loses* money (Sharpe −0.38).

---

## 1. Distribution (scipy, statsmodels) — `results/01_descriptive.txt`
Jarque-Bera, D'Agostino K², Shapiro-Wilk, Anderson-Darling, KS, Lilliefors,
Student-t fit. Only **3/51** reject normality; mean excess kurtosis −0.04;
median t-dof effectively ∞. → Returns are essentially Gaussian with **constant**
variance. This is synthetic data with no fat tails to exploit and warns against
any strategy that relies on tail events.

## 2. Time-series predictability (statsmodels, arch) — `results/02_stationarity.txt`
- **ADF + KPSS on prices:** 0/51 stationary — no single name mean-reverts to a level.
- **Ljung-Box / Durbin-Watson / ACF:** only **1/51** (ANSO) has significant lag-1
  autocorrelation. Mean AC1 = −0.0002. → No momentum, no single-name reversion.
- **Lo-MacKinlay variance ratio:** 6/51 reject random walk (barely above the
  ~2–3 false-positive floor). Mean VR(5)=0.97 → faint aggregate weekly reversion.
- **BDS test:** no nonlinear structure.

→ **Single-instrument time-series alpha does not exist here.**

## 3. Cross-sectional structure (statsmodels, sklearn) — `results/03/04`
- **Return correlations:** mean +0.20; the 15 highest pairwise correlations are
  *all* ALGO-vs-something (0.48–0.59).
- **PCA:** PC1 = 22.7% of variance, all-positive loadings (a level/market factor),
  and **corr(PC1 score, ALGO return) = 0.975** → **ALGO *is* the market factor.**
- **Cointegration (Engle-Granger, all 1275 pairs):** 7 pairs at p<0.001 vs 1.3
  expected under the null — a **5× excess**, i.e. real. Strongest, with 4–8 day
  half-lives: `AENO-NWIG`, `EORC-NGTE`, `SMAH-ILVX`, `HUXZ-ACAC`, `HETT-ULXY`,
  `CTGI-EELT`, `ACIX-ITPA`.
- **Johansen:** confirms ≥1 cointegrating relation in the top basket.
- **Residual / cross-sectional reversal:** NOT significant (Sharpe 0.25, t=0.36) —
  the naive market-neutral reversal does not work; the edge is *pair-specific*.

## 4. Machine learning (sklearn, PyTorch, TensorFlow) — `results/05_ml.txt`
Walk-forward, compared to shuffled-target baselines.
- Ridge/Lasso/ElasticNet, PyTorch MLP, TensorFlow LSTM: **OOS R² ≤ 0** — no edge.
- RandomForest & GradientBoosting: **OOS R² +0.36% / +0.42%** (vs −0.23% / −0.29%
  shuffled), 52.3% directional accuracy. Small but consistent — nonlinear
  market-beta + mild reversion. *(The binomial p=1.6e-11 is overstated: the 51
  names each day are cross-sectionally correlated, so effective N ≈ 500 days.)*

## 5. Volatility & misc (arch, statsmodels) — `results/06_volatility.txt`
- **ARCH-LM / GARCH:** 1/51 significant → no volatility clustering (constant vol).
- **Granger causality:** ALGO Granger-causes **9/50** names (vs ~2 expected) —
  real lead-lag, but trading it directly loses money after commissions.
- **Hurst:** prices 0.476 (mildly mean-reverting), returns 0.50 (random walk).
- **CUSUM breaks: 0/51; runs test: random; 5-day seasonality: none.**

## 6. Strategy backtests (eval.py scoring) — `results/07`, `results/08`
| Strategy | Sharpe | Score |
| :--- | ---: | ---: |
| Starter (momentum) | −0.38 | −30.5 |
| ALGO lead-lag | −1.38 | −54.9 |
| Cross-sectional reversal | −0.29 | −13.6 |
| **Cointegration pairs (in-sample)** | **4.5–6.4** | **108–166** |
| **Cointegration pairs (true OOS)** | **2.53** | **84.5** |

---

## Recommendation
Build the submission around **cointegration pair trading**: rank pairs by
Engle-Granger p-value on a rolling window, trade the spread z-score
(enter ~0.5σ, hedge ratio from OLS), keep it market-neutral. Use ALGO's
oversized limit/cheap commission as the factor-hedge leg. Avoid momentum,
single-name signals, and heavy ML — they don't generalise here.
See `analysis/strategy_pairs.py` for a working `getMyPosition`.
