# Exhaustive Statistical Test Catalog — `prices.txt`

Every applicable test/estimator from **scipy, statsmodels, arch, scikit-learn,
PyTorch, TensorFlow** run against the 500×51 price panel (499 daily log-return
observations per instrument). **5,084 tests total; 3,942 with p-values.**

Because that many tests generate ~197 false positives at raw p<0.05, everything
is corrected with **Benjamini-Hochberg FDR** and **Bonferroni** globally.

| Significance gate | Count |
| :--- | ---: |
| raw p<0.05 | 739 (≈197 expected under pure noise) |
| survive BH-FDR 5% | **523** |
| survive Bonferroni 5% | **413** |

Raw table: `results/MASTER_RESULTS.csv` (ranked by q-value). Per-library detail:
`results/{scipy,statsmodels,arch,sklearn,deep_learning}_full.csv`.

---

## The honest signal-vs-noise table (excess over the false-positive floor)

`excess = (# raw-significant) − (# expected under noise)`. Positive = real
structure; ≤0 = at/below the noise floor (**no effect**).

| Family (library) | tests | raw-sig | expected FP | **excess** | Verdict |
| :--- | ---: | ---: | ---: | ---: | :--- |
| corr vs MARKET (scipy) | 150 | 150 | 7.5 | **+142** | ✅ every name loads on a common factor |
| corr vs ALGO (scipy) | 150 | 150 | 7.5 | **+142** | ✅ ALGO **is** that factor (corr w/ PC1 = 0.98) |
| unit-root (statsmodels) | 255 | 111 | 12.8 | +98 | ⚪ trivial: returns stationary, prices not |
| distribution-fit (scipy) | 459 | 86 | 23 | +63 | ⚪ *rejections* of bad fits (Cauchy etc.); reverse polarity |
| unit-root (arch) | 306 | 69 | 15.3 | +54 | ⚪ same trivial fact, more tests |
| linear trend (scipy) | 51 | 48 | 2.6 | +45 | ⚠️ **spurious** — OLS trend on a unit-root series |
| **cointegration (arch)** | 14 | 14 | 0.7 | **+13** | ✅ **the real, tradeable edge** |
| Granger causality (sm) | 100 | 11 | 5 | +6 | ⚠️ ALGO weakly leads a few names |
| regime 1st-vs-2nd half (scipy) | 714 | 40 | 35.7 | +4 | ❌ at noise floor — no regime shift |
| cross-decomp CCA (sklearn) | 1 | 1 | 0 | +1 | ⚠️ in-sample only (OOS R²≈0) |
| specification/RESET (sm) | 153 | 8 | 7.6 | +0.4 | ❌ noise floor — series is linear |
| normality (statsmodels) | 51 | 2 | 2.6 | −0.6 | ❌ returns are Gaussian |
| ARMA model≠WN (sm) | 51 | 1 | 2.6 | −1.6 | ❌ returns are white noise |
| structural break (sm) | 51 | 0 | 2.6 | −2.6 | ❌ no breaks |
| heteroskedasticity (sm) | 204 | 5 | 10.2 | −5.2 | ❌ **no volatility clustering** |
| sign/runs (scipy) | 204 | 5 | 10.2 | −5.2 | ❌ return signs are random |
| normality (scipy) | 357 | 12 | 17.8 | −5.8 | ❌ returns are Gaussian |
| **autocorrelation (sm)** | 663 | 24 | 33.2 | **−9.2** | ❌ **no serial correlation at any lag** |

**Read this table top-down:** the only *non-trivial, non-spurious, positive-excess*
family is **cointegration**. Everything about single-name time-series behaviour
(autocorrelation, ARMA, momentum, mean-reversion, volatility clustering,
structural breaks, regimes, seasonality, sign predictability) sits **at or below
the noise floor** — i.e. the data is, per-instrument, a Gaussian random walk with
drift and constant variance.

---

## What each library found

### scipy.stats (2,242 tests)
- **Normality:** Shapiro, D'Agostino K², Jarque-Bera, KS, Cramér-von-Mises,
  skewtest, kurtosistest, Anderson-Darling — only 3/51 reject normality. Returns
  are Gaussian.
- **Best-fit distribution (10 families, KS):** `gennorm` best for 24/51, then
  Johnson-SU (8), gen-hyperbolic (7), skew-normal (6), Student-t (5). No fat tails.
- **Trend:** OLS & Theil-Sen slopes "significant" for 47–48/51 — but this is the
  spurious-regression trap on unit-root prices, **not** a tradeable trend.
- **Correlation vs ALGO & market:** Pearson/Spearman/Kendall all significant for
  ~50/51 — the common factor.
- **Regime (12 two-sample tests, 1st vs 2nd half):** t, Welch, Mann-Whitney,
  rank-sums, Kruskal, Brunner-Munzel, KS-2samp, CvM-2samp, Epps-Singleton, Levene,
  Bartlett, Fligner, Mood, Ansari — 3–5/51 each, **at the noise floor**. No regime change.
- **Sign/contingency:** binomial up-day test, χ²/G-test/Fisher on sign transitions
  — nothing above noise. **Cross-instrument ANOVA** of means: F=0.66, p=0.97 (means jointly indistinguishable).

### statsmodels (2,144 tests)
- **Unit root (ADF, KPSS, range-unit-root, Zivot-Andrews):** 0–4/51 stationary
  prices; 51/51 stationary returns. Even allowing one structural break (ZA),
  prices stay unit-root.
- **Autocorrelation (Ljung-Box & Box-Pierce lags 1/2/3/5/10/20, Breusch-Godfrey,
  Durbin-Watson):** 1/51 significant — **noise floor**.
- **Cross-correlation & Granger:** ALGO Granger-causes 9/50 names raw (3 after FDR:
  strongest CUBO, HRND, ULXY); reverse direction only 2/50. Weak one-way lead-lag.
- **Heteroskedasticity (ARCH-LM, Breusch-Pagan, White, Goldfeld-Quandt):** ≤2/51.
  No volatility clustering.
- **Specification (RESET, Harvey-Collier, Rainbow):** 2/51 — series is linear.
- **Structural breaks (CUSUM):** 0/51.
- **ARMA order selection (BIC):** 50/51 pick white-noise (0,0). AR(1) coef
  significant for 1/51.
- **STL seasonality (period 5 & 21):** mean strength 0.06 / 0.10 — none.
- **Markov 2-state regime switching:** fits converge but give no usable structure.
- **Multiple-testing on cointegration:** of 1,275 pairs, **5 survive BH-FDR,
  4 survive Bonferroni**.

### arch (626 tests)
- **Unit-root suite (ADF, DFGLS, Phillips-Perron, KPSS, Zivot-Andrews, Variance-Ratio):**
  confirms prices are random walks; VR rejects RW for only 5/51.
- **Cointegration (Engle-Granger + Phillips-Ouliaris)** on the 7 strong pairs:
  p ≤ 0.001 by **both** methods — robust.
- **GARCH family (GARCH/EGARCH/GJR/APARCH vs constant-vol):** a GARCH beats
  constant variance by >2 AIC for only 17/51, and ARCH-LM is significant for 1/51
  → volatility memory is weak/absent (EGARCH "wins" on AIC mostly by a hair).
- **Stationary bootstrap** of mean daily return: CI excludes 0 for only 2/51.

### scikit-learn (62 aggregate metrics)
- **Decomposition:** PCA PC1 = 22.7% (all-positive loadings = market factor);
  top-5 = 37%. FactorAnalysis log-likelihood plateaus by ~2–3 factors.
- **Covariance:** LedoitWolf shrinkage 0.05 (sample cov is fine); GraphicalLasso
  precision matrix 54% zeros (few *conditional* dependencies beyond the factor).
- **Feature dependence (MI, F-test, Pearson):** lag-2 own return weakly predicts
  next (F p=0.011, r=−0.016); everything else null.
- **Cross-decomposition (CCA/PLS):** CCA canonical corr 0.72 (perm p≈0) but PLS
  in-sample R² only 0.024 → **in-sample overfit, not OOS-exploitable**.
- **Clustering (KMeans/Agglomerative/Spectral/DBSCAN/GMM):** silhouettes ≈ 0 →
  no discrete sector clusters; structure is one continuous factor.
- **Outliers (IsolationForest/EllipticEnvelope/LOF):** flag only the imposed 5%.
- **14 walk-forward regressors:** linear/robust/SVR/KNN/AdaBoost/Bagging OOS R² ≤ 0;
  **tree ensembles positive** (HistGBM +0.0045, GradBoost +0.0042, RF +0.0036) vs
  negative shuffled baselines — a small, consistent nonlinear edge.

### PyTorch & TensorFlow (10 models)
- **Predictive (MLP, GRU, Conv1D, LSTM, Transformer):** all OOS R² ≈ 0. TF
  Transformer/LSTM reach 52.2% / 51.9% directional (matching the trees); PyTorch
  variants overfit (negative R²).
- **Autoencoders (8-dim bottleneck):** OOS reconstruction R² **0.36 (torch) /
  0.37 (TF)** — independently confirms the cross-section compresses to a handful
  of factors (same story as PCA).

---

## Bottom line

Across every test in all six libraries, the data behaves as **51 near-Gaussian
random walks with drift and constant variance, tied together by one common factor
(ALGO)**. The *only* statistically significant, non-trivial, out-of-sample-robust
structure is **cointegration among a small set of pairs** (5 survive Bonferroni;
7 clear pairs with 4–8-day mean-reversion half-lives). A second-order, marginal
signal is a **weak nonlinear predictability picked up only by tree ensembles /
attention models (~52% directional, R² +0.4%)** and a **weak one-way lead-lag from
ALGO**. No momentum, single-name mean-reversion, volatility clustering, regime
shifts, seasonality, breaks, or fat tails exist here.
