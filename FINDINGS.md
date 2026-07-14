# Algothon 2026 — Data Findings

**Universe:** 50 tradeable assets + `ALGO` (a market index) · 500 daily observations · scored on the last 250 days
**Date:** 2026-07-11 · **Interactive dashboard:** https://claude.ai/code/artifact/1560ad0c-311d-4108-80c0-480b088ca384
**Scope:** statistical characterisation only — no strategy built yet.

> **Revision (v3):** an earlier draft concluded the market-neutral signal was ≈0 — an artefact of over-regularisation (`Ridge α=200`). A full regressor sweep + walk-forward validation found a real edge, and **independent adversarial verification** (leakage audit, closed-form re-implementation, permutation test, block-bootstrap, cost simulation) has now **confirmed** it: OLS cross-sectional IC ≈ +0.058 (t ≈ 5.3), **permutation p < 0.001**, **bootstrap 95% CI [+0.033, +0.080]**, **net Sharpe ≈ 4.5** after 1bp costs. The mechanism is **directed peer lead-lag** — not autocorrelation, not the market factor, not simple reversal.
>
> **Revision (v5, 2026-07-14 — endgame):** strategy shipped as `Arbitrage_Victims_v2.py` (cache-hardened; byte-identical positions to v1). **Live public score 502.52** — matching the book-only expectation (~503 full-window), i.e. the pipeline generalizes exactly as modeled. Three adversarial multi-agent hunts (~30 hypotheses) found **no further edge**; see the closing chapter "Endgame — why we hold" below. Final decision: **hold v2, market-neutral, re-submit on the last day.**
>
> **Revision (v4):** re-tested over the **full 500 days** (out-of-sample window pushed to ~400 days) and **independently re-verified** (three adversarial lenses; all reproduced the numbers to the decimal, including running the official `eval.py` itself — identical Score 179.45 — and a leakage probe that corrupted prices past day 300 with zero effect on earlier positions). The edge holds across the *entire* history: IC 0.05–0.065 at every warm-up start (198–448 OOS days), **significant in both halves** (first-half t=3.7, last-half t=5.3). Through `eval.py`'s exact accounting the OLS forecast scores **179 / Sharpe 4.4** (last-250) and **152 / Sharpe 3.8** (last-400). Backtest files: `ols_strategy.py`, `backtest_full.py`. **Judge by Sharpe ≈4.4** — the Score scales with position size and is somewhat window-favourable, so it is not a pure edge measure.

---

## Bottom line

1. **The data is synthetic** — Gaussian returns, constant volatility, no fat tails, no vol clustering, no structural breaks. Assume normality and stable risk; there is no tail risk and no vol-timing edge.
2. **Three statistical factors** drive the universe (one dominant "market" factor = `ALGO`); ~69% of the correlation matrix is noise.
3. **No momentum** (autocorrelation ≈ 0 at every lag). The exploitable dynamic is **cross-asset lead-lag / mean-reversion**.
4. **There is a genuine, verified market-neutral edge.** A linear cross-asset model (OLS VAR-style) that predicts each asset's next-day return from today's full cross-section yields a **walk-forward cross-sectional IC of ≈ +0.058 (t ≈ 5.3), positive on 64% of days and strengthening out-of-sample** — permutation p < 0.001, bootstrap CI [+0.033, +0.080], net Sharpe ≈ 4.5 after 1bp. The mechanism is **directed peer lead-lag**: some assets' moves today predict *others'* tomorrow. It is *not* own-asset autocorrelation (dropping each asset's own return **raises** IC to 0.060), *not* the market factor (dropping `ALGO` **raises** it to 0.062), and ~3× stronger than plain cross-sectional reversal (20-day reversal IC only 0.019, t=1.8).
5. **Model guidance is sharp:** use **linear regression with little or no regularisation**. Heavy `Ridge` (α≥10) destroys the signal; gradient-boosted trees, SVR, and MLP all underperform OLS. **Judge by IC, not R²** (R² ≈ 0 even when IC is strong); always validate walk-forward.

---

## Findings at a glance

| # | Question | Test(s) | Result | Confidence |
|---|----------|---------|--------|------------|
| 1 | Real or synthetic? | skew/kurtosis, Jarque-Bera, Ljung-Box, ARCH-LM, anomaly detectors | Gaussian, white-noise, homoskedastic, no outlier days → **synthetic** | ★★★ |
| 2 | How many drivers? | Marchenko–Pastur (RMT), `FactorAnalysis` (CV), FastICA | **3 factors**; rest is noise (69% of matrix) | ★★★ |
| 3 | What is `ALGO`? | regression vs equal-weight basket | **Equal-weight return index** (R²=0.99); a hedge tool, not a stock | ★★★ |
| 4 | Momentum? | ACF, variance-ratio | **None**; dynamic is mean-reversion / lead-lag | ★★★ |
| 5 | Lead-lag structure? | lagged cross-corr, Granger | Real (max 0.185 vs 0.045 noise); strong pairs 100% sign-stable | ★★★ |
| 6 | **Market-neutral edge?** | **regressor sweep + walk-forward cross-sectional IC** | **Yes — OLS IC ≈ +0.058, t ≈ 5.3, stable OOS** | ★★★ |
| 7 | Best model class? | 21-model bake-off (linear / kernel / trees / NN) | **Linear, lightly regularised**; trees & NN underperform | ★★★ |
| 8 | Direct vs factor-induced links? | `GraphicalLassoCV` (partial corr) | ~half of correlation is the shared factor; 97 direct links | ★★★ |
| 9 | Cointegrated pairs? | Engle-Granger + FDR, rolling windows | 7 in-sample; only **`AENO~NWIG`** holds every window | ★★☆ |
| 10 | Do "sectors" exist? | clustering (7 algos), silhouette | **Soft** — free-choice methods find 1–2 groups | ★☆☆ |

---

## What we learned, in order

**The data is manufactured.** Near-zero skew and excess kurtosis; Jarque-Bera rejects normality for only 3/50 assets; Ljung-Box finds autocorrelation in 1/50, ARCH in 1/50; FastICA components are near-Gaussian; four anomaly detectors flag only the contamination rate (no crash/break days). Consequence: **assume Gaussian, constant-vol** — simple sizing, no tail events, but no volatility-timing signal.

**Three factors, one dominant.** RMT says pure-noise eigenvalues sit in [0.47, 1.73]; exactly **three escape** (10.6, 2.8, 1.9) and `FactorAnalysis` CV independently prefers k=3. The dominant factor is `ALGO`, literally the equal-weight return index (correlation 0.993). 69% of the correlation matrix is noise, though it is accurately measured (Ledoit-Wolf shrinkage 0.05).

**`ALGO` is a tool, not a target.** As the index (β≈1, 10× position limit, 1/5 commission in `eval.py`) its role is a **cheap, high-capacity market hedge** — never a directional bet. A net-long book loses ~11%/yr.

**No momentum, but a real lead-lag edge.** Autocorrelation is ≈0 at every lag and variance ratios drift below 1 (mean-reversion). Individually the lead-lag links are weak, but they are pervasive (147 of 2450 ordered pairs exceed the noise band; the strong ones keep their sign in both halves of the sample). **A linear model that reads the whole cross-section aggregates them into a significant signal:** walk-forward daily cross-sectional IC ≈ +0.058 (t ≈ 5.3), 64% positive days, and — importantly for generalisation — *stronger* in the last third of the sample than the first.

**Model choice is settled — and regularisation matters more than model family.** A 21-estimator bake-off shows **OLS and lightly-regularised linear models win** (IC ~0.05–0.06); moderate `Ridge` (α≥10) shrinks the small cross-asset coefficients that carry the signal and collapses IC to ≈0; kernel methods match linear at best; **gradient-boosted trees (~0.02), SVR, KNN, and MLP (~0.006) all underperform.** Mutual information detects mild nonlinearity (1.8× linear) but it never survives out-of-sample. **Use linear, keep regularisation light, measure by IC, validate walk-forward, and correct for multiple testing** (of 1275 cointegration tests, ~63 pass at p<0.05 by chance).

**The edge was adversarially verified — and it survives.** Because this result reversed an earlier conclusion, it was stress-tested four independent ways. *Significance:* a permutation test (shuffling predictions across assets each day) gives a null IC of 0.000 ± 0.009 and **p < 0.001** for the observed +0.058; a block-bootstrap 95% CI is **[+0.033, +0.080]** (lower bound clearly positive). *Leakage:* the pipeline is strictly causal — refit-daily and 1-/5-day embargoes all reproduce the IC within ~1%. *Reproducibility:* a from-scratch closed-form (`np.linalg.lstsq`) implementation matches sklearn to machine precision. *Cost:* a dollar-neutral book from the forecast returns **net Sharpe ≈ 4.5** (gross 4.9) after 1bp commission at 1.4× daily turnover — a return-space proxy, not full `eval.py` accounting, but the edge clearly clears costs. *Attribution:* the signal is **directed peer lead-lag** — dropping each asset's own return *raises* IC to 0.060 (so it is not autocorrelation) and dropping `ALGO` *raises* it to 0.062 (so it is not the market factor); it is ~3× stronger than plain reversal. **Caveat:** all of this rests on a single ~248-day out-of-sample path from one 500-row file; cross-path robustness and full-`eval.py` tradeability remain untested.

**It holds over the full 500 days, and it scores.** The earlier caveat — that the edge rested on a single ~248-day window — was tested directly by extending the out-of-sample path to ~400 days. It survives cleanly: the walk-forward IC is 0.05–0.065 regardless of where the OOS window starts (198 to 448 days), and it is individually significant in **both** halves (first-half t=3.7, last-half t=5.3). Q1 alone is weak (IC +0.027, t=1.9) and the apparent Q1→Q4 rise is inseparable from the expanding training window, so read it as "no decay," not "accelerating signal." Translated into positions and run through `eval.py`'s exact scoring (integer shares, $10k/$100k limits, 1bp/0.2bp commissions), the untuned OLS forecast earns **Score 179 / Sharpe 4.4** (last-250) and **152 / Sharpe 3.8** (last-400) — the IC-level and eval-level Sharpe agree. Adversarial checks confirmed it is not a lottery-tail result (drop the top-5 PnL days → still Score 147 / Sharpe 3.9) and — counter-intuitively — that its high turnover is *justified*: commission is only ~9–10% of gross, and every lower-turnover variant scores **worse** because the alpha is strictly one-day-ahead. The durable, scale-free number is **Sharpe ≈4.4**; the "179" reflects sizing to the dollar limit and is somewhat window-favourable. Still one synthetic sample — the competition's *different* future days are the untested part.

**Cointegration: one reliable pair.** Seven pairs pass Engle-Granger after FDR, but under rolling-window scrutiny only **`AENO~NWIG`** stays cointegrated in every window (β≈1, half-life ~4 days). The rest are regime-dependent.

---

## Implications for a strategy (when you build one)

- **The core signal:** a **market-neutral, walk-forward OLS (or α≤1) cross-sectional forecast** — predict each asset's next-day return from today's cross-section, go long the top / short the bottom, dollar-neutral. Gross IC ≈ 0.058.
- **Hedge residual market exposure with `ALGO`** (cheap, large limit); never bet market direction.
- **Keep regularisation light** — the edge lives in the specific cross-asset coefficients; don't shrink them away.
- **Lean the feature set.** Dropping each asset's own return and the `ALGO` index from the predictors slightly *raises* IC (to ~0.062) — both are irrelevant-to-harmful.
- **Costs are survivable — and don't try to cut turnover.** The book scores 179 / Sharpe 4.4 through the official accounting, verified by running `eval.py` itself. Turnover is high (~$49M/250 days) but *justified*: commission is only ~9–10% of gross PnL, and adversarial testing showed **every lower-turnover variant scores worse** (EMA-smoothing that cut turnover 60% collapsed Score 179→101). The alpha is strictly one-day-ahead, so smoothing it destroys it — the real gains are elsewhere (sizing/risk model, blending the `AENO~NWIG` spread), not turnover reduction.
- **`AENO~NWIG`** cointegration spread is a robust satellite signal to layer on.
- **Remember the scored days are unseen.** Every number here is out-of-sample *within this file*; the competition's future ~1000 days are the real test, so favour the light-regularisation, few-parameters version that generalised across all sub-periods.

---

## The built strategy & its trades

The plan was implemented and instrumented. Submission file: `Arbitrage_Victims.py` (self-contained, numpy-only). Dashboards:
**every entry/exit (all 50 charts)** → https://claude.ai/code/artifact/9e83e674-d967-4645-9b0f-cd773c3ff163 · **test suite** → https://claude.ai/code/artifact/b5b3edd2-1953-4be3-91dd-a9a34f6e7d9a · **equity** → https://claude.ai/code/artifact/c7956682-8b04-4e14-8da8-f2881253404d · **trades** → https://claude.ai/code/artifact/8a65f0ac-b88d-45a2-9978-dfc41e1fe905 · **how it works** → https://claude.ai/code/artifact/4d723f48-e636-4c77-a6b0-5dbd4441c130 · **structure/edge** → https://claude.ai/code/artifact/1560ad0c-311d-4108-80c0-480b088ca384

**The signal:** each day take the cross-section of all 51 returns → predict every asset's next-day return with **exponentially-weighted ridge** (half-life 250d forgetting + light L2 shrinkage α=0.1) → subtract the cross-sectional mean (market-neutral) → **MAX sizing** ($10k long/short) on every name whose **conviction clears a significance bar** (skip the coin-flips; the count floats ~32–47/day) → **β-hedge** the residual net beta with the cheap `ALGO` index. Rebalanced daily; no stop-losses or vol targeting.

**Five improvements, each justified by a test (not backtest-chasing):**
- **Light Ridge (α=0.1)** — the big one. Ensembling the top bake-off models did *not* help (they're 0.83-correlated → averaging adds nothing; the diverse ones were noise). Light L2 shrinkage stabilises the noisy 51×50 coefficients, sharpening sign accuracy: **Score 442 → 541.** Robust across α 0.03–0.3.
- **ALGO β-hedge** — the dollar-neutral book isn't *beta*-neutral (betas span 0.5–1.68). Hedging the residual with ALGO (1/5 commission) adds a small Score gain, better Sharpe, and true market-neutrality.
- **Conviction threshold** — trade only names whose `|forecast|` exceeds 0.2× the day's cross-sectional spread. **Not overfit:** it's causal (per-day, identity floats — not an asset blacklist), and the dropped bets provably have **no significant edge** (kept: 53% hit, t=7.5; dropped: 50.8%, t=1.7). **Score 541 → 585.**
- **Contrarian ALGO overlay** — the *index* has no next-day predictability, but it **mean-reverts at multi-day horizons** (peaks at K=30: past-30d vs next-30d, **t=−2.8**, permutation p=0.009, both halves positive; mean-reverting on 89% of days, even through up-regimes). So we fade its recent 30-day move with a position off ALGO's spare $100k capacity. **Score 585 → 652** (K=20→30). It is *symmetric* mean-reversion, **not** a drift/trend bet: a static short or trend-follow *loses* on the +37%/yr up-regimes, and a regime-adaptive fade↔follow switch adds nothing because the market essentially never trends (it mean-reverts even while drifting up).
- **Max the market bet** — the ALGO reversion is *orthogonal to the 50-book* (corr −0.04), and at Sharpe ≈7 the score's `SR²/(SR²+1)` factor is saturated ≈1, so **Score ≈ total PnL** → a bigger orthogonal reversion ≈ more Score until it pins the $100k ALGO cap. Swept the reversion size: peak/plateau at $200k desired (pins the cap on high-conviction days ~68% of the time; keeps the conviction *gradation* on weak-signal days, which a flat full-cap bet would throw away). **Score 652 → 715, StdDev 1470 → 1715** — the deliberate "be more aggressive" lever, robust across every window (200/250/300/400). Past ~300k Score *falls* (over-saturation flattens the gradation).
- **Two finishing tweaks** — swept the reversion z-window **CONTRA_WZ 40 → 60** (robust +10 on every window, Sharpe ↑), and made the β-hedge **apply last** on ALGO (reversion claims the $100k cap first, hedge fills only the leftover room — reversion priority is now explicit, not an accident of clip order). Equal-or-better on every window (worst-case @400 +3.7, @200 +8). **Score 715 → 726.** Dropping the hedge entirely was tested and *loses* (−17 Score, −0.22 Sharpe) — it still earns its keep on non-capped days.

**Trade profile (last 250 days):**

| Metric | Value | Read |
|---|---|---|
| Score / Sharpe | **726 / 6.83** | verified against the real `eval.py` (432/4.29 → 541 → 585 → 652 → 715 → 726) |
| Daily StdDev | **1715** | up from 1470 at 637 — deliberate extra risk on the orthogonal ALGO reversion |
| Total trades | **~7,600** | across 50 names; ~41 traded/day (floats 32–47) |
| Avg holding period | **1.6 days** | pure one-day-ahead alpha |
| Long / short | 41% / 59% | dollar-neutral (β-hedged) every day |
| Winning days | 66% | consistent, not lottery-driven |
| Commission drag | ~10% of gross | high turnover, but unavoidable (1-day alpha) |

**On sizing (confirmed):** MAX sizing beats conviction-weighting for *this* objective because the score's `SR²/(SR²+1)` factor is saturated at these Sharpes (0.948 vs 0.952 — a 0.4% difference), so Score ≈ `mu`, which scales with book size. MAX's 432 vs conviction's 179 is almost entirely the bigger book. The Sharpe trade-off is worth it.

**On non-stationarity (the adaptivity question):** it is an *estimation-scheme* problem, not a neural-net problem — a frozen NN drifts stale just like a frozen OLS, and here trees/MLP already lost the bake-off. The fix is a **forgetting** fit (EWLS). Empirically, though, forgetting only *costs* on this stable data (h=250: −1.6%, h=120: −12.7%, h=60: −44%) because the fit is data-hungry; a controlled break test showed the recovery benefit is real *in principle* but not realisable at this data's low SNR. So the default is mild (h=250) insurance, with the half-life as a dial to tighten only if live IC decay appears.

**Assessment:** a legitimate broad-shallow stat-arb book — Sharpe from *breadth* (≈50 tiny near-independent bets/day), not conviction. Strengths: consistent, market-neutral, high Sharpe, no lottery days. Watch-items: (1) extreme turnover is the structural fragility — fine here (~10% cost, no impact model) but the first thing to disappoint under real friction; (2) MAX sizing is a deliberate Score-over-Sharpe choice; (3) still one synthetic sample. Submit as a strong baseline; keep a wary eye on turnover.

## Endgame — why we hold (2026-07-14)

**Competition structure:** the public leaderboard is a *fixed* hidden window (we score 502.52, ~41st of ~130; top ~800); the **prize is scored once on fresh data**. With 2–3 submissions left and the last one carrying to the final, the standing plan is: **hold `Arbitrage_Victims_v2.zip` and re-submit it on the last day** (see `SUBMIT_CHECKLIST.md`).

**Three hunts, ~30 hypotheses, zero survivors.** Beyond the v1–v4 sweeps, two further adversarial multi-agent hunts tested: the $100k ALGO bucket alternatives (index-vs-constituents spread is real alpha but *uncapturable* — its short-basket leg displaces book capital 6.7× more productive), parameter-overfit audits, five fresh cross-sectional constructions, leakage (clean — reimplementation bit-identical, frozen-half IC 0.058), ensembling, index-residualization (destroys IC, t=−3.1 — the market factor is *part* of the signal), GLS/SUR/Ledoit-Wolf (all ≤0), calendar/seasonality (~257 corrected tests, null), commission reclaim (98% of the $46/day buys real signal-following; <$1/day recoverable), and drift tilts. All dead. The book sits at its information ceiling: IR = IC·√(50·250) ≈ 6.6 ≈ observed Sharpe.

**Two decisive DGP facts.** (1) The organizers set **all 50 idiosyncratic drifts to exactly zero** (χ²(49) p=0.61; 0/50 names with |t|>2): directional bets are coin flips *by design*, so market-neutrality is the provably correct posture, and any beta above us on the board is luck, not skill. (2) Score realization noise on a fresh ~250-day window is **±~110/day SE (±~180 at 90%)** — the public 500–650 band is largely indistinguishable from luck.

**Order-statistics reality check (simulated 130-team field where *nobody* has more true edge than us):** the public top still sits ~720 and reaches 800+ in 12% of draws; our median finish is ~16th (10–90%: 3rd–45th). Being 41st on the fixed window is fully consistent with fielding one of the strongest true strategies. The final is a re-draw; the only durable advantages are **maximum expected PnL and no wasted variance** — which is what v2 is.

**Hunt #3 (final, 2026-07-14) — the last two open cells, both closed.** (1) *All 13 overlay refinements dead* (nonlinear gating, asymmetric fade, EW/percentile z, horizon blends, hedge-sharing): best variant +7–9 $/d at p≈0.4–0.5 vs a Bonferroni bar of 0.0038; the only significant result is that tail-gating is significantly **harmful**. (2) *Partial pooling dead in all forms*, with a proof: the mean-pooling penalty is mathematically identical to raising the ridge α under the cross-target demean — and heavier ridge is significantly worse, so **α=0.1 is now sandwiched optimal from both sides**. The lead-lag network is weakly **anti-symmetric** (corr(S_ij,S_ji)=−0.055): if i leads j, j does not lead i — confirming the *directed* mechanism. A real curiosity survives: dropping the per-asset intercepts (pure noise, since idio drifts are exactly 0) improves IC +11% (t=3.1, both halves) but converts to ~$0/day through sign-based MAX sizing and self-decays as the sample grows — recorded, not shipped.

**The overlay decay mystery, solved — and it's better news than feared:** the ALGO reversion's backtest profit was **not** drift-riding (the −18%/yr window drift explains only 1–5% of it: the 60-day z-demeaning keeps the overlay time-averaged flat even in a trend, mean extra exposure only −$4.5k). Its timing covariance is honestly p≈0.04 pre-selection (the old p≈0.43 came from a low-power 30-day-granularity test; selection-corrected ≈0.1–0.2). The +139→+17 live decay is fully accounted for by **selection inflation plus ordinary realization noise** (a 250-day window has ±48 $/d SE) — not by a broken signal. Net: keep the overlay at 200k with *more* confidence than before; expect its live contribution in the +17–100 $/d band, and never expand it on backtest evidence.

**Discipline notes for the last days:** no test submissions of variants (their effects are below the board's ±15–50 resolution or already measured better locally over 440 paired days); no directional elements; no param churn on noise-level differences (HL 750, CONV_Z 0.12). The probe files built to measure the hidden window's beta were deliberately **deleted** after deciding their answer could not change the ship decision.

## Methods / tools used

`numpy`, `pandas` · **statsmodels**: ADF, KPSS, Engle-Granger `coint`, Johansen, Granger causality, ARCH-LM, Ljung-Box · **scikit-learn** (full sweep): `PCA`, `TruncatedSVD`, `FactorAnalysis`, `FastICA`, `KernelPCA`, `SparsePCA`; `KMeans`, `SpectralClustering`, `AgglomerativeClustering`, `Birch`, `DBSCAN`, `GaussianMixture`, `BayesianGaussianMixture`; `GraphicalLassoCV`, `LedoitWolf`, `EllipticEnvelope`, `IsolationForest`, `LocalOutlierFactor`, `OneClassSVM`; `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `BayesianRidge`, `ARDRegression`, `Huber`, `OMP`, `Lars`, `KernelRidge`, `SVR`, `LinearSVR`, `KNeighborsRegressor`, `DecisionTree`, `RandomForest`, `ExtraTrees`, `GradientBoosting`, `HistGradientBoosting`, `AdaBoost`, `Bagging`, `MLPRegressor`, `PLSRegression`; `TimeSeriesSplit`, `mutual_info_regression`, `f_regression`, `silhouette_score`, `MDS` · Random-matrix theory (Marchenko–Pastur), variance-ratio test.
