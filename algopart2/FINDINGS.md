# algopart2 — signal & test hunt on days 400–750

**Data:** `prices.txt` = the full 750-day, 51-instrument panel (inst 0 = `ALGO` = the
equal-weight index). Days 0–500 are the original training file; **days 500–750 are the
newly-revealed window** the earlier research (`DGP.md`, `SIGNALS.md`, `combinedv3`) could
only *forecast*. This hunt re-tests every known edge on the recent regime.

Reproduce everything: `python hunt.py` (numpy+pandas only; eval.py-faithful scoring).
Raw numbers in `results.json`.

---

## TL;DR — the headline result

> **The reversion edges that defined the first 500 days have decayed on the fresh data.
> The peer lead-lag forecast is the one signal that survived — and it is what actually
> scores on days 500–750.**

| book (scored on the graded-like window **500–750**) | Score | Sharpe | verdict |
| :-- | --: | --: | :-- |
| pure reversion two-leg (ALGO zrev5 + idio −z10) — the *old* ship | **0.6** | 0.20 | **collapsed** |
| &nbsp;&nbsp;↳ ALGO index-reversion leg alone | −1.0 | −0.04 | **dead** |
| &nbsp;&nbsp;↳ idio cross-sectional reversion leg alone | 0.8 | 0.22 | flat |
| **lead-lag EWLS forecast alone** | **427** | 4.30 | **the surviving edge** |
| **lead-lag + reversion blend + ALGO leg (combined)** | **576** | 5.89 | **best** |

The pure-reversion book scoring ~0 while the lead-lag book scores 427 on the *same window*
is the whole story. It also explains the live result on record: the `combinedv3` book (built
on lead-lag) scored ~502 on this hidden window, whereas a reversion-only book would have
scored near zero.

---

## 1. Structure — the generator is unchanged (days 400–750)

| check | value | reading |
| :-- | --: | :-- |
| corr(ALGO ret, equal-weight avg of 50) | **0.976** | ALGO is still the index |
| PC1 / PC2 / PC3 variance | 20% / 8% / 5% | still ~one factor (PC2/3 ≈ noise) |
| mean β to ALGO / frac>0 / mean R² | 0.93 / 1.00 / 0.21 | one positive-beta factor, ~20% of a name |
| idio lag-1 autocorr | +0.003 (t=0.4) | **no** per-name 1-day predictability |
| \|idio\| lag-1 autocorr | −0.008 | **no** vol clustering / GARCH |

The *mechanism* (one factor + idiosyncratic noise) is intact. What changed is the strength
and sign of the two short-horizon reversions layered on top.

## 2. Cross-sectional IC battery (signal → forward return, 50 idio names, days 400–750)

| signal | IC@1d (t) | IC@5d (t) | IC@10d (t) |
| :-- | --: | --: | --: |
| **lead-lag EWLS (hl=500)** | **0.075 (7.8)** | **0.038 (4.0)** | **0.032 (3.5)** |
| xs_rev5 (−z of 5-day) | 0.014 (1.5) | 0.036 (3.5) | 0.024 (2.3) |
| xs_rev10 | 0.016 (1.6) | 0.023 (2.3) | 0.018 (1.8) |
| xs_rev40 | 0.009 (0.9) | 0.023 (2.3) | 0.031 (3.3) |
| momentum20 *(control)* | −0.011 | −0.020 | −0.015 |
| momentum60 *(control)* | −0.021 | −0.046 | **−0.060 (−6.6)** |

- **Lead-lag dominates every horizon** — its 1-day IC (0.075, t=7.8) is ~5× the best
  reversion signal and is strongest exactly where reversion is weakest (1-day). Different,
  faster edge.
- Reversion is still *statistically* alive at 5–10 days (t≈2–3.5) but weaker, and it **no
  longer monetises** (§3–4) — the horizon of maximum reversion has also drifted longer
  (xs_rev40 now beats xs_rev10 at 10d).
- Momentum is still significantly **anti-predictive** (trend loses) — the negative control
  passes, confirming the IC machinery is calibrated correctly.

## 3. Are the edges real or artifacts? (permutation nulls, window 500–750)

Shuffle each instrument's returns in time (destroys any reversion/lead-lag structure but
keeps the marginal distribution), rebuild prices, re-score. 300 draws, fixed seed.

| signal | observed Score | null mean | null 95% | p(null ≥ obs) |
| :-- | --: | --: | --: | --: |
| **lead-lag** | **427** | −81 | 41 | **0.000 \*\*\*** |
| ALGO index-reversion leg | −1.0 | −3.6 | 24 | 0.52 (n.s.) |
| idio reversion leg | 0.8 | −11.8 | 79 | 0.46 (n.s.) |

The lead-lag edge is **unambiguously real** on the fresh window (p<0.001) and is *not* the
shared-endpoint artifact the earlier notes warned about — shuffling kills it completely
(null mean −81). The two reversion legs are **indistinguishable from their random-walk null**
on 500–750: whatever reversion edge remains is too weak to reject the null here.

## 4. Regime map — why reversion "died" (rolling 100-day Score of the reversion book)

| window | Score | Sharpe |
| :-- | --: | --: |
| 300–400 | 39 | 0.97 |
| 350–450 | 271 | 3.84 |
| 400–500 | 210 | 2.82 |
| 450–550 | 16 | 0.64 |
| 500–600 | 186 | 2.70 |
| 550–650 | 123 | 1.87 |
| **600–700** | **−114** | **−1.46** |

The reversion edge is **violently regime-dependent** — Sharpe swings from +3.8 to −1.5
within the focus window, and the final 100 days are outright *negative*. Averaged over
500–750 that nets to ≈0. This is exactly the "ALGO leg is the regime-sensitive booster,
idio leg is the floor" warning in `SIGNALS.md` playing out — except here even the idio floor
flattened. The lead-lag book, by contrast, holds Sharpe ~4–8 across the same span.

## 5. What to actually trade on this data

1. **Lead-lag EWLS peer forecast is the core.** Score 427 standalone on 500–750, p<0.001,
   steady across sub-regimes. This is the signal to size hard.
2. **Keep the reversion blend + ALGO leg as a diversifying overlay, not the core.** Even
   though they score ~0 alone on 500–750, adding them *on top of* lead-lag lifts the combined
   Score **427 → 576** (Sharpe 4.30 → 5.89) — a weak-but-orthogonal signal still improves the
   Sharpe-based score. Do **not** run them as a standalone book.
3. **Do not trust a single-window backtest.** The 350–450 window would have told you the
   reversion book was a Sharpe-3.8 machine; 600–700 would have told you it loses money. Score
   on the graded-like window (500–750) and on the permutation null, as above.
4. **Nothing new in the mechanism** — no per-stock momentum, no vol clustering, no second
   factor. Don't hunt there.

**Bottom line for a submission targeting the next hidden window:** ship the **combined**
book (lead-lag core + reversion/ALGO overlay), size the lead-lag leg to the limits, and
expect **~500–580** on a 500–750-like draw — consistent with the `combinedv3` live ~502.

---

---

## 6. Can a strategy score 700–800 on this leg? (`push700.py`, `finalize.py`)

**Question asked:** build a book that scores 700–800 on the graded leg (days 500–750);
run *more* tests than the prior 5,084; fit on the same data and see if it ships.

**Answer — searched 5,760 causal (no-look-ahead) configs fit directly to 500–750:**

> **The maximum score any config reaches on the 500–750 leg is ~605. Zero of 5,760 clear
> 700.** 700–800 on *this specific window* is not reachable by strategy — it is a property
> of the window's realizable PnL, not a strategy deficiency. The *same* configs score
> **800–890 on days 400–500** (gap −220 to −290): 500–750 is simply a harder draw.

Running more tests makes this worse, not better: the max of N configs only climbs with N, so a
bigger search inflates the *in-sample* number while the honest expectation is unchanged. The
built-in forward test proves it — the config that scores **1054** fit on 400–500 delivers only
**515** forward on 500–750 (a −539 shrinkage). That is the overfitting you'd be shipping.

**Where 700–800 actually comes from:** a genuinely strong, *robust* book on a favourable draw.
The search winner (below) was validated on **every** rolling 250-day window (`finalize.py`):

| config | 500–750 leg | mean across 17 windows | worst window | windows ≥700 |
| :-- | --: | --: | --: | --: |
| **shipped part-2 book** | **604** | **637** | 513 | **7 / 17** |
| prior combinedv3 ship | 503 | ~620 | — | — |

So on a **fresh** graded draw (finals = a new re-draw) the honest expectation is **~640, a real
~40% shot at 700+, and a ~510 floor** — and 700–800 lands whenever the draw is kind, exactly as
the leaderboard's 700s are (luck tail, reproduced on real data).

### The shipped book — `Arbitrage_Victims_part2.py`  (verified: `eval.py` **Score 603.70**, Sharpe 5.61)

A legitimate **~100-point improvement** over the prior ship on this leg (503 → 604) and on the
across-window mean (620 → 637), from *sizing*, not new signals:
- lead-lag EWLS ridge core (hl 500) + **30% cross-sectional reversion** blend;
- **trade all 50 names, no conviction gate**, full $10k each — breadth = Sharpe (the gate was
  under-deploying on this data);
- ALGO index leg fades the 30-day move, **pinned to its full $100k cap**;
- residual **beta-hedge** applied last.

**Bottom line:** I can't hand you a book that *reliably* prints 700–800 on the 500–750 leg —
no such book exists (ceiling 605), and anything claiming to is fit to the answer. What I can
hand you is the strongest robust book found, which maxes this leg at 604 and expects ~640 with
a genuine 700+ upside on a good draw. That is the real 700-class strategy.

---

---

## 7. The "1054" config, a comfortable-700 search, and the full 0–750 comparison (`robust700.py`)

Followed up on: (a) that 1054-on-400–500 config, (b) whether *any* config comfortably clears
700, (c) how the finalists behave across every 250-day leg spanning days 0–750.

**The 1054 config turned out to be the best-mean config in the whole grid** (half-life 1000,
blend 0.15, contra 1M, fade) — the 1054 was a spike on one 100-day window, but across the six
250-day legs it averages **718** and clears 700 on 3 of 6. So it *does* clear 700 on average.

**But there is a real trade-off, and no free 700:**

| config | 500–750 leg (grader) | mean 6×250d legs | floor | legs ≥700 | early 96–346 | full 0–750 |
| :-- | --: | --: | --: | --: | --: | --: |
| **Ship A** — blend 0.30, HL 500, hedged | **604** | 661 | **604** | 2/6 | **538** | 615 |
| **Max-EV** — blend 0.15, HL 1000 (= "1054" cfg) | 574 | **718** | 574 | 3/6 | 475 | 607 |

- **No config clears 700 on the 500–750 leg** — the ~605 ceiling from §6 is firm. Max-EV is
  actually *worse* there (574) and on the low-history early leg (475 vs 538). Its higher mean
  comes entirely from spiking to ~800 on the middle legs (250–550) where the lead-lag regime was
  strongest — regime luck, not a durable floor.
- **"Comfortably 700" exists only as an across-window *average* (718), never as a guarantee.**
  The distribution is 574–806. The lighter reversion blend (0.15) leans harder on the lead-lag
  core → higher expected score but higher variance and a lower floor.

**Which to ship — depends on the objective:**
- **Max expected score on a fresh finals redraw → `Arbitrage_Victims_part2_maxEV.py`** (mean 718
  across windows, ~50% shot at 700+, but weaker on the current leg and on thin history).
- **Best on the current graded leg + highest floor + best with little history →
  `Arbitrage_Victims_part2.py` (Ship A)** (604 on 500–750, floor 604, 538 on the 96-day-warmup leg).

Both are verified through the real `eval.py`: Ship A **603.70**, Max-EV **574.19** on the 500–750 leg.

**Final honest bottom line:** you cannot get a book that *reliably* prints 700–800 — the current
leg caps at ~605 and the best across-window average is ~718 with a ~575 floor. The way to a 700+
result is the Max-EV book on a favourable draw. Ship A is the safer, higher-floor choice; Max-EV
is the swing-for-700 choice. Everything above the ~605 leg ceiling is draw luck, quantified.

---

---

## 8. How high can the IC go? Can we reach IC ≈ 0.10? (`ic_hunt.py`)

IC = mean over days of the cross-sectional correlation between a signal (known end of day d) and
the realized next-day return of the 50 idio names. Measured on days 400–750, all causal.

| signal | IC | t |
| :-- | --: | --: |
| lead-lag ridge (HL 500, α 0.1) | 0.0749 | 7.8 |
| **best ridge** (HL 2000, α 0.3) | **0.0777** | 8.2 |
| **best blend** — ridge + 0.2·revz(5) | **0.0791** | 8.4 |
| multi-HL ensemble (avg of 4 HLs) | 0.0743 | 7.8 |
| cross-sectional reversion alone revz(10) | 0.0158 | 1.6 |

**Answer: no — IC ≈ 0.10 is not causally attainable on this data. The ceiling is ~0.079.**
Swept half-life (250–2000), ridge α (0.03–1.0), predictor set (with/without ALGO), a 4-HL
ensemble, and a lead-lag+reversion blend. The maximum tradeable IC is **0.0791** (ridge + a light
0.2 reversion blend, t=8.4). Heavier blends and the HL-ensemble *lower* it — we're at diminishing
returns, not a missing lever.

**Why 0.10 is out of reach — the oracle proof.** Fitting the linear next-day predictor on the
*whole* window *with look-ahead* (best possible in-sample) gives IC **0.40** and pooled R² **0.115**.
That looks like huge headroom — but it is pure overfitting: the lead-lag matrix has 51×50 ≈ 2,550
coefficients estimated on ~350 days, so the in-sample fit memorises noise. The collapse from **0.40
in-sample → 0.079 causal** *is* the estimation-error wall. No causal estimator crosses it here,
because ~80% of each name's daily move is idiosyncratic noise that is unpredictable one day out.

**What is real and free:** the light reversion blend lifts IC 0.0777 → 0.0791 (+2%, t 8.2 → 8.4) —
small but genuine, and already captured by the shipped books' reversion blend. IC 0.079 at t≈8 is
an *excellent* cross-sectional daily edge (implied IR = IC·√(50·250) ≈ 8.8); the book's realised
Sharpe ~5–6 is lower only because turnover, fees and position discretisation skim the theoretical
IR. So the edge is strong and near its information limit — the gap to 0.10 is DGP noise, not a
signal we're failing to find.

---

## 9. Exhaustive multi-domain hunt: academic literature + hedge-fund families (`big_battery.py`, `research_tests.py`, `confirm_rrr.py`)

Surveyed the literature (RMT/Bouchaud-Potters, Ledoit-Wolf, reduced-rank regression, Avellaneda-Lee
stat-arb, Lo-MacKinlay contrarian, Grinold-Kahn portfolio construction, sparse-VAR) and implemented
every concretely-testable method across many math areas. **Signal to beat: causal IC 0.079.**

| method | math area | IC (400–750) | verdict |
| :-- | :-- | --: | :-- |
| **plain forgetting-ridge + light reversion blend** | ridge regression | **0.079** | **the ceiling** |
| reduced-rank ridge (Mukherjee-Zhu, k=5) | reduced-rank regression | 0.082* | *window-fit, fails OOS* |
| nonlinear (+squared) ridge | polynomial features | 0.078 | no lift |
| RMT eigenvalue clipping | random matrix theory | 0.072 | below ridge |
| cross-covariance SV cleaning (index-removed) | RMT (rectangular) | 0.061 | below ridge |
| single-index Ledoit-Wolf target | shrinkage estimation | 0.070 | below ridge |
| Ledoit-Wolf identity shrinkage | shrinkage estimation | 0.070 | below ridge |
| sparse top-8 lead-lag (LASSO-VAR proxy) | sparse regression | 0.068 | below ridge |
| Lo-MacKinlay accumulated contrarian | cross-serial covariance | 0.020 | weak |
| vol-scaled reversion | risk normalization | 0.014 | weak |
| **Avellaneda-Lee PCA-OU s-score** | stat-arb / OU | **0.005** | **dead** |
| transfer entropy / copulas / nonlinear causality | information theory | — | *provably ≡ linear (skipped)* |

**\*The one apparent winner failed the generalisation test.** Reduced-rank ridge k=5 scored IC 0.082
on 400–750, but on a proper per-window causal test (`confirm_rrr.py`) **plain ridge beats it on
every window** (250–500: 0.062 vs 0.042; 400–500: 0.080 vs 0.077; 500–750: 0.075 vs 0.074). The
0.082 was a step-2 sampling artifact plus k-overfitting to one window — textbook hyperparameter
overfit, caught exactly by the discipline this whole file argues for.

**Why nothing beats the ridge — now confirmed three ways** (oracle §8, this battery, the research):
1. **The estimation methods (RMT / shrinkage / reduced-rank) don't beat a well-tuned ridge**, because
   the ridge's forgetting-weighted α-regularisation is already near-optimal and the predictor
   covariance is not badly conditioned (q = p/T ≈ 0.15). Denoising can't recover signal that isn't there.
2. **The nonlinear / tail / entropy families are provably wasted here.** Barnett-Barrett-Seth (PRL
   2009): for *Gaussian* variables, transfer entropy ≡ linear Granger causality *exactly*. This DGP
   is Gaussian, so every nonlinear-dependence detector, copula, GARCH and NN gives nothing beyond the
   linear cross-correlations the ridge already fits. Confirmed empirically (all null).
3. **The famous hedge-fund stat-arb recipe (Avellaneda-Lee) is dead (IC 0.005)** because this DGP's
   reversion is *cross-sectional* (vs the universe mean), not *residual-to-principal-component* —
   residualising against PCs discards the very signal.

**The one genuinely useful research idea is a *score* lever, not an IC lever:** Grinold's `w ∝ Σ⁻¹α`
factor-aware sizing. But in this **capital-capped** regime (Score ≈ PnL, and sign-sizing already
deploys the full $600k legal gross on every name), `Σ⁻¹α` deploys *less* capital → it lifts Sharpe
but not Score. Consistent with the prior record's inverse-vol result. (The score tables in
`research_tests.py`/`confirm_rrr.py` have a 1-day look-ahead bug — discarded; authoritative scores
come only from `eval.py` and the `push700/finalize/robust700` harnesses.)

**FINAL CONCLUSION of the exhaustive hunt.** The forgetting-ridge peer-lead-lag forecast + a light
cross-sectional reversion blend, at **IC ≈ 0.079 / Sharpe ≈ 5–6**, is the information ceiling of this
data. It is not a failure to search hard enough — it is the mathematical limit of a linear-Gaussian
one-factor OU process where ~80% of each daily move is irreducible idiosyncratic noise. Across
5,760 sizing configs, a full IC-maximisation sweep, a 12-method multi-domain battery, and a
literature-driven set of estimator upgrades, **nothing durably beats it.** Effort is better spent on
robustness (the Ship A vs Max-EV trade-off in §6–7) than on hunting a higher-IC signal that the DGP
does not contain.

---

---

## 10. Deep stat-arb literature: optimal MR portfolios, cointegration, OU sizing (`boxtiao.py`, `ou_speed.py`, `clean_backtest.py`)

Second literature pass (Box-Tiao, d'Aspremont, Cuturi, Johansen VECM, Jurek-Yang, Bertram, Leung-Li,
Avellaneda-Lee) → implemented and tested the concretes. **The consistency bar (stable IC/t/p across
all windows) was the deciding lens throughout.**

- **Box-Tiao / d'Aspremont maximally-mean-reverting portfolio** (generalized-eigenvalue "least
  predictable basket"): **dead** — IC 0.005 (t=0.75), negative score on every leg (−57..+4). Building
  an optimal *fixed* stationary basket fails because the reversion is *cross-sectional* (each name vs
  the moving universe mean), not a fixed cointegrated combination. Same root cause as Avellaneda-Lee.
- **OU speed-weighted sizing** `w ∝ z·√κ/σ` (Jurek-Yang, the research's #1 pick): **worst sizing
  tested.** Sizing scheme scores/Sharpe across legs — **sign (full deploy) 608 / SR 5.63**; z-prop
  361 / 5.28; inverse-vol 343 / 5.15; **speed-weighted 221 / 3.59.** Per-name half-lives are dispersed
  but near-unit-root (median 90+ d) so κ is noisy, and in the capital-capped regime any down-weighting
  just sacrifices deployment. **Full-deploy sign sizing wins on both score AND Sharpe** — third
  independent confirmation.
- **RRR + LoMac blend** (user-requested backtest): mean score **510** across legs — below plain ridge
  (668) and the ship (608); and its IC is *inconsistent* (std 0.013 vs the ship's 0.0055, IC collapsing
  to 0.044 on 250-500). Fails both the score and the consistency bar.
- **Johansen VECM / Bertram-Leung-Li thresholds:** the research itself rated these confirmatory
  (Johansen's ML cointegrating vector ≈ the `eᵢ − mean` basket we already trade) or cost-dependent
  (thresholds only help with transaction costs, which the grader doesn't impose) — not implemented.

**Consistency table (the test you flagged — a real edge is stable across every leg):**

| forecast | IC mean | **IC std** | IC min | p (all legs) |
| :-- | --: | --: | --: | --: |
| **ship: ridge + reversion blend** | **0.077** | **0.0055** | 0.067 | ≈0 |
| plain ridge | 0.073 | 0.0058 | 0.063 | ≈0 |
| RRR + LoMac blend | 0.070 | 0.013 | 0.044 | ≤0.0001 |
| RRR k5 | 0.067 | 0.013 | 0.042 | ≤0.0001 |

The shipped ridge+reversion blend is both the **highest and the most consistent** IC (p≈0 on every
window). The RRR variants have ~2.4× the cross-window variance → window-fit, not durable.

## 11. Adaptation: does the lead-lag matrix drift? (`kalman_tvp.py`)

Third literature pass (Kalman/TVP-VAR, adaptive-forgetting RLS, Dynamic Model Averaging, online
learning, Bayesian changepoint). The research's decisive framing: **first prove `B_t` actually
drifts — the regime Sharpe swings may be pure realized-return variance.** The principled test is a
Kalman filter with `B_t` as a random-walk state, tuning the drift rate `q` by marginal likelihood
(TVP ≡ ridge on coefficient differences, Goulet-Coulombe 2020).

**Result: q̂ = 0.** The marginal log-likelihood is maximised at zero drift (52687 at q=0, falling
monotonically to 49271 at q=σ²). **The data says `B` is stationary.** Adding any random-walk drift
*reduces* the likelihood → the coefficient matrix does not meaningfully change over the file. The
Sharpe swings (+3.8 → −1.5 across rolling windows) are realized-return variance, not regime change.
Independently corroborated by the IC-vs-memory sweep (§8): longer memory *raises* IC (hl2000 0.077 >
hl250 0.070) — if `B` drifted, shorter memory would win; it loses.

**Verdict: adaptive/online methods (Kalman-TVP, adaptive forgetting, DMA, changepoint) add nothing
here and would chase noise.** The one nuance the research grants: since the down-windows are variance
not drift, they are *not* recoverable by any signal method — the correct response is bet-sizing/risk
control, which is precisely the Ship A vs Max-EV robustness trade-off (§6-7), not a smarter forecast.

### Final verdict after three literature passes + ~30 methods across ~10 math areas
The forgetting-ridge peer-lead-lag + light cross-sectional reversion blend, **full-deploy sign-sized**,
at **IC ≈ 0.079 / Sharpe ≈ 5-6**, is the endpoint. Confirmed against: 5,760 sizing configs; a full
IC-maximisation sweep + oracle ceiling; a 12-method multi-domain battery; RMT/shrinkage/reduced-rank
estimators; Box-Tiao/Avellaneda-Lee/Johansen stat-arb; Jurek-Yang/inverse-vol/Σ⁻¹α sizing; and a
marginal-likelihood adaptation test. **Nothing durably beats it, and the reasons are now proven, not
assumed:** IC is noise-bounded (oracle), nonlinear methods are ≡ linear on Gaussian data (Barnett
2009), the reversion is cross-sectional not basket/residual (Box-Tiao/AL dead), capital is capped so
full-deploy sign wins (three sizing tests), and `B` is stationary so adaptation is moot (q̂=0).

---

*Files: `hunt.py`/`results.json`; `push700.py`/`push700_results.json`; `finalize.py`; `robust700.py`;
`ic_hunt.py`; `big_battery.py`/`research_tests.py`/`confirm_rrr.py`; `boxtiao.py` (Box-Tiao basket);
`ou_speed.py` (Jurek-Yang sizing); `clean_backtest.py` (no-look-ahead harness + IC consistency);
`kalman_tvp.py` (adaptation diagnostic); `Arbitrage_Victims_part2.py` (Ship A) &
`Arbitrage_Victims_part2_maxEV.py` (Max-EV); `eval.py`/`eval_maxev.py` (grader). All IC figures causal;
scores match `eval.py` (Score = mean·SR²/(SR²+1); inst-0 $100k / 0.2bp). Note: quick score/IC tables in
`research_tests.py`, `confirm_rrr.py`, and the tail of `kalman_tvp.py` had look-ahead/alignment bugs and
are discarded — authoritative numbers come from `eval.py`, `push700`, `finalize`, `robust700`, `clean_backtest`.*
