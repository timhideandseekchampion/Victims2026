# Signal research — findings

What predicts returns in this universe, and what to build. Reproduce everything
with `python research.py` (and see it visually on the dashboard **Signals** tab).

## ⭐ BREAKTHROUGH — the two levers that were being missed (Score ~105 → 304)

The earlier research below concluded the ceiling was ~105 and that ALGO was an
untradeable random walk. **Both were wrong.** Two levers, confirmed on the real file
and on synthetic re-draws, take the official `eval.py` Score to **304**:

1. **SIZING is the dominant lever.** Score = mean·SR²/(SR²+1) and SR is
   scale-invariant, so Score scales ~**linearly** with deployed capital until positions
   clip at the dollar limits. The old book (`SCALE=2`) deployed ~1/5 of the legal
   capital. Running near the limits roughly **triples** Score by itself
   (`rev_blend` 69 → 126; `zrev(10)` → ~262). *Turnover/fees do not eat this* at these
   Sharpes — the fee-aware Score is 304.
2. **The ALGO index leg is a real, large edge.** ALGO (inst 0) mean-reverts over ~5
   days. Traded alone at its **$100k limit (10×) / 0.2bp fee (5×)** it scores ~134.
   Permutation null (shuffle ALGO's returns → kill any reversion): null mean ≈ −16,
   95%ile ≈ 79, **P(null ≥ observed) = 0%** — not a random-walk artifact. (The
   overlapping-window autocorrelation is biased negative and *looked* like noise; judge
   reversion by clean backtest **Score**, never by that autocorr.) The idio leg is
   similarly real (Score 148, ~95th pctile of its null).

**The shipped book — `strategy.py` §3b `two_leg`** — trades both, each on its own window,
near its limits:
- **Leg A (index):** ALGO reversion, window 5, at its $100k limit.
- **Leg B (idio):** −zscore(10) on the 50 names, cross-sectionally demeaned, each near
  $10k. Use **raw** z + demean; residual/beta-neutralising the idio leg *hurts*
  (Score 148 → 7) and there are no sector blocks (PC2/PC3 ≈ noise) to demean against.

**Verification (eval.py + backtester):** Score **304.12**, Sharpe **2.51**, Sortino 4.08,
Calmar 5.75, maxDD $15.3k. Monte Carlo (2000): **99.3% profitable**, Score 5% 49 / median
299. Walk-forward 5×50d: 149 / 574 / 10 / 684 / 102 — **all folds positive**. Held-out
earlier window ~220; disjoint 100-day folds all positive (min ~+87).

**Exhaustive sweep (108 configs, `simulate.py`-faithful panels + real windows).** All top
configs use `algo_frac=1.0` (deploy the index leg fully — cutting it lowers *both* mean
and the 5th-pctile) and `algo_w=5`. The chosen `idio_w=10, algo_scale/idio_scale≈0.10,
fraction` had the best real-data profile (top of a tied cluster):

| idio_w | algo_w | scale | algo_frac | sizing | synth μ | synth 5% | real graded | real held-out | worst real fold |
| ---: | ---: | ---: | ---: | :-- | ---: | ---: | ---: | ---: | ---: |
| 10 | 5 | 0.10 | 1.0 | fraction | 320 | 94 | **304** | 220 | **+87** |
| 10 | 5 | 0.08 | 1.0 | fraction | 320 | 97 | 303 | 219 | +89 |
| 10 | 5 | 0.10 | 1.0 | inverse_vol | 321 | 107 | 289 | 206 | +70 |
| 12 | 5 | 0.12 | 1.0 | inverse_vol | 332 | 116 | 251 | 229 | +32 |

**Honest expectation — the 304 is the FAVOURABLE end, not a floor.** The grader scores the
**last 250 days with a full 250-day warmup** (`startDay = nt−250`; `getMyPosition` always
sees ≥250 days) → 304 on this file. A full-500-day backtest gives only **~198** — partly a
warmup artifact (early days scored with little history; grader never sees them), but mostly
because the edge is **regime-dependent** and the early period was weaker. Rolling 100-day
windows (each with full warmup) climb monotonically: days 101–200 Score **87**, 151–250 125,
201–300 231, 251–350 365, 301–400 281, 351–450 342, 401–500 **402**. Every window is positive
(worst +87), so the edge is real throughout, but the market-reversion regime improved over
time. Leg split confirms resilience: the **idio leg is the steady floor** (≈+$25k/half), the
**ALGO index leg is the regime-sensitive booster**. So on a fresh graded draw expect **~150–300
central, ~90–160 in a weak regime, ~400 in a good one** (MC 5%ile ~49–137). The occasional ~400
leaderboard score is this edge on a favourable draw, not a floor. `simulate.py` must be calibrated to
achievable **Score** (kappa≈0.02, theta_m≈0.2) — an IC-calibrated or random-walk-market
generator hides the index edge, which is what capped the earlier analysis at ~105.

### Full-500-day significance re-run (statistical power over all history)

Re-ran everything scored over the full ~500 days (warmup 50) with significance tests. The
two edges are real; nothing new is:

- **Shipped two-leg: Sharpe t = 3.15 (***).** idio leg t=2.75, ALGO leg t=2.02. Permutation
  nulls (shuffle returns): idio-leg **p=0.010**, ALGO-leg **p=0.007** — both real at 1%.
- **IC t-stats (full history, 1-day):** `−zscore(10/20/40/60)` t = 2.65 / 2.29 / 2.10 / 2.14
  (all sig); `momentum(20)` **t=−2.62** (significantly anti-predictive — trend loses money).
- **No new tradeable structure:** rolling no-lookahead eigenportfolios PC2–PC5 t = 0.11 / 0.14
  / −1.46 / −0.79 (one-factor confirmed); ACF(r²)≈0 (no GARCH / vol-timing); flat FFT (no
  calendar/cycle); no single index lag breaks 2σ (reversion is diffuse across ~2–8d lags);
  lead-lag `catchup` IC-series is +0.82 correlated with reversion (same edge); short-side
  asymmetry is not monetizable (neutral on real & synthetic).
- **Longer-window "zone" reversion (w40/60)** is significant (t≈2.1) but collinear with w10.
  A blend looks better *on the synthetic* (247→345) yet worse *on real* (270→219). **Caveat:**
  `simulate.py`'s idio OU (κ≈0.02, ~35d half-life) reverts SLOWER than reality (~5–10d), so it
  over-rewards long windows — trust real data on horizon choice. w10 wins at every real horizon.

**Conclusion:** the two-leg book captures all statistically significant structure; no change.

### Push-higher experiments — measured, none robustly beat the base (all rejected)

We're already deploying ~$583k of the ~$600k max legal gross, so more Score must come from
higher PnL-per-dollar (a better/extra edge), not more capital. Every idea below was judged
on real disjoint windows AND ≥120 synthetic re-draws; **none survived**:

| Idea | Result | Verdict |
| :--- | :--- | :--- |
| **ALGO trend-gate** (cut the index bet when the index is trending) | +4 graded on *this* window, but on 120 synthetic panels it *lowers* both mean (360→~310) and 5th-pctile (155→~110) | ✗ overfit to this sample's trending episodes — the real fold-min improvement was 4-sample noise |
| **idio window blend** (3/10, 5/10, 3/10/20, 10/20) | every blend < single w10 (304) — best was 10/20 at 290 | ✗ the idio edge is fast; mixing windows adds noise/turnover |
| **ALGO window** variants (3, 7, blends) | all ≤ base zrev5 | ✗ w5 is the sweet spot |
| **secondary factors / eigenportfolios** (PC2–PC6, contrarian) | ~0 or negative *even with full-history look-ahead* (an upper bound) | ✗ no hidden factor / cointegrated-basket structure — genuinely one-factor |
| **lead-lag ALGO↔stocks** (`catchup`, clean backtest) | ~0 (graded −4…+2) | ✗ the apparent lead-lag is the shared-endpoint artifact |
| **deploy FULL cash** (sign-only / idio_scale→0.03) | synth μ ~316 either way, 5%ile 125→~137; graded ~flat | ≈ neutral — we already max the high-conviction names; the spare ~$17k goes to near-zero-signal names that add ~equal edge and risk |
| **skip 'random'/low-conviction charts** (keep top 80/60/40%, or drop hi/lo-vol) | every subset worse: synth μ 316→295/256/194 | ✗ the edge is cross-sectional BREADTH — dropping names kills diversification (Sharpe). 28/50 names were net-positive this window but you can't know which ex-ante (homogeneous process) |
| **trend-follow the 'trending' charts** (momentum on names with +autocorr or strong trend; per-stock trend/revert switch) | held-out −178, synth μ 66 (vs 253); pure momentum synth μ −153 | ✗ the visible trends are the shared MARKET factor (fell ~14%) + idio noise, not per-stock alpha — per-stock daily autocorr ≈ 0 (coin-flip). The idio leg already demeans out the market drift; the market's own move is captured (short-horizon) by the ALGO leg |

**Conclusion:** `two_leg(idio_w=10, algo_w=5, full limits)` sits at the efficient frontier of
this process. The ~500–680 seen in individual 50-day walk-forward folds are favourable
sub-period draws, not a sustainable level. Inspect/replay any of these via
`backtester.py --two-leg [--idio-w/--algo-w/--idio-scale/--algo-scale/--algo-frac/--idio-sizing]`.

---

The sections below are the **earlier, gentle-sizing research** (SCALE=2). The *relative*
signal quality (reversion beats momentum, fast beats slow, blend beats single window)
still holds; the absolute Scores are superseded by full-limit sizing above.

## The data (what predicts, and what doesn't)

Cross-sectional **Information Coefficient** (IC) = the daily correlation between a
signal's score and the *forward* return, averaged over days. Small but positive &
stable IC = a real edge.

| Finding | Evidence |
| :--- | :--- |
| **Multi-day mean reversion is the edge** | `−zscore` signals have IC **+0.02–0.03**, peaking at a **~5-day horizon**, decaying to ~0 by ~12 days. |
| **1-day effects are dead** | return lag-1 autocorr ≈ 0.000; `ret_{t-1}→ret_t` IC ≈ −0.001. |
| **Momentum is anti-predictive** | `momentum(60)` IC ≈ **−0.02 to −0.05** (worse at longer horizons); baseline Score −13.5. |
| **Normalisation matters** | raw past-return reversion is weak (IC ≈ −0.01); vol-normalised (z-score) is the real signal. |
| **No volatility clustering** | `|ret|` autocorr ≈ −0.007 → vol-*timing* has little edge; vol-*scaling* still helps risk. |
| **~0.20 common factor, no tight pairs** | max pairwise corr 0.59, zero pairs >0.6 → neutralise the market; pairs trading won't find much. |

The **quantile chart** (dashboard) makes it visual: bucket instruments by signal
each day, average their market-neutral forward return. For reversion it slopes up
(cheap names outperform); for momentum it slopes down (recent winners revert).

## Signal ranking (IC over last ~250 days, from research.py)

| signal | IC 1d | IC 5d | IC 10d | note |
| :--- | ---: | ---: | ---: | :--- |
| xs_rev5 (rank of 5d return) | +0.017 | **+0.033** | +0.023 | best raw IC; most *different* from the z-scores → diversifier |
| rev_z10 | +0.021 | +0.033 | +0.020 | short lookback |
| rev_rankz (rank of z20) | +0.019 | +0.033 | +0.017 | outlier-robust |
| rev_z20 (the old baseline) | +0.019 | +0.032 | +0.016 | |
| rev_blend (z10/20/40 avg) | +0.019 | +0.032 | ~ | diversified across horizons |
| rev_z40 / z30 / z60 | +0.017 | +0.02–0.025 | | slower |
| momentum60 | −0.014 | −0.025 | −0.030 | negative control (trend fails here) |

The reversion z-scores are highly correlated (0.6–0.97); **xs_rev5** is the most
independent (0.47–0.76), so it's the best candidate to *blend* for diversification.

## Net-of-fees backtest (last 250 days, from research.py)

This is where turnover/fees bite — IC is necessary but not sufficient.

| config | Score | Sharpe | Sortino | maxDD | turn/day |
| :--- | ---: | ---: | ---: | ---: | ---: |
| rev20 baseline | 57.2 | 1.28 | 1.89 | 15.9k | 124k |
| **blend 10/20/40** | **69.0** | 1.48 | 2.22 | 13.1k | 128k |
| blend + smooth5 + inverse_vol | 53.9 | **1.58** | **2.44** | **7.0k** | **54k** |
| rev20 inverse_vol | 41.3 | 1.33 | 1.97 | 7.8k | 95k |
| rev20 smooth5 | 28.8 | 0.96 | 1.38 | 15.8k | 58k |
| rev20 hold5 | 34.8 | 0.99 | 1.46 | 19.3k | 51k |

**Two winners, two purposes:**
- **`blend 10/20/40`** — best raw Score (69.0) and best walk-forward (mean 85.8,
  worst fold −40 vs the baseline's −138). Diversifying across horizons is the
  single biggest, most robust improvement.
- **`blend + smooth5 + inverse_vol`** — best *risk-adjusted* profile: Sharpe 1.58,
  Sortino 2.44, **half the drawdown and half the turnover**. Lower headline Score,
  much steadier.

## Surprise (why we backtest, not just eyeball IC)

**Turnover reduction alone HURT Score** (smooth5 → 28.8, hold5 → 34.8). Prior
hypothesis was that cutting turnover would help net of fees; it didn't, because the
reversion edge is *fast* — delaying entries misses the bounce, and that costs more
than the fees saved. Turnover reduction only paid off when **combined with
vol-scaling** (which lifts Sharpe enough to matter). Lesson: IC and net Score are
different questions; always confirm in the backtester.

## Recommended direction

1. Start from **`rev_blend`** (already in `strategy.py`; set `alpha` to it, or
   `ACTIVE`-style). It's the current best and robust.
2. Try adding **`inverse_vol` sizing** for a steadier book (better Sharpe/DD).
3. Explore **blending in `xs_rev5`** — it's the most independent signal, so a
   `rev_blend + xs_rev5` combination may add diversification (test it: does the
   *combined* IC / Score beat either alone, out of sample?).
4. Judge everything on **`research.py --walk-forward`**, not one window.

## Adaptivity (EWMA z-normalization) — tested, does NOT win here

Question: should the signal *adapt* over time (a threshold that moves)? First, the
data says there's little to adapt to on this sample: strategy vol doesn't cluster
(autocorr of book r² ≈ −0.002), cross-sectional dispersion is flat (1.99%→2.12%),
and the reversion edge is stable across periods (IC@5 +0.019 / +0.016 / +0.045).
Also, the z-score is *already* adaptive in the good, parameter-free way — it
standardises each name by its own rolling mean/std.

We built the lowest-overfit adaptive variant anyway: **EWMA z-normalization**
(`ewma_z`, `zrev_ewma`, `alpha_rev_eblend`; signals `rev_ez10/20/40`, `rev_eblend`),
and measured it head-to-head. Results:

- **IC:** adaptive is marginally *better* and decays slower — `rev_eblend` IC5 +0.034
  vs `rev_blend` +0.032, IC10 +0.023 vs +0.018.
- **Net Score (the thing that's graded):** adaptive is *worse* at every matched
  aggression. At ~equal turnover (127k): static blend **69.0** vs EWMA **62.1**.
  (scale 1.5 → 93.7 vs 62.1; scale 1.0 → 106.6 vs 89.0.)
- **Robustness:** EWMA's one edge is a steadier worst walk-forward fold
  (−14.5 vs −40.5 at matched turnover) — its smoother positions bleed less in bad
  stretches, but give up Score to do it.

**Why:** EWMA smooths the signal, so it trades a bit less and misses part of the
*fast* reversion that (on this data) pays — the same lesson as the turnover-reduction
result. Marginally better prediction (IC) ≠ better monetisation (Score).

**Decision:** static normalisation stays the default. The EWMA signals stay
registered and inspectable (dashboard Signals tab, `research.py`) so they can be
re-checked on the graded price stage, where clustering/regime structure may differ.
Rule stands: ship adaptive only if it beats static out-of-sample.

## Skeptical experiments — measured (research.py), net Score + walk-forward

Built as measured experiments (data argued against most). Net Score / walk-forward
mean / worst fold, vs the static `blend10/20/40` (Score 69.0, WF mean 85.8, min −40.5):

| Experiment | Score | WF mean | WF min | Verdict |
| :--- | ---: | ---: | ---: | :--- |
| **blend + regime-gate** (GMM on ALGO) | **73.8** | 80.8 | **−6.1** | **genuine OOS tail-protection** — cuts the worst fold from −40 to −6; slightly lower mean. The one experiment that earned its keep. |
| blend + Kelly (inverse-variance) size | 30.6 | 33.8 | −29.5 | best Sharpe (1.77) & tiny drawdown, but under-sizes → low Score. Kelly optimises growth, not the Sharpe-based Score. |
| combine blend + xs_rev5 (0.7/0.3) | 50.4 | 65.2 | −56.6 | worse than blend alone — xs_rev5 diluted it here. |
| trend/revert mixed (per-stock) | 16.7 | 30.2 | −72.1 | worst — confirms per-stock momentum is noise. |

**Surprise:** regime-gating helped *out-of-sample* despite no vol-clustering in the
autocorrelation — it de-risks in the market's high-variance state and that state
lines up with the bad reversion period (fold f300). Real, but modest and adds model
risk (a GMM refit each day). **Decision:** static blend stays the default; regime-gate
is available (`strategy.regime_gate`) as a robustness option to consider, not an
auto-ship. Everything else stays as measured negatives on record.

Combine tooling: `strategy.combine_signals` / `combine_positions`, and
`backtester.py --combine modA,modB --weights ...` for a book-level combination.

## EV / confidence / regime-scale overlays — measured, none beat blend+invvol

Built as stateless overlays (`strategy.cost_gate`, `confidence_scale`,
`regime_scale`; backtester flags `--ev-gate`, `--confidence`, `--regime-scale`).
vs `blend + inverse_vol` (Score 74, WF mean 77, min +0.8):

| Overlay | Score | WF mean | WF min | Verdict |
| :--- | ---: | ---: | ---: | :--- |
| ev-gate (keep top 60% conviction) | ~16–40 | 48 | −20 | ✗ hurts — the cross-sectional book wants breadth; dropping marginal names loses diversification |
| confidence (scale by recent IC) | ~21 | 37 | −28 | ✗ hurts — de-risks at the wrong times on this data |
| regime-scale (continuous GMM) | ~59 | 58 | +20 | slightly worse than the binary regime-gate (73.8); decent downside but lower mean |

So the EV/confidence/gate ideas don't add net Score here — consistent with the
theme that simple beats clever on this generator.

**ML dropped (recorded decision):** a Ridge and a gradient-boosted model both tie
the blend out-of-sample (IC +0.031); GBM overfits (train 0.41 → test 0.03). No
ML alpha pipeline — the edge is near-linear cross-sectional reversion.

**Simulator finding (see DGP.md):** across 80 synthetic paths, plain `rev_blend`
is profitable 96% of the time; `inverse_vol`'s historical win is NOT clearly
robust out-of-process. Leans toward `rev_blend` (or +regime-gate for tails) as the
most defensible submission core.

## Deliberately out of scope (recorded decisions)

Factor/PCA neutralisation, pairs trading, vol-regime timing, and calendar effects
were **not** pursued: the data shows little/no edge (no vol clustering, no tight
pairs, momentum negative) and they were de-prioritised. Revisit only if a new data
stage changes the picture.

## Tools

- `python research.py [--walk-forward]` — IC report, signal correlation, net
  backtest report. Edit the `configs`/catalog to test your own ideas.
- Dashboard **Signals** tab — IC by horizon, IC decay, and the market-neutral
  quantile chart, per signal, interactively.
- Add a new idea as a function in `strategy.py` §3 and register it in `SIGNALS`;
  it then appears in both `research.py` and the dashboard.
