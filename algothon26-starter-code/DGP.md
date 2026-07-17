# Reverse-engineering the data generator

How `prices.txt` was most likely produced, and how to exploit it. Reproduce the
numbers with `python analyze.py` (structure) and `python simulate.py` (generator +
stress test). This is planning intel; the shipped strategy is `strategy.py` §3b.

## The inferred generator

A **one-factor market model with mostly-idiosyncratic names, and TWO tradeable
short-horizon reversions** — one on the market factor, one cross-sectional:

```
market_t     = market_{t-1} · exp(μ_m + drift)  ·  exp(−θ·(logm_{t-1} − MA₅))   # (A) index reverts
stock_k ret  ≈ β_k · market_return_t  +  idio_k_t        # β_k ≈ 1, one factor, ~20% R²
idio_k_t     = mean-reverts toward the CROSS-SECTIONAL average (relative-value OU)  # (B) idio reverts
ALGO (inst0) = the market factor itself (β=1, ~0 idiosyncratic)
```

- **Two edges, both real and both needed:** (A) the **index (ALGO) mean-reverts over
  ~5 days**, and (B) names that get rich/cheap **relative to the universe** revert over
  ~5–10 days. A random-walk market (θ=0) reproduces neither the ALGO edge nor the
  observed scores.
- The score is monetised by **sizing both edges near the dollar limits** (see below).

## Evidence (from analyze.py)

| Claim | Number |
| :--- | :--- |
| **ALGO = the index** | corr(ALGO return, equal-weight avg of other 50) ≈ **0.99**; identical total move |
| One factor, all positive beta | mean β on ALGO ≈ 0.98, frac>0 = 1.00; PC1 ≈ 21%, PC2 5.6%, PC3 3.9% (≈ noise → single factor) |
| Factor is ~20% of a name | mean R² to ALGO ≈ 0.20 → ~80% idiosyncratic |
| Prices random-walk-ish | log-price ADF stationary in only ~8% of names |
| **Index reversion is REAL** | ALGO-leg zrev(5) Score ≈ **134**; shuffled-returns (random-walk) null mean ≈ −20, **P(null ≥ observed) = 0%** |
| Cross-sectional reversion IC | −zscore signals IC ≈ **+0.02–0.03**, peaking ~5 days |
| No per-stock trend/revert bias | lag-1 autocorr ≈ 0; momentum negative |

**Why the earlier write-up said "ALGO is an untradeable random walk" — and why that was
wrong.** The overlapping-window k-day autocorrelation is biased negative (a genuine random
walk scores ≈ −0.13 on that metric), so it *looked* like noise. The clean test is the
tradeable **Score** vs a shuffled-returns null: there, the index edge is unambiguous
(P = 0%). Judge reversion by Score, never by that autocorrelation.

## Fingerprints the designers left

- **Instrument 0 = ALGO = the index**, and `eval.py` gives it a **10× position limit
  ($100k)** and **5× lower commission (0.2bp)**. That is a direct hint: trade the index
  leg hard. On its own it is the single biggest score contributor.
- Everything is scored on the **last 250 days**; the graded stage is *different* price
  data from the same generator — so exploit the **structure** (both reversions), not this
  sample's quirks.

## How to exploit it — the two-leg book (`strategy.py` §3b)

1. **SIZING is the dominant lever.** Score = mean·SR²/(SR²+1), and SR is scale-invariant,
   so Score scales ~**linearly** with deployed capital up to the dollar-limit clip. The
   old book used `SCALE=2` and deployed ~1/5 of the legal capital → Score ~105. Running
   both legs near their limits is most of the jump to ~300.
2. **Leg A — the index.** Trade ALGO on its own fast reversion (window ≈ 5) at ~full
   $100k. Cheap fees (0.2bp) + 10× capacity make it dominate.
3. **Leg B — cross-sectional relative value.** −zscore(≈10) on the 50 names, demeaned
   (≈market-neutral), each near its $10k limit. Use **raw** z-scores + cross-sectional
   demean — explicit beta-residualisation *hurts* (over-corrects; Score 148 → 7).
4. **Size for Sharpe, not growth.** The Score rewards consistency; full Kelly under-sizes.

## What NOT to waste time on (evidence-backed)

- **Pairs / cointegration stat-arb** — few names cointegrate; residual reversion is slow.
- **Momentum / trend-following** — universe momentum IC is negative.
- **Volatility-regime timing / HMM, residual/beta-neutral reversion, sector-relative
  demeaning** — no vol clustering, single factor (no sector blocks), and residualising
  the idio leg over-corrects. Simple beats clever on this generator.

## Simulator confirmation & stress test (`simulate.py`)

We fit the generator, **calibrate it to achievable Score** (not IC/autocorr — the OU
generator can't match both, because the real reversion is more *monetizable* than an OU
with the same IC), then stress-test on many independent synthetic panels.

- **Calibration.** `kappa ≈ 0.02` reproduces the observed idio-leg Score (~148); a market
  reversion `theta_m ≈ 0.2` reproduces the observed ALGO-leg Score (~134). `theta_m = 0`
  (random-walk index) gives an ALGO-leg Score ≈ 0 — confirming the index edge is imposed,
  not incidental.
- **Stress test (≥150 synthetic 250-day panels, calibrated generator):** the two-leg book
  has median Score ≈ **363**, mean ≈ **352**, 5th-percentile ≈ **+137**, **100%
  profitable** — dominating the single-signal book (~333) and the old `rev_blend` core
  (~221). Cutting the index leg (`algo_frac 0.6`) lowers *both* mean and the 5th
  percentile: the index leg is +EV and the idio leg diversifies it, so deploy it fully.
- **Real held-out check (the stronger evidence):** on the real file the config scores
  ~304 on the graded window, ~220 on a held-out earlier window, and is positive on every
  disjoint 100-day fold (min ~+87). Honest expectation on the graded stage: **~200–300
  with real variance** — the occasional ~400 leaderboard score is this edge on a
  favourable draw, not a floor.
