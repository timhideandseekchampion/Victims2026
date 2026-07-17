# Every variable in the strategy, scrutinised (2026-07-14)

Complete audit of `Arbitrage_Victims_v4.py`. Categorised by whether it's worth spending an
out-of-sample test on. Run the tests with `python validate_oos.py sweep` when real data lands.

## A. LIVE KNOBS — real, tunable, already in the OOS sweep

| variable | v4 value | grid tested | prior it beats v4 OOS | notes |
|---|---|---|---|---|
| `HALF_LIFE` | 2000 | 60 → ∞ | **MEDIUM** | memory length. Optimum drops only if data drifts → short HLs (60–250) double as the **non-stationarity detector**. |
| `ALPHA` | 0.1 | 0.01 → 0.5 | **MEDIUM** | ridge shrinkage. Optimum falls as data grows → the 0.01–0.05 candidates are bets on the more-data regime. |
| `CONTRA_DOLLARS` | 200k | 0 → 400k | **the keep/drop-overlay decision** | 0 = lean (book only). This is the single biggest structural fork; plateau 200–400k. |
| `CONV_Z` | 0.2 | 0.0 → 0.3 | LOW | book conviction bar; flat plateau 0.1–0.2. |
| `CONTRA_K` | 30 | 15 → 50 | LOW | reversion lookback; 30 is both-halves-stable (not overfit). |
| `CONTRA_WZ` | 60 | 40 → 100 | LOW | z-window; 60 is both-halves-stable. |
| `HEDGE` | True | on/off | LOW | β-neutralization; no-hedge loses ~17. |
| `contra_clip` | 3.0 | 1.5 → 5 | **ZERO (inert)** | proven inert — $200k saturates the $100k cap before the z-clip binds. |

## B. INERT / REDUNDANT / NO-OP — nothing to test (proven, not asserted)

| variable | why it can't matter |
|---|---|
| lam base `0.5` | redundant with `HALF_LIFE` — any base gives identical weights with a rescaled half-life (demonstrated). Testing it = testing `HALF_LIFE` twice. |
| warm-up `t < 60` | **no-op for every scored window** — the eval always calls with t ≥ 501, so it never triggers. |
| `contra_clip` (±3) | inert at current sizing (sweep = 0.0 effect across 1.5–5). |
| `eps = 1e-8` | numerical conditioning floor, ~10⁷× smaller than ALPHA — dominated. |
| CONTRA gate `+2` | just prevents an index error at series start; irrelevant mid-eval. |
| `age` (arange) | the time index (day k-ago is k days old) — a fact, not a parameter. |

## C. FIXED BY COMPETITION RULES — cannot test

| variable | value | why fixed |
|---|---|---|
| `LIMIT` | $10k | per-stock position cap set by the grader. Can't exceed; going lower just deploys less capital = lower score, so max is optimal. |
| `ALGO_LIMIT` | $100k | index cap (10×) set by the grader. Same logic. |

## D. STRUCTURAL CHOICES — already tested across 4 hunts, all dead (don't re-chase)

| choice (current) | alternatives tested | verdict |
|---|---|---|
| features = today's 51-asset 1-day cross-section | lag-2, PCA/factor, index-residualised, rank, sign-only, Granger/LASSO net, partial-corr | all worse |
| target = next-day return | 2-day, 1d+2d blend | worse / no add |
| estimator = linear EWLS ridge | trees, RF, KNN, MLP, kernel, ElasticNet | all worse (DGP is linear-Gaussian) |
| market-neutral demean | net-long/short tilt | coin flip (idio drifts set to exactly 0) |
| sizing = MAX sign-based $10k | magnitude-proportional, rank-proportional | worse (R²≈0, magnitude carries no info) |
| refit = daily | (finer impossible) | already maximal |
| returns = log | simple returns | ~identical at daily horizon |
| reversion = fade K-day move, z-scored | 2nd horizon, asymmetric, nonlinear gate, EW-z, signal blend | all dead (hunt #3) |
| hedge = full-window β, applied last | rolling/EW β, no-hedge, proportional sharing, hedge-first | current is best |
| intercept `my` kept | drop it (v3) | IC↑ but ~$0/day (sizing discards it), warm-up-only |
| GLS / Ledoit-Wolf error weighting | diag-GLS, SUR, LW/OAS α | all ≤ 0 |
| calendar / seasonality timing | day-of-week, spectral, PnL autocorr | no structure (~257 tests, null) |

## Bottom line
Genuinely worth an OOS test: **`HALF_LIFE`, `ALPHA`, and the overlay keep/drop (`CONTRA_DOLLARS`=0 vs 200k).**
The rest are either at robust optima (test for completeness/drift, low prior), inert, rules-fixed, or
structurally dead. The sweep covers all of A; run it on 501–750 (+751–1000) and demand |t|≥2 across
both windows before shipping anything over v4.
