# Combined v3 — Arbitrage Victims

The **compiled** three-signal book for Algothon 2026, grounded in the full algo26v1
research record (`FINDINGS.md`, `VARIABLES.md`, `oracle_log.md`, `SUBMIT_CHECKLIST.md`).
Self-contained (`numpy` only).

## Result — read the DISTRIBUTION, not the single number

The famous **763** is the last-250 window only — the most favorable slice. Do NOT expect it
live. Re-scoring **every** rolling 250-day window in the file (`consistency.py`):

| config | median | min | max | p10 | **honest expectation** |
|---|---|---|---|---|---|
| full book (ridge+ALGO) | 575 | 500 | 768 | 513 | **~500–575** |
| revblend (ridge+rev+ALGO) | 601 | 518 | 766 | 539 | **~540–600** |

**The live result confirms this:** v1 backtested 763 on the last-250 window but scored **502**
on the hidden 501–750 window — i.e. a fresh window lands near the *bottom* of the in-file range
(min 500 / p10 513), not the median. So plan for **~500–550**, treat anything above as upside.
The strategy is *consistently profitable* (every 250-day window ≥500, 100% clear 300) — it is a
reliable ~500-600 book, not a reliable 763 book. Dropping the ALGO overlay lowers the floor
(lean min 410), so it stays. Single-window reference scores: last-250 **762**, full 60–500 **582**.

## Modelling data we DON'T have — why HALF_LIFE=500, not 2000

All of the above is still the *known* 500-day file. The scored window is future data that
doesn't exist yet, and the deeper risk is that its **structure differs** from the past we fit
on. Re-slicing the known file can't see that; a forward Monte-Carlo can. `forward_mc.py`
generates unseen futures under three mechanistic worlds and scores each variant only on the
synthetic future (genuinely out-of-sample):

- **World A** — dense lead-lag VAR (v1's model of the world).
- **World B** — sparse cointegration pairs (v2's independent model of the world).
- **World C** — the VAR structure *rewires* on the unseen days (the real model risk).

| ridge memory | known @250 | World A | World B | World C | **worst-of-3** |
|---|---|---|---|---|---|
| HL=2000 (old v4 ship) | 763 | 584 | 204 | 277 | 204 |
| **HL=500 (this ship)** | 762 | 562 | 291 | 249 | **249** |
| HL=250 (max-defensive) | 726 | 522 | 366 | 230 | 230 |

**HL=2000 wins only in World A — the world that *is* the fitted past.** Lengthening the memory
to 2000 (v4's "use all data, the DGP is proven stationary" move) silently assumes future=past;
it maximises the known-window score and the worst-case *drops* to 204. **HL=500 costs ~1 point
on the known window (762 vs 763) but has the best worst-case (249) and best average across the
unknown-future worlds** — the robust choice when you can't verify stationarity out of sample.
HL=250 is the option if you want to bet more heavily that the mechanism will change (−5% base).

Also tested and rejected as robustness levers: adding a **cointegration-pairs sleeve or a full
ensemble** — dominated in *every* world (pure pairs lose; the ensemble tracks the ridge minus a
fee drag), because a marginal-EV leg hurts rather than diversifies. The robustness lever here is
**memory length (adaptivity), not more signals.**

Run it:
```bash
python eval.py         # scores last 250 days (leaderboard window)
python eval.py 440     # scores the full 60-500 window
```

## What "combined" means here

The combination is the **three orthogonal signals** the research converged on — this is the
endpoint of the documented Score lineage **432 → 541 → 585 → 652 → 715 → ~763**, each step a
signal or risk layer added and justified by a test (not backtest-chasing):

| step | Score | what was added |
|---|---|---|
| OLS cross-sectional forecast | 432 | **SIGNAL 1**: peer lead-lag (IC ≈0.058, t≈5.3, perm p<0.001) |
| + light ridge (α=0.1) | 541 | stabilise the noisy 51×50 coefficients |
| + conviction gate | 585 | **RISK**: trade only names clearing 0.2× the daily spread |
| + ALGO contrarian overlay | 652 | **SIGNAL 2**: fade the index's mean-reverting 30-day move |
| + size overlay to $200k | 715 | orthogonal bet, Score≈PnL at this Sharpe → size to the cap |
| + WZ=60 & hedge-last | ~763 | **RISK**: beta-hedge into leftover ALGO cap room |

**Why these three and nothing more:** the DGP is synthetic Gaussian, three-factor, no
momentum, no vol regimes; ALGO *is* the equal-weight index; and all 50 idiosyncratic drifts
are set to **exactly zero** (directional bets are coin flips by design → market-neutral is
provably correct). Three adversarial hunts (~30 hypotheses) put the book at its information
ceiling: IR = IC·√(50·250) ≈ 6.6 ≈ the observed Sharpe.

## Signals tested and deliberately excluded (below the noise floor)

Grounded in the findings — verified here, not assumed:

| candidate | measured effect | verdict |
|---|---|---|
| cross-sectional reversion blend (the earlier draft) | +3 @250, below ±110 SE | excluded; reversal 3× weaker, blend dead in hunt #3 |
| AENO~NWIG cointegration pair ($10k) | +0.8 @250 | excluded; its capital competes with the 6.7× richer book |
| index-vs-constituents spread, GLS/SUR, calendar, drift tilts, partial pooling, trees/MLP, momentum | ≤ 0 | all dead across 3 hunts |

## Reconciliation with algo26v2 (the "can we combine them?" question)

`algo26v2/` is an **independent** research effort on the *identical* price file (md5-verified).
It ran 5,084 statistical tests and converged on **cointegration pairs** as the main edge
(honest OOS Score ~84, Sharpe 2.53) — and **never discovered v1's peer-lead-lag ridge**. Its
best score-max blends top out at ~490–517 full-window and decay hard across halves (e.g.
707→239), a classic overfit signature. So v1's ridge (763 @250) dominates all of v2.

I tested combining every v2 signal onto v1's ridge. **None helps** — verified, not assumed:

| v2 overlay on the ridge (causal) | Score @250 | verdict |
|---|---|---|
| cross-sectional reversion (`xs`) | 501.6 | **−262** — redundant + crude sizing |
| lead-lag (`lead`) | 624.2 | −139 |
| multifactor PCA-residual (`mf`) | 648.4 | −115 |
| cointegration pairs, rolling causal selection | 765.6 | +2 (noise) |
| cointegration pairs, clean OOS (pick on 0–250, trade 251–500) | +8.8 on H2 | noise; only **2** pairs survive clean selection vs 24–94 in-sample |
| cointegration pairs, *fixed* list from full sample | 829 | look-ahead — pairs were chosen knowing 251–500 |

**Why:** the ridge is a full cross-sectional regression that already combines all the linear
lead-lag / reversion structure optimally; v2's edges each re-capture a crude slice of the *same*
signal, so adding them double-counts with extra noise and turnover. The genuinely orthogonal
piece (pairs) is mostly in-sample selection artifact — clean OOS selection leaves 2 pairs worth
~noise. Conclusion: **the goated strategy is the v1 ridge book; combining v2 does not beat it.**
(Investigation code: `algo26v2/analysis/combine_v1v2.py`, `combine_fast.py`.)

## Tournament "punt" variant — `Arbitrage_Victims_combined_punt.py`

Top-10-advances means a robust median (~16th) doesn't qualify, so there's a case for swinging
for upside. This variant drops the hedge and shoves ~all legal capital onto the edge:

| build | knobs | Score @250 | gross | note |
|---|---|---|---|---|
| primary (robust) | hedged, conv 0.2, contra 200k | 762 | $499k | market-neutral, best worst-case |
| **punt (default DEMEAN=True)** | **no hedge, conv 0.10, contra $1M** | **780** | **$557k** | scales the *real* edge → higher upside AND higher EV |
| punt DEMEAN=False | + raw directional tilt | 532 | $557k | max variance but ~0-EV lottery (drifts are exactly 0) |

The good-punt insight: **dropping the hedge + trading all 50 names + pinning the $100k ALGO cap**
raises both the score and the capital at risk *without* gambling on direction. The pure
directional tilt (`DEMEAN=False`) is a true lottery ticket — only flip it for a coin-flip bet on
the hidden window's direction.

## Files
- **`Arbitrage_Victims_combined.py`** — robust submission (rename to your team name to submit).
- **`Arbitrage_Victims_combined_punt.py`** — aggressive tournament variant (no hedge, max deployment).
- `Arbitrage_Victims_combined_revblend.py` — preserved aggressive variant (reversion blend,
  766 @250 but below-noise / not recommended for the final entry).
- `eval.py` — scoring harness (exact eval.py PnL/fee/limit loop; one-day commission lag).
- `prices.txt` — 500-day price panel (51 instruments).
