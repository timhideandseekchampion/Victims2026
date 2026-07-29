# algopart3

Part-3 workspace: the current strategies + the 1000-day `prices.txt`, plus two visualizers.

## Strategies (imported from algopart2)
| file | ALGO index leg (instrument 0) |
|------|-------------------------------|
| `SAFE.py`         | fade ALGO's 30-day move (reversion) — the base book |
| `SWING.py`        | catch-up variant |
| `QUAL.py`         | qualifier variant |
| `SAFE_llalgo.py`  | lead-lag skew gate, **name-count** unit (shipped) |
| `SAFE_lldollar.py`| lead-lag skew gate, **net-$** unit (twin of llalgo) |
| `SAFE_llmatch.py` | lead-lag **volume-matched**: index $ = `MATCH_K` × book net-$ tilt, no gate/reversion (`MATCH_K=1.0`) |
| `SAFE_llvol.py`   | **adaptive realized-vol** leg: long ALGO when 20d realized vol is elevated, sized by the signal's live trailing IC (`VOL_GAIN`, `IC_LOOKBACK`) |
| `SAFE_llmeta.py`  | **trailing-performance switch**: runs whichever of LLVOL's or LLDOLLAR's ALGO-leg mechanism had the better trailing `META_L`-day realized PnL, at full size — **overfit, rejected, kept as a documented dead end** |
| `SAFE_llboost.py` | **SAFE_llvol + significance-gated pairwise idio boost**: same adaptive-vol ALGO leg; the idio book additionally boosts a stock's ridge signal from its best statistically-significant "leader" stock's move, gated on Bonferroni-corrected significance *and* a 500-day minimum history requirement — **validated, current best** |
| `SAFE_llboost_v2.py` | **SAFE_llboost + vol-regime-adaptive momentum lookback**: identical idio book and pairwise boost; ALGO leg's momentum lookback switches between 7 days (elevated vol) and 12 days (calm vol) instead of a fixed 10 — **promising but not a clean pass, see below; kept as a separate candidate, not a replacement** |
| `SAFE_llboost_v3.py` | **SAFE_llboost + volatility-restricted boost candidate pool**: identical ridge/ALGO leg; the pairwise boost's leader search is restricted to the 39 highest (causally, trailing-vol-ranked) volatility idio stocks instead of all 49, Bonferroni divisor adjusted to match — **validated, very clean (n_worse=1/61), see below** |
| `SAFE_llboost_v4.py` | **SAFE_llboost_v3 + SAFE_llboost_v2 combined**: the two changes are structurally independent (one in the idio boost, one in the ALGO leg's momentum) and compound — superseded by v6, see below |
| `SAFE_llboost_v5.py` | **SAFE_llboost_v3 + re-tuned `BOOST_IC_L`/`BOOST_MIN_DAY`**: same N=39 candidate-pool restriction as v3, with the other boost sub-parameters re-swept for the new pool size — **validated, cleanest result (n_worse=0/61), see below** |
| `SAFE_llboost_v6.py` | **SAFE_llboost_v5 + SAFE_llboost_v2 combined**: same orthogonal-components logic as v4, built on the refined v5 boost instead of v3 — best result of that session |
| `SAFE_llboost_v7.py` | **SAFE_llboost_v6 + re-tuned `COMBINE_GAIN`**: ALGO leg re-swept against the TRUE, final v6 idio book (never checked before) — every parameter confirmed still-optimal except `COMBINE_GAIN` (3.5→16.0), independently confirmed by a 720-combo joint grid search — best result of that session |
| `SAFE_llboost_v8.py` | **SAFE_llboost_v7 + ALGO min-conviction HOLD deadband**: identical idio book, boost, and ALGO signal construction; on days the ALGO leg's raw combine target falls under 25% of the $100k cap (a near-cancellation of the vol-regime vs. momentum sub-signals, empirically loss-making) it holds yesterday's share count instead of resizing into the small, uncertain-sign target, gated off before 400 days of ALGO history exist — **validated, cleanest result (n_worse=0/61, rolling floor unchanged), see below — current best** |

The idio book (instruments 1–49) is identical across the first eight; `SAFE_llboost.py` and its v2–v7 variants are the exceptions (they extend the idio book with the pairwise boost described below).

## Data
`prices.txt` — 51 instruments × **1000 days** (grader scores the last 250 → days 751–1000).

## Regenerate everything
```bash
python compute_positions.py      # -> positions_data.json   (per-asset positions/PnL, all days)
python build_dashboard.py        # -> dashboard.html         (per-asset entries/exits explorer)
python compute_diagnostics.py    # -> diag_data.json         (exact eval scores + leg attribution)
python build_diagnostics.py      # -> diagnostics.html       ("what changed at 1000 days")
python compute_signals.py        # -> signals_data.json      (ALGO vol-signal significance + persistence)
python build_signals.py          # -> signals.html           ("is the vol edge real, or overfit?")
```
Both `build_*.py` scripts are day-count agnostic (read `nt` from the data), so they work if
`prices.txt` grows again. Use the same interpreter that has numpy/pandas
(`/home/SIG2026/Victims2026/algo26v1/.venv/bin/python` on this box).

## Visualizers
- **`dashboard.html`** — per-asset explorer: price + long/short bands + entry triangles + cumulative
  PnL for each of the 51 names, the ALGO lead-lag skew gate, and a 1st-half/2nd-half persistence
  scatter. Switch strategy/asset; drag to zoom.
- **`diagnostics.html`** — the score story on the fresh draw: scoreboard dumbbell (old vs new window),
  idio-vs-ALGO leg attribution, rolling 250-day score (incl. LLVOL), MATCH_K sweep, cumulative PnL by leg.
- **`signals.html`** — is the LLVOL vol edge real or overfit: cross-sectional vol IC across all 51 names,
  per-name persistence scatter (H1 vs H2), live trailing IC (vol vs lead-lag), and the combine head-to-head.

## Is the LLVOL vol edge real? (see `signals.html`)
- **Not universal:** across all 51 names the vol→return IC averages +0.004 (27/51 positive) — not a broad
  generator property. But signals here are instrument-specific (lead-lag lives in the stocks, absent on ALGO).
- **Does not persist per-name:** corr(H1 IC, H2 IC) across names = −0.04 — so selecting the "significant"
  stocks in-sample is **data-snooping / trading noise**. Do **not** extend vol to the book.
- **Real on ALGO only:** ALGO's vol IC is +0.02→+0.11→+0.14 (strengthening every sub-period) and survives a
  circular-shift surrogate test (p<0.001 full, <0.00025 new).
- **Caveat:** it is index-specific, so persistence to the finals draw depends on the generator. The adaptive
  gate is the safety net — it sizes the leg to zero if the effect is absent. Combining with lead-lag hurts.
- **What the "strengthening" actually is (`test_algo_ic_regime_drivers.py`):** not a recurring switching
  regime — a single, roughly monotonic sign transition concentrated in days ~100–500 (quartile means
  −0.00078 → +0.00092 → +0.00154 → +0.00230; **100% of trailing-IC days negative in 0–250, 0% negative in
  750–1000**; 9 of 13 total zero-crossings fall in the 250–500 quartile alone). The double-IC veto's actual
  firings track this almost exactly: 36/300 days before day 500 (12.0%) vs only 7/500 from day 500 on
  (1.4%) — it did its real work cleaning up the transition and is nearly dormant in the graded window.
  Tested two candidate leading indicators (vol level, trailing-trend direction) hoping to find something
  tradeable *ahead of* the reactive trailing-IC estimate — neither gives a clean signal (vol tertile IC
  +0.052/+0.075/+0.005, no monotonic risk-premium-in-high-vol pattern; trend up/down +0.059/+0.105, a
  modest difference at best). A full-sample GARCH(1,1)-in-Mean fit directly tests the "risk premium"
  framing above and does **not** support it as a single stable coefficient (λ=+0.125, t=0.53, p=0.60) —
  per-quartile λ does drift in the same direction as everything else (−0.11→−1.31→+3.98→+10.45, each
  individually insignificant), consistent with one drifting transition rather than a stationary premium.
  **No better leading indicator found** — the existing reactive gate already concentrates its activity
  almost exactly where the real transition happened and goes quiet once it resolves, which is close to
  the best available response given the alternatives tested.

## Headline (days 751–1000 vs 501–750)
The idio book is intact (~+$160k both windows; idio-only score **585 → 586**). The whole drop is the
ALGO index leg, which flipped **+$28k → −$29k**. Shipped LLALGO score **694 → 452**; idio-only **585 → 586**.
Turning the ALGO leg off recovers ~587 on the new window.

## ALGO index-leg options — head to head (exact eval score; idio book identical)
| index leg | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor |
|---|---|---|---|---|
| OFF (idio only) | 585 | 586 | 651 | 493 |
| SAFE — reversion (fade 30d) | 613 | 444 | 697 | 444 |
| LLALGO — lead-lag binary gate (shipped) | 694 | 452 | 731 | 452 |
| LLMATCH k=1 — volume-matched lead-lag | 564 | 600 | 657 | 482 |
| **LLVOL — adaptive realized-vol** | **655** | **701** | **699** | **527** |

**LLVOL** is the only leg that lifts both the rolling mean and the floor. Its edge is a vol→next-return
effect (IC +0.14 on days 751–1000, stable & positive across every sub-period; permutation p≈0.01 full /
0.025 new). **Caveat:** "high vol → higher next return" is a GARCH-in-mean *risk-premium* pattern — plausible
but likely specific to the **synthetic generator**, not a universal edge. The adaptive gate is the safety
net: if the effect is absent in the finals draw, the causal trailing-IC sizes the leg toward zero.
Momentum and vol-conditioned-momentum were tested and are **not** significant (drop them).

## `SAFE_llmeta.py` — an experiment that turned out to be overfit; kept as a documented dead end
Neither ALGO-leg mechanism is stable across the whole file: LLDOLLAR's fixed-direction lead-lag
skew gate wins days ~150–750 (694 OLD) then decays past ~day 750 (452 NEW); LLVOL's adaptive
vol/momentum switch is flat/choppy before ~day 500 then strengthens through day 1000 (684 → 761).
`SAFE_llmeta.py` tries to exploit that by running whichever mechanism had the better trailing
`META_L`-day realized PnL, at full size. A **perfect-hindsight** day-by-day version of this is
worse than either mechanism alone (noise-dominated, pays heavy commission whipsawing between the
two very differently-sized position schemes) — but a **slow, trailing-mean** switch (sweeping the
lookback 10–250 days) found what looked like a real result at `META_L≈30–36`:

| index leg | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor |
|---|---|---|---|---|
| SAFE_llvol (shipped) | 683.9 | 761.1 | **759.4** | 564.9 |
| SAFE_llmeta, `META_L=33` | 671.1 | 710.3 | 757.1 | **638.0** |

**This is overfitting, not a structural improvement — confirmed, not just suspected:**
- LLVOL's rolling floor comes from ONE specific window (days 190–440), nowhere near OLD or NEW.
  That's the early stretch before LLVOL's own vol-continuation edge had kicked in (the IC-block
  analysis shows day 100–300 was actually negative).
- `SAFE_lldollar` was *itself* originally discovered by hunting specifically on days 400–750 (see
  `SAFE.py`'s own docstring) — i.e. the "fix" for LLVOL's weak window is another mechanism that was
  independently fitted to cover almost that same stretch. `META_L` just tunes how fast to lean on it.
- **`META_L=33` scores WORSE than shipped LLVOL on BOTH OLD and NEW individually** (671.1 vs 683.9,
  710.3 vs 761.1). Every parameter in this file that's actually validated (e.g. `COMBINE_GAIN`)
  earns it by improving *both* disjoint sub-periods at once — this one clears neither.
- Checking which window the "improved" floor actually comes from: it's STILL days 180–430/440
  (638.8/652.5, barely better) but a **new** worst window appears at days 430–680 (638.0) that
  wasn't a problem for plain LLVOL at all. Checked the switch's actual choices during days 190–440:
  it leans on LLDOLLAR 70% of those days (vs 50% for LLVOL) — confirming the mechanism is exactly
  "borrow LLDOLLAR to patch LLVOL's known weak spot," not a general regime-detector. It didn't fix
  the weakness, it moved it next door.

Three compounding layers of in-sample fitting (LLDOLLAR's own params + LLVOL's own params + `META_L`
on top), validated only against metrics computed from the one file all three were fit to, with the
"win" traced to patching one specific known-bad historical window — this is as clean a textbook
overfitting example as this whole investigation produced. **Recommendation: do not ship
`SAFE_llmeta.py`.** It's kept in the repo as a documented negative result, not a candidate.

## `SAFE_llboost.py` — significance-gated pairwise idio boost (validated, clean baseline — superseded by v5/v6, see below)
The idio ridge already captures most extractable pairwise/lead-lag structure (a long series of
gate/blend/stack combination attempts this session all failed to add anything on top of it — see the
rejected pairwise-boost family below). One variant did work: **boost a stock's ridge z-score using
its own best statistically-significant "leader" stock's move**, sized `BOOST_K=1.5`, but only once
each leader relationship clears a **Bonferroni-corrected significance test given the actual sample
size available** (not a fixed threshold), *and* only once **500 days of history** exist at all.

The 500-day minimum is the load-bearing piece. Without it, the boost lifted OLD/NEW/rolling-mean at
every strength tested but monotonically **worsened the rolling floor** (563.8 → 531.9 → … → 479.9 as
the boost size rose) — traced to thin-sample "significant" pairs found on the earliest checkpoints
(200–550 days of history) that were still lucky false positives despite passing the corrected test,
each then trading with real size for dozens of days before the next re-estimate. Gating out any
boost before day 500 removes that failure mode entirely:

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llvol (baseline) | 687.1 | 761.8 | 760.7 | 563.8 | — |
| +boost, no min-history, K=1.5 | 749.2 | 762.7 | 772.6 | 522.3 | 25/61 |
| SAFE_llboost (min-history=500, K=1.5) | 772.5 | 822.2 | 809.2 | 563.8 | 0/61 |
| **SAFE_llboost (this file, sub-params tuned)** | **774.1** | **828.6** | **811.4** | **563.8** | **0/61** |

Validated on three independent robustness axes, not just the one setting above:
- a neighbor sweep over min-history ∈ {450,480,500,520,550} × `K` ∈ {1.0,1.25,1.5,1.75,2.0} lands in
  the same region (several combinations hit `n_worse=0/61` exactly)
- a checkpoint-refit-cadence sweep (10/25/50/75/100 days) gives `n_worse=0/61` at every cadence, with
  NEW actually *improving* as the re-estimate gets fresher — consistent with the shipped design
  (re-estimates fresh from all available history on every call, no stale caching)
- a follow-up sub-parameter sweep (`test_boost_subparam_sweep.py`) over the three knobs never
  touched in the first pass — `BOOST_P` (the magnitude exponent, 0.5–3.0), `BOOST_SCALE_W` (the
  leader's own return-scale window, 100–1100), `BOOST_IC_L` (the sign-check window, 100–400) — found
  `BOOST_P=2.0` was already sitting at its peak, but `BOOST_SCALE_W=500→1000` and `BOOST_IC_L=220→190`
  clear OLD/NEW/rolling-mean simultaneously, confirmed on a joint 27-point neighbor grid (scale_w ∈
  {900,1000,1100} × IC_L ∈ {180,190,200} × K ∈ {1.4,1.5,1.6}) where every single combination scores
  `n_worse=0/61` — a broad plateau, not a lucky point. (`BOOST_SCALE_W=1000` also turned out to just
  saturate to "use all available history" at this file's length — 900/1000/1100 score identically —
  so it's not a fragile fixed window either.)

See `test_boost_floor_mitigation.py`, `test_boost_cadence_robustness.py`, and
`test_boost_subparam_sweep.py` for the full sweeps, and `validate_llboost_full.py` (runs the real
`getMyPosition`, not a backtest approximation — reproduces the official `eval_llboost.py` score of
828.60 exactly).

**Rejected on the way here** (same underlying "pairwise leader → predict next move" hypothesis,
tested about ten different ways, all before the significance+min-history combination above): raw
lead-lag continuation, confirm-gate against a GBM model, additive z-score blend, stacking the ridge
forecast as a GBM feature, pooled-network gating, partial-pooling boost, causal full-window
re-estimation, and a same-day "divergence from usual co-mover" signal (`test_contemporaneous_
divergence.py`, IC=-0.0023, p=0.57, sign doesn't hold across halves). All of these either failed
outright or hit exactly the thin-sample floor problem above before the fix was found.

**Caveat, same category as everything else in this file:** the floor is *unchanged*, not improved —
this is real average edge, not free downside protection, and this session's own synthetic
stress test (see below) put a single 1000-day file's exact score at only the 8th percentile of its
own resampled distribution (std ≈ 118) — so treat the lift above as directionally real, not as a
guaranteed fixed number at finals.

## `SAFE_llboost_v2.py` — vol-regime-adaptive momentum lookback (promising, not a clean pass)
`SAFE_llboost.py`'s ALGO leg uses a fixed `MOM_LB=10`-day momentum lookback, swept and re-swept
multiple times this session — every neighbor tested (every integer 6–20) scored decisively worse,
confirming 10 as a genuine, isolated optimum *as a fixed constant*. Untested until now: whether the
lookback itself should be **regime-dependent** instead of fixed — a common pattern in practice is
momentum decaying faster in high-vol regimes and persisting longer in calm ones.

`SAFE_llboost_v2.py` switches between a 7-day lookback when today's realized vol is elevated and a
12-day lookback when it's calm (`MOM_LB_SHORT`/`MOM_LB_LONG`). Validated on the actual
`getMyPosition` pathway (`test_vol_adaptive_validate.py`, `validate_llboost_v2_full.py` —
reproduces `eval_llboost_v2.py`'s official score of 858.40 exactly):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost (baseline) | 774.1 | 828.6 | 811.4 | 563.8 | — |
| **SAFE_llboost_v2** | **788.9** | **858.4** | **840.1** | **669.5** | 18/61 |

All four headline metrics improve substantially — the +105.7 floor jump is the largest single
improvement found in the whole investigation. Stable across a real neighborhood too
(`test_adaptive_mom_lb.py`): the short=7 lookback works well paired with every long lookback tested
from 11 to 16, not just this one specific pair.

**Why this is v2, not a replacement:** unlike every parameter in `SAFE_llboost.py` itself, this does
not clear `n_worse=0/61` — 18 of 61 rolling windows are worse. The window-concentration diagnostic
found a reassuring, if imperfect, shape: the 18 worse windows lose an average of only **-9.8**
(worst case -19.4), while the 43 better windows gain an average of **+44.8** (up to +114.5) — a
favorable asymmetry, structurally different from every previously-rejected candidate this session
(which typically traded one metric for another outright, e.g. the top-2/3-leader idea's ~41-point
NEW regression). The worse windows cluster mildly around days 610–670, but the losses there are
small — not the single-catastrophic-window pattern documented in `SAFE_llmeta`'s postmortem above.
Net judgment call, not a clean pass: kept as a separate file so both can be compared before deciding
which to actually submit.

## `SAFE_llboost_v3.py` — volatility-restricted boost candidate pool (validated, very clean)
From a 20-idea test queue (`test_ncandidates_causal.py`): the pairwise boost's leader search
originally considers all 49 idio stocks as candidate "leaders" (Bonferroni divisor = 49). Restricting
that search to only the `BOOST_N_CANDIDATES` highest (causally, trailing-realized-vol-ranked)
volatility stocks — and shrinking the Bonferroni divisor to match — removes low-vol, low-power
candidates whose "significant" correlations are more likely to be noise that happens to clear the
bar on a given day.

Swept N from 20–48 with a causal (no look-ahead) trailing-vol ranking, confirmed the ranking is a
stable structural stock property here (39/39 overlap between expanding and trailing-500-day
rankings, checked at days 600/800/999 — not a fragile boundary). Found a genuine, monotonically
improving region from N=29 to N=39 (n_worse falling from 26/61 to 1/61), unlike the isolated
`IC_EW_W=150` spike rejected earlier this session. Validated on the actual `getMyPosition` pathway
(`eval_llboost_v3.py`: official score 837.79 exactly; `validate_llboost_v3_full.py`):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost (baseline) | 774.1 | 828.6 | 811.4 | 563.8 | — |
| **SAFE_llboost_v3 (N=39)** | **793.8** | **837.8** | **825.5** | 563.8 | **1/61** |

The single worse window (end-day 520) is -0.1 points — a rounding-level blip; every other window
from day 500–1000 is flat or a real gain (up to +52 in the 850–990 stretch). Cleaner than v2's own
profile. Caveat: N=40 (n_worse=21/61) sits right after N=39 with a sharp discontinuity, so the exact
choice of 39 (vs. e.g. 35–38, also solid at n_worse 7–16/61) still carries some parameter-sensitivity
risk — see the docstring in `SAFE_llboost_v3.py` for the full sweep.

## `SAFE_llboost_v4.py` — v3 + v2 combined (superseded by v6)
v3's change (idio boost candidate pool) and v2's change (ALGO leg momentum lookback) are structurally
independent — they don't touch the same code path — so they were tested combined to see if the gains
compound. They do. Validated on the actual `getMyPosition` pathway (`eval_llboost_v4.py`: official
score 867.52 exactly; `validate_llboost_v4_full.py`):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost (baseline) | 774.1 | 828.6 | 811.4 | 563.8 | — |
| SAFE_llboost_v3 (N=39 only) | 793.8 | 837.8 | 825.5 | 563.8 | 1/61 |
| SAFE_llboost_v2 (adapt-mom only) | 788.9 | 858.4 | 840.1 | 669.5 | 18/61 |
| **SAFE_llboost_v4 (combined)** | **808.7** | **867.5** | **854.3** | **669.5** | **10/61** |

v4 is strictly better than v2 alone on every metric, including n_worse (10/61 vs 18/61) — the
candidate-pool restriction doesn't just add its own gain, it also reduces the count of windows where
v2's momentum change underperforms. The floor gain (+105.7 over baseline) is identical to v2's own,
since the boost restriction never touches the ALGO leg. The 10 remaining worse windows (end-days
610–720, inherited from v2's own known soft spot) lose an average of -10.4 (worst -19.0) against a
+53.3 average gain (best +114.5) on the other 51 — same favorable asymmetry as v2, fewer bad windows.
Confirmed identical to v2 on days 100–400 (out-of-sample, boost inactive before day 500 in both:
576.0 in both) — zero side effects from the candidate-pool change before it can possibly activate.

## `SAFE_llboost_v5.py` — v3 refined (validated, cleanest result)
From an 80-idea follow-up test batch (`test_batch80_catA_boostpool.py`), run after v3/v4 shipped:
since v3 restricted the boost's leader pool to N=39, the OTHER boost sub-parameters — originally
tuned against the old N=49/unrestricted pool — were re-swept to check they still sit at their
optimum. Two didn't: `BOOST_IC_L` 190→250 and `BOOST_MIN_DAY` 500→480 each independently improve
slightly, confirmed via a neighbor-stability check (IC_L in 230–270 all give n_worse=1/61; MIN_DAY
in 470–490 all give n_worse=0/61 — stable plateaus, not isolated spikes), and they compound.
Validated on the actual `getMyPosition` pathway (`eval_llboost_v5.py`: official score 839.13
exactly; `validate_llboost_v5_full.py`):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost (baseline) | 774.1 | 828.6 | 811.4 | 563.8 | — |
| SAFE_llboost_v3 (N=39) | 793.8 | 837.8 | 825.5 | 563.8 | 1/61 |
| **SAFE_llboost_v5 (N=39 + retuned)** | **796.6** | **839.1** | **828.3** | 563.8 | **0/61** |

A small refinement on top of v3, not a new mechanism — same N=39 pool, just re-tuned IC_L/MIN_DAY.
Beats v3 on every metric and achieves a clean n_worse=0/61, matching the bar `SAFE_llboost.py`
itself was held to.

## `SAFE_llboost_v6.py` — v5 + v2 combined (best result of the session)
Same orthogonal-components logic as v4 (v3+v2), built on the refined v5 boost instead of v3.
Validated on the actual `getMyPosition` pathway (`eval_llboost_v6.py`: official score 868.87
exactly; `validate_llboost_v6_full.py`):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost (baseline) | 774.1 | 828.6 | 811.4 | 563.8 | — |
| SAFE_llboost_v5 (boost only) | 796.6 | 839.1 | 828.3 | 563.8 | 0/61 |
| SAFE_llboost_v2 (mom only) | 788.9 | 858.4 | 840.1 | 669.5 | 18/61 |
| SAFE_llboost_v4 (v3+v2) | 808.7 | 867.5 | 854.3 | 669.5 | 10/61 |
| **SAFE_llboost_v6 (v5+v2)** | **811.4** | **868.9** | **857.0** | **669.5** | **9/61** |

Beats v4 on every metric (marginally) — the IC_L/MIN_DAY refinement carries through the
combination cleanly. Confirmed identical to v2 on days 100–400 (boost still inactive there in
both: 576.0 in both).

## `SAFE_llboost_v7.py` — re-tuned `COMBINE_GAIN` (superseded by v8, see below)
v6's ALGO-leg parameters were validated at two different, now-stale points: `COMBINE_GAIN=3.5` was
chosen before ANY pairwise boost existed; `MOM_LB_SHORT/LONG=7/12` was chosen against the ORIGINAL
`SAFE_llboost`'s boost (N=49, IC_L=190, MIN_DAY=500), not v6's final one. Neither had ever been
re-checked against the true, final v6 book (N=39, IC_L=250, MIN_DAY=480, SCALE_W=1000 boost +
the v2 adaptive momentum, both active at once) — `test_algo_leg_resweep.py`'s own resweep predates
both v3's pool restriction and v5's IC_L/MIN_DAY retune. Two independent methods closed this gap:

1. **Direct coordinate resweep** (`test_v7cand_algoresweep.py`) against the precomputed, exact true-
   v6 idio book (sanity-checked to reproduce the real `getMyPosition` bit-for-bit first): `VOL_WIN`,
   `VOL_Z`, `IC_FAST`, `SWITCH_GAIN`, `IC_EW_HL`, `MOM_LB_SHORT`, `MOM_LB_LONG` are all still sharp,
   isolated optima — every neighbor tested falls to `n_worse` in the 38–61/61 range. `COMBINE_GAIN`
   was the one exception: monotonically improving on every metric from 2.0 up through a broad
   plateau at 15–17 (`test_v7cand_combine_gain_extend.py`, `_fine.py`), then mildly rolling over by
   25–30 — a real peak, not an unbounded artifact.
2. **Independent 720-combo joint grid search** (`test_v7cand_joint_search.py`) over `BOOST_K` ×
   `BOOST_IC_L` × `MOM_LB_SHORT` × `MOM_LB_LONG` × `COMBINE_GAIN` simultaneously — converged on
   the exact same single lever. Perturbing `BOOST_K`, `BOOST_IC_L`, or either `MOM_LB` away from
   their v6 values while `COMBINE_GAIN` sits at its new best value only makes things worse; only
   `COMBINE_GAIN` itself was stale.

**Why this is real, not overfit — the mechanism, not just the number:** `COMBINE_GAIN` only scales
the raw ALGO dollar target before it's clipped to the $100k cap. Since the underlying (vol+momentum)
signal sum is bounded, raising the gain just lowers the signal magnitude needed to hit the cap —
i.e. it pushes the ALGO leg from partial magnitude-weighted sizing towards full-conviction
sign-based sizing whenever the two signals agree, the *same* principle already validated
everywhere else in this repo (idio book sizing; the 80-idea batch's own conclusion that
"full-conviction sign-based sizing keeps winning" against every magnitude/Kelly/confidence-ramp
scheme tried). It isn't a "turn the dial to infinity" result either — the curve genuinely peaks and
rolls over, because a large enough gain eventually forces even near-cancelling, low-conviction
disagreement to the cap too. Turnover is unaffected (the gain changes magnitude only, not
`sign(av)`, so no extra commission churn). Validated on the actual `getMyPosition` pathway
(`eval_llboost_v7.py`: official score 888.53 exactly; `validate_llboost_v7_full.py`, which —
matching every prior `vN` validator's convention — compares against the original `SAFE_llboost.py`):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost (original baseline) | 774.1 | 828.6 | 811.4 | 563.8 | — |
| SAFE_llboost_v6 | 811.4 | 868.9 | 857.0 | 669.5 | 9/61 |
| **SAFE_llboost_v7 (COMBINE_GAIN=16)** | **830.3** | **888.5** | **876.8** | **674.4** | **1/61** |

Every one of OLD/NEW/rolling-mean/rolling-floor improves over v6, simultaneously, and `n_worse=1/61`
against the original baseline is cleaner than v6's own 9/61.

## `SAFE_llboost_v8.py` — ALGO min-conviction HOLD deadband (superseded by v9, see below)
The "v7 budget" diagnostic above found the ALGO leg runs at 95.1% cap utilisation and is essentially
never flat, but on the 25/499 days where the raw combine target lands under ~50% of the $100k cap —
a near-cancellation of the vol-regime signal against the momentum signal — those days lose **-$81/day**
on average, against +$309/day (50–99% util.) and +$188/day (≥99% util.) everywhere else: low
conviction there is also wrong-sign-prone, not just small. Separately, the regime-driver investigation
(above) found no better *predictive* signal for the ALGO edge's sign — so rather than trying to see the
flip coming, this targets the same information the shipped leg already has: **don't resize into a
small, uncertain-sign position — hold yesterday's share count instead.**

Two treatments tested (`test_v7_algo_deadband.py`): FLATTEN (go to 0) and HOLD (keep yesterday's
position). HOLD passed OLD+NEW+rolling-mean at every threshold 0.10–0.25, but all 16 "worse" rolling
windows were the earliest ones (end_day 400–470, ~-15 pts each) — the same shape as the reason
`BOOST_MIN_DAY` exists: an adaptive mechanism unreliable on thin history. Fix: add an analogous
minimum-history gate. A joint sweep over threshold × `DEADBAND_MIN_DAY` (`test_v7_algo_deadband_v2.py`,
40 configs) found `min_day≥400` gives a clean plateau — n_worse=0/61 at min_day∈{400,450,550,600} ×
thresh∈{0.10,0.15,0.20,0.25} — not a single lucky point; a neighbor grid around the selected
thresh=0.25/min_day=400 confirms it. Implemented as a real standalone module (module-level HOLD
state, same pattern `_limits`' `_DLR` cache already uses) and validated on the actual `getMyPosition`
pathway, not just the backtest reconstruction used for the sweep — reproduces the sweep's numbers
exactly (`validate_llboost_v8_full.py`) and gives official score **888.86** (`eval_llboost_v8.py`,
up from v7's 888.53):

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 (vs v7) |
|---|---|---|---|---|---|
| SAFE_llboost_v7 | 830.3 | 888.5 | 876.8 | 674.4 | — |
| **SAFE_llboost_v8 (HOLD, thresh=0.25, min_day=400)** | **847.4** | **888.9** | **886.2** | **674.4** | **0/61** |

Every one of OLD/NEW/rolling-mean improves; the rolling **floor is unchanged to the decimal** — the
deadband never touches the worst window at all, unlike most prior improvements in this file which
trade a bit of floor for mean. n_worse=0/61 matches v5's own cleanest result.

**Implementation caveat, found and fixed during validation, not just noted:** HOLD needs a real
"yesterday", which is cross-call state — safe under this repo's full walk-up harnesses
(`validate_*_full.py`, live sequential trading), but `eval_llboost_vN.py`'s official-score convention
calls `getPosition` only over the graded window, skipping the walk-up — a genuine cold start with no
real prior position to hold. Fixed by only trusting the cached position when the current call is the
immediate sequential successor of the last one; otherwise the deadband is bypassed for that one call
(computed exactly as v7 would, never a fabricated flat position) and resumes normally from the next
call. Confirmed consistent under both conventions (888.86 official vs 888.9 full-walk — the same
rounding-level agreement v7 itself shows between its two numbers), so this isn't a latent bug waiting
to surprise a different harness.

## `SAFE_llboost_v9.py` — beta-adjusted idio ridge target (superseded by v10, see below)
`test_pc2_probe.py` found the ridge fit's same-day residual cross-correlation across the 50 idio
names is **+0.20 even after fitting** — real, unexplained common-mode co-movement left in the
training target Y, which a lagged (yesterday→tomorrow) regression can never remove since it's
contemporaneous. That shared variance isn't stock-specific signal; every one of the 50 per-half-life
fits was spending estimation effort jointly explaining it.

**First attempt, proven a no-op, not just rejected** (`test_v10cand_demean_y.py`): subtracting the
same value (the daily equal-weighted average) from every one of the 50 response columns before
fitting is **algebraically inert** here. The fit is linear in Y, so a uniform per-day shift moves
every stock's forecast by an *identical* constant that day — which the existing
`fi = pred - pred.mean()` step removes anyway. Verified by hand (completing the square) and
numerically: every partial-demean weight 0.1–1.0 gave bit-identical scores to v8 (n_worse=0/61 at
every one — literally zero windows differed, not just "no improvement").

**The fix** (`test_v10cand_beta_demean.py`): make the correction non-uniform. Subtract
`beta_j * factor` using each stock's OWN causally-estimated beta to the idio common-mode factor
(trailing `BETA_DEMEAN_W` days), instead of the factor itself. Since `beta_j` varies by stock this
does not reduce to a uniform shift (confirmed on synthetic data before touching real data: a
beta-weighted correction changes the forecast by a real amount, a uniform one measurably does not —
diff ~1e-15 vs ~0.07 on the same toy problem). A joint sweep (3×5, then a finer 5×5 grid) found a
genuine **plateau**, not a lucky point — every config in `lam∈[0.4,0.6] × BETA_DEMEAN_W∈[400,600]`
improves rolling mean AND floor simultaneously:

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 (vs v8) |
|---|---|---|---|---|---|
| SAFE_llboost_v8 | 847.4 | 888.9 | 886.2 | 674.4 | — |
| **SAFE_llboost_v9 (lam=0.6, BETA_DEMEAN_W=500)** | **848.8** | **893.3** | **894.1** | **708.6** | 16/61 |

This is the first candidate in this file's whole history to improve the rolling **floor by this
much (+34.1) simultaneously with the mean** — every prior improvement (v3 through v8) either left
the floor unchanged or improved it only marginally while gaining mean elsewhere. n_worse=16/61 isn't
as clean as v8's own 0/61 — reported honestly, not hidden — but is 0/61 against the **original**
`SAFE_llboost` baseline, the same convention used to headline every prior version. Validated on the
real `getMyPosition` pathway (`validate_llboost_v9_full.py`): reproduces the sweep's numbers exactly.
Official score (`eval_llboost_v9.py`): **893.32** (vs v8's 888.86). Fully causal, no cross-call state
(unlike v8's ALGO HOLD deadband) — the eval-harness cold-start class of bug v8 needed a fix for
cannot recur here.

## v10 follow-ups: walk-forward check (reassuring) and an ALGO crossover extension (rejected)
**Walk-forward robustness check on v10's parameter selection** (extending the same diligence applied
to v9's beta-demean earlier this session): re-ranked the full 45-config `(long_w, weight)` grid using
ONLY the OLD window, then checked the untouched NEW window as a genuine holdout — and the reverse.
This holds up far better than v9's did: **v10's actual pick (`long_w=22, weight=0.015`) is the
#1-ranked config by OLD alone** (and 8/10 top-OLD configs also beat baseline on the untouched NEW
window), and ranks **#5/45 by NEW alone** (7/10 top-NEW configs also beat baseline on untouched OLD).
Across the full grid, **18/45 (40%) beat baseline on both windows with no selection at all** — versus
9% for v9's grid — spanning nearly every `long_w` and `weight` tested, not a narrow corner. v10's
parameters would very likely have been selected under strict temporal separation.

**Extending the same idea to ALGO — rejected, on three independent, converging checks.** The
idio-side signal fades a stock's short-term move only when it opposes its own medium-term trend,
relative to the cross-section. ALGO has no cross-section, so the natural analogue is a pure
time-series version: fade ALGO's own short-term price move only when it opposes ALGO's own
medium-term trend, voted in as an additional blend on top of the existing target
(`test_v17cand_algo_crossover.py`). The initial sweep (0/45 configs beating v10) showed a suspicious
pattern — at small windows, NEW kept improving as blend weight rose while OLD kept degrading — and
rather than reject on that alone, three follow-up checks probed *why*:

1. **Raw, model-free IC of the vote against ALGO's own next-day return, by era**
   (`test_v17_algo_crossover_diag.py`) — no blend weight, no interaction with the rest of the book,
   just the crossover vote itself. It is **not stable across parameter choice**: `short5_long10` and
   `short5_long15` show positive IC in both OLD and NEW (just much stronger in NEW: +0.05→+0.33 and
   +0.04→+0.23), but `short8_long22` **flips sign entirely** (OLD −0.09, NEW +0.03) and
   `short10_long30` flips the *other* direction (OLD ≈0, NEW −0.13). A genuinely robust signal
   shouldn't reverse sign from minor lookback changes — v10's rank-stability didn't, across
   `long_w`∈[15,28] and `short_w`∈[6,12]. The trailing-250d IC also crosses zero 2-5 times over the
   file for these pairings — oscillating, not the single clean transition ALGO's own vol-timing edge
   showed earlier this session, nor the stable-throughout pattern the idio boost shows.
2. **Walk-forward check on the full blended-book grid** (49 configs, same method applied to v9/v10):
   **0/49 configs beat baseline on both windows with no selection at all** — versus 9% for v9's grid
   and 40% for v10's. Selecting by NEW alone, **every single one of the top 8 configs fails the OLD
   holdout**. Categorically weaker than either shipped result, not just "less clean."
3. **Trailing-IC-gated version** (`test_v17_algo_crossover_gated.py`) — tests whether making the vote
   regime-adaptive (only trust it when it's recently been working, the same philosophy already
   validated in ALGO's own `_side()` double-IC gate) rescues it, the way a gate might rescue a signal
   that's real but intermittent. **0/27 gated configs pass**, and the OLD-degrades/NEW-improves
   pattern persists identically under the gate — this isn't a signal that's sometimes on and
   sometimes off in a way a trailing filter can separate; it's a persistent asymmetry between the two
   eras that no amount of regime-adaptivity fixes (contrast with `test_v7cand_adaptive_boostk.py`,
   where a similar gate failed for the OPPOSITE reason — the boost's edge was too *stable* to need
   one; here the edge is too *unstable* for the gate to salvage).

Three independent methods — including the one method (adaptive gating) that has previously rescued
borderline ideas elsewhere in this file — agree. Rejected with high confidence, not as a judgment
call on a marginal weight.

## Parametric-bootstrap stress test for v9/v10 — a genuinely nuanced result, not a clean pass or fail
The walk-forward checks above only re-slice the SAME 1000 historical days used to find both
mechanisms. `test_v10_stress_synthetic.py` asks a harder question: does the improvement generalize
to FRESH synthetic draws from a generator that does NOT specifically encode beta-demean or
rank-stability (extending `stress_test_synthetic.py`'s existing one-factor-market + VAR-like-ridge +
stochastic-vol calibration)? Paired design: run baseline/v9/v10 on the SAME draw, compare.

**Original generator (i.i.d. residuals, N=25 draws): both mechanisms look like noise.** v9 beats
baseline on only 44% of draws (mean diff −6.5, p=0.56); v10 beats v9 on 56% (mean diff −0.6, p=0.79).
Neither remotely significant.

**But the generator has a specific, checkable flaw.** Directly measured: the real data's same-day
residual cross-correlation (this exact ridge spec) is **+0.202** — matching `test_pc2_probe.py`'s
earlier finding almost exactly — while the original generator draws residuals independently (zero
correlation by construction). It is structurally missing the *exact* feature v9's beta-demean
targets, making it an unfair null for that specific mechanism.

**Refined generator (residuals given the measured +0.202 common-mode correlation, N=20 draws):** v9
beats baseline on **65%** of draws (mean diff **+11.6**, still p=0.35 — directionally supportive, not
formally significant at this sample size). **v10 vs v9 stays a coin flip (50%, mean diff −1.4,
p=0.49)** even with this fix, because the refinement targets v9's specific mechanism (residual
correlation), not v10's (short/long price-level crossover dynamics) — a fair synthetic test of v10
would need a differently-enriched generator (added own-asset momentum/reversion autocorrelation
beyond a plain lag-1 VAR) that hasn't been built.

**Honest reading:** this is real, informative evidence, not a verdict either direction. v9's
mechanism gains meaningful support once the null model stops being structurally biased against
exactly what it targets — a good sign, short of formal confirmation. v10's mechanism is **genuinely
untested by this method, not refuted by it** — the synthetic generator simply isn't built to contain
what it exploits. Both real-data deltas (v9-baseline +64.7, v10-v9 +19.3) remain within roughly
1-1.5 standard deviations of these synthetic null distributions, which is the correct, calibrated
way to describe the uncertainty here rather than treating either the walk-forward checks or this
test as the final word alone.

## Follow-up: an honestly-calibrated synthetic test built specifically for v10 — still doesn't confirm it
The prior stress test explicitly flagged that no generator had been built containing what v10's
rank-stability mechanism targets. `test_v10_stress_synthetic_v2.py` attempts this properly rather
than leaving it unresolved:

**Checked for an honest calibration target first, before building anything.** The naive idea — raw
per-stock own-return autocorrelation at short vs medium lags — is a dead end: measured on ridge
residuals at lags 1-30, every value is under 0.013 in magnitude, no pattern. There is no honest,
non-arbitrary structure here to inject. But v10 doesn't actually use raw own-autocorrelation — it
uses the *cross-sectional* z-scored short/long divergence. That specific quantity, measured directly:
pooled IC of v10's exact vote construction against next-day return, full real sample = **+0.0147**
(n=14,487) — small, but real and directly measurable.

**The enrichment:** on top of the already-validated residual common-mode fix (ρ=0.202), inject a
small return component proportional to the same causally-computed vote signal, calibrated (via a
quick interpolation loop) so the resulting synthetic pooled IC matches +0.0147 — not tuned to
whatever magnitude would make v10 win.

**Result: v10 still doesn't beat v9 reliably — if anything, slightly worse.** Across 25 fresh draws
from this fully-enriched generator: v9-baseline mean diff +9.1 (56% win rate, p=0.48 — consistent in
direction with the earlier refined result of +11.6/65%); **v10-v9 mean diff −2.3 (36% win rate,
p=0.27)** — pointing the wrong way, though not significantly.

**Two honest limitations stated plainly, not hidden:** (1) the injection is a *uniform* average
effect on every disagreement day — it may not reproduce whatever more heterogeneous, concentrated
relationship actually drives the real edge, even with the average IC matched; (2) the +0.0147
calibration target is itself measured **in-sample** on the same historical data v10 was tuned
against, so even this more careful attempt isn't a fully independent external validation.

**Updated honest picture:** v9's evidence has now been mildly supportive across two independently-
built generators (56-65% win rates, consistently positive mean, still short of formal significance).
v10's has failed to generalize on both attempts (56%→36% win rates), including one specifically
engineered to contain its exact target structure. Not decisive — but this is real, additional
evidence that should raise, not lower, the weight placed on eventually seeing genuine out-of-sample
tournament data before trusting v10's magnitude. The real-data and walk-forward evidence for v10
both remain genuinely positive; this tempers confidence in the *size* of the edge, not a reason to
revert it.

**Follow-up diagnosis: WHY doesn't it replicate, and should that worry us?** Broke down the real-data
(v10 − v9) daily PnL difference directly, and benchmarked it against the (v9 − v8) transition (already
better-supported by the synthetic tests) as a calibration point for what "normal" looks like:

| | v9 vs v8 (beta-demean) | v10 vs v9 (rank-stability) |
|---|---|---|
| Days differing | 317/500 (63%) | 130/500 (26%) |
| Win rate on differing days | 52.7% | 57.7% |
| Top 5 \|diff\| share of total | 11.0% | 26.5% |
| Top 10 \|diff\| share of total | 19.0% | 40.2% |
| Effect excluding top 10 days | **increases** (1390→4703) | **drops by more than half** (10273→4260) |

**v9's edge is "death by a thousand cuts"** — broad (317 days), barely above a coin flip per
instance (52.7%), and it actually *loses* on its biggest-magnitude days; the real edge comes from
volume (many small favorable calls), not a few lucky wins. That is the textbook signature of a
genuine, low-variance improvement, consistent with its more favorable synthetic showing.

**v10's edge is lumpier.** It engages broadly across names (41/50 — not a one-stock fluke) but on
far fewer days (130), with a higher per-instance win rate (57.7%), and **more than half its total
magnitude comes from just 10 of 500 days (2% of the period).** A signal whose realized size depends
this much on whether a handful of large-return days happen to land favorably is inherently
higher-variance across independent redraws — which is exactly what makes a 25-draw synthetic test
underpowered to detect it reliably, even if the underlying mechanism is genuinely real on average.
This does not mean v10 is fake (57.7% win rate and 41-name breadth are real positive signs) — but it
locates precisely *why* the synthetic tests struggled, and confirms the size of v10's improvement is
materially less robust than v9's, not just "harder to prove."

## Closing out five more orphaned ideas: skewness, hub-degree, two nonlinear transforms
Continuing the search for a genuinely new signal family (in the spirit of what made v9/v10 work,
rather than re-mining the already-exhausted ridge/boost estimator space): five previously-written
but never-scored test scripts, run to a real verdict.

- **Return skewness** (`test_skewness_signal.py`) — per-stock rolling skewness as a predictor of
  its own next-day return (raw, |skew|→|return|, and self-relative z-scored versions). **Rejected**:
  all three ICs are tiny (≤0.01 in magnitude) with permutation p-values 0.26–0.69 — no detectable
  signal in any form tested.
- **Hub/influencer degree** (`test_hub_influencer.py`) — is a stock's fan-in count (how many other
  names currently have it as their significant leader) itself informative, either about its own
  return or about how reliable the ridge forecast is for it? **Rejected**: the degree distribution is
  heavily degenerate (78.3% of stock-days have degree 0), and ridge sign-hit-rate is essentially flat
  across every degree quartile (52.4–52.7%) — being a hub carries no information either way.
- **Nonlinear (power-law) reversion transform** (`test_nonlinear_reversion.py`) — since the boost's
  own convex `sign(x)*(|x|/scale)^P` transform (P=2.0) is validated, does the same shape help the
  reversion leg (currently linear, P=1)? **Rejected, cleanly**: P=1.0 is the exact optimum
  (n_worse=0/61), degrading monotonically in both directions (P=0.5→rmean 769.1, P=2.0→732.3, vs
  P=1.0's 811.4) — a real, isolated optimum, not an artifact.
- **Nonlinear lead-lag probe** (`test_nonlinear_probe.py`) — across the full 2450-candidate-pair grid,
  is any single relationship's strength convex in move size (large leader moves predict
  disproportionately more than small ones)? Found one pair, DUCT→AMRP, that clears a max-corrected
  permutation test decisively (p<0.3%, persistent across both H1/H2 sub-periods). **Investigated
  further rather than either adopted or dismissed outright**: checked whether this relationship is
  already exploited — it is. DUCT is in the trailing-vol candidate pool on 100% of days, and AMRP has
  DUCT as its Bonferroni-significant leader on 100% of days since the boost activated. This is the
  boost's own primary, always-active relationship for AMRP, and the boost already applies a convex
  (P=2.0) transform to it. The finding is a **confirmation that the existing design choice is
  well-justified for its most heavily-relied-upon pair**, not a new, unexploited signal.

## Commission/turnover sanity check on v10 — unchanged economics, gain is genuine sign accuracy
Checked whether v10's new blended signal shifted the book's basic economics in a way that could
change standing conclusions (the 30:1 mean-vs-variance elasticity, the ~5% commission ceiling from
the "v7 budget" diagnostic). It hasn't:

| | commission (% of gross) | idio flips/name (NEW, 250d) | ALGO turnover (mean \|Δshares\|/day) |
|---|---|---|---|
| v9 | 5.23% (OLD) / 4.88% (NEW) | 114.3 | 237.18 |
| v10 | 5.11% (OLD) / 4.77% (NEW) | 113.9 | 237.18 (byte-identical — v10 never touches ALGO) |

Commission is essentially flat (v10 marginally *lower*), turnover is nearly identical (114.3 vs
113.9), and ALGO's own turnover is exactly unchanged since v10 doesn't touch that leg. **v10's entire
gain is coming from genuinely better sign accuracy, not from trading more** — and since turnover
hasn't moved, the standing rejection of sign-stickiness/hysteresis (killed by the same 30:1 elasticity
against v8 and v9) still applies unchanged against v10; no need to re-test it.

## Four externally-suggested ideas, tested — one shipped as `SAFE_llboost_v10.py` (current best)
A user shared four suggestions from an external research conversation for improving on `SAFE_llboost_v7.py`.
All four were tested against the current best at the time (v9); the source descriptions were
partially or fully truncated for two of them, so the reconstructions are stated explicitly below
rather than presented as verified replicas.

**1. Let the pairwise boost use negative leader relationships — rejected, decisively.** The current
`_pairwise_boost` selects each follower's leader by strongest absolute correlation (already symmetric
to sign), but then discards the pair if the realized boost IC is non-positive (`if ic <= 0: continue`)
— even though a Bonferroni-significant relationship was just found, just an inverse one. Tested
inverting instead of discarding (`test_v13cand_signed_boost.py`): unlocks 33% more boost coverage
(1837 additional stock-days) but **0/6 configs beat v9** — rmean drops from 894.1 to 877-881 across
every magnitude-floor setting, driven mainly by OLD degrading. A follow-up magnitude-threshold sweep
on the negative side (0.0 to 1.0, the maximum possible correlation) shows results converge
monotonically toward — but never past — the v9 baseline as the threshold tightens, reaching an exact
tie once the threshold is strict enough that no negative pair ever clears it. The existing `ic<=0:
discard` rule already sits at the optimum of this entire spectrum; it isn't leaving value on the table.

**2. A "signal agreement" gate on ALGO's `sig`/`msig` — rejected, decisively.** Reconstructed from a
partially-visible description as gating on SIGN agreement between the vol-regime (`sig`) and momentum
(`msig`) sub-signals — distinct from the shipped magnitude-based HOLD deadband (v8), since two
sub-signals can disagree in sign while summing to something large, or agree while summing to
something small. Tested four treatments on disagreement days (flatten / fall back to vol-only sizing
/ reduced combine-gain / hold yesterday's position, `test_v15cand_algo_agreement.py`): **every
variant scores far worse** (rmean 729-883 vs 894.1, n_worse 60-61/61 in every case). The reason:
`sig` and `msig` disagree on **48% of days** — nearly half the time, not a rare event — so damping
conviction on all of them guts real edge. This is fundamentally different from the deadband, which
targets a narrow ~5% minority of genuinely near-zero-magnitude days; "sign disagreement" is far too
broad a criterion here.

**3. Improve the pairwise boost with leader stability — not a clean win, re-confirms the existing H3
rejection.** This is the same mechanism as `test_h3_leader_stability.py`/`test_h3_stage2_backtest.py`
(already tested and rejected earlier this session against an older baseline). Re-ran fresh against v9
(`test_v14cand_leader_stability.py`) rather than just citing the old numbers: every HARD gate variant
fails; one SOFT-multiplier config (`stab_w=40, shrink=0.3`) technically clears the joint bar
(rmean=894.5), but a 5×5 neighbor grid around it shows only that single point passes — the other 24
neighbors mostly land close (rmean 890-895) but fail on OLD or NEW individually. A near-miss cluster,
not a validated one; doesn't meet this repo's bar for adoption.

**4. Rank-stability trend/pullback signal — validated, shipped as `SAFE_llboost_v10.py`.** Named
`rank_stability_short8_long15` in the source, described only as "bought medium-term leaders after
short-term pullbacks and shorted medium-term laggards after short-term rebounds" — the exact
construction was never fully visible. Reconstructed as a cross-sectional short/long return z-score
crossover, algebraically reducing to a short-term reversal gated to fire only when it opposes the
medium-term trend (see `SAFE_llboost_v10.py`'s docstring for the exact derivation). A joint grid over
blend weight × long-window found a genuine, broad, multi-dimensional plateau — `long_w∈{18,20,22,24,28}`
all pass at weight≈0.015-0.02 (26, 30 roll over, a real peak not an unbounded artifact), and at the
best point every `short_w` from 6-12 also passes. Selected `short_w=8, long_w=22, weight=0.015`
(best by rolling mean among the cleanest configs) — **the cleanest, largest single-step result in
this file's whole history:**

| | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor | n_worse/61 (vs v9) |
|---|---|---|---|---|---|
| SAFE_llboost_v9 | 848.8 | 893.3 | 894.1 | 708.6 | — |
| **SAFE_llboost_v10 (short8/long22, weight=0.015)** | **871.0** | **912.6** | **909.8** | **709.7** | **0/61** |

n_worse=0/61 against both v9 directly and the original `SAFE_llboost` baseline — as clean as v5's and
v8's own 0/61 results, on the largest rmean gain (+15.6) of any single step so far. Validated on the
real `getMyPosition` pathway (`validate_llboost_v10_full.py`): reproduces the sweep's numbers exactly.
Official score (`eval_llboost_v10.py`): **912.64** (vs v9's 893.32). Fully causal, no cross-call
state — same reasoning as v9's beta-demean, no eval-harness cold-start risk.

## Closing out the ridge-alternative backlog: 6 more mechanisms rejected
Before designing anything new, a batch of already-written but never-scored test scripts was re-run
against a real baseline to check nothing was secretly a win. All decisive, all rejected:

- **Kalman filter / RLS coefficients** (`test_q20_items01_04_ridge_variants.py`, continuously-adapting
  state instead of the fixed 4-half-life EW blend): rmean collapses from 811.4 to 614-634 at every
  process-noise setting tested, floor collapses to 353-364. Decisive.
- **PCA pre-reduction** (project the 51-instrument panel onto top-K components before fitting):
  rmean 474-701 (vs 811.4), floor as low as 61.2 at K=5. Consistent with RRR's own rejection — this
  relationship isn't compressible from either the predictor or response side.
- **Quantile regression** (median-target forecast instead of ridge's MSE target, periodic refit):
  rmean 386.2 vs 811.4. Decisive.
- **Winsorized returns** before fitting: no clean win — best config (K=3.0) improves OLD and rmean
  but is worse on NEW, failing the joint bar; K=4.0 is essentially a wash.
- **Elastic Net / Lasso** (replacing ridge's dense L2 with sparse/mixed shrinkage): best IC achieved
  (0.0527 / 0.0504) never exceeds the ridge reference's own IC (0.0563) at any tested
  alpha/l1_ratio — rejected on the cheaper IC-only bar before a full traded-score harness was needed.
- **GBM vs. ridge on a hand-engineered feature panel** (`test_gbm_vs_ridge_score.py`): both the GBM
  and a simple sklearn Ridge score far below the actual shipped idio book (≈0 to −125 vs 584-676) —
  the feature-panel approach itself isn't competitive, regardless of model on top of it.

All six point the same direction as RRR: this relationship is dense, not sparse/low-rank/compressible,
and the linear L2 ridge on the full 51→50 structure is hard to beat with an alternative estimator or
loss. Not exhaustive (the batch80 categories and a `test_stacking.py`/`test_gbm_panel_v2.py` family
remain unscored), but six independent, decisive, one-directional results is a strong prior against
that remaining pile containing a win.

## Predictor-wise (empirical-Bayes-style) ridge shrinkage — rejected, 0/15, monotonic
Uniform `RIDGE_A` shrinks all 51 predictors' loadings by the same fixed amount regardless of how
reliably each predictor's own signal is estimated. Tested a differential version: each predictor's
penalty scaled by `(mean-reliability / its own trailing reliability)^GAMMA`, where reliability is
the pooled average |correlation| of that predictor against all 50 idio targets over a trailing
window (`test_v11cand_predictor_shrink.py`, on top of the shipped `SAFE_llboost_v9`). `GAMMA=0`
reproduces v9 exactly (confirmed). Every `GAMMA>0` tested makes it worse, monotonically: `GAMMA=0.5`
is roughly a wash (rmean 892-895, but never clears OLD+NEW+rmean jointly), and it degrades steadily
from there (`GAMMA=3.0`: rmean drops to 851-863 across all three windows tested). Same shape as RRR's
rejection — best at "don't do this." Consistent with the mild prior already in this file: per-half-life
differential shrinkage (a different axis of non-uniformity) was already rejected as "uniformly lose."

## Huber-robustified ridge (IRLS) — rejected, isolated spike, not a plateau
Tested whether down-weighting extreme training days (Huber loss via IRLS, one reweighting pass; see
`test_v12cand_huber.py` for the honest simplification — one combined per-day robustness weight from
the pooled residual magnitude across all 50 targets, not a true per-response Huber fit, to keep the
shared-weight closed-form ridge solve intact) beats the shipped v9 ridge. At every "reasonable"
threshold (`huber_k≥1.5`) the mechanism never actually engages — scores are bit-identical to v9,
confirming it's a no-op there, not a null result. Below `huber_k=1.0` it engages and hurts sharply
(rmean falls to 840-880). A narrow window at `huber_k∈[1.20,1.22]` technically clears the OLD+NEW+rmean
bar (rmean 895.3-895.7) — but its immediate neighbors do not (`1.18`: fails; `1.25`: fails badly,
n_worse=53/61), a spike roughly 0.02-0.05 wide surrounded by failures on both sides. Contrast with
v9's beta-demean result, which held across a plateau spanning `lam∈[0.4,0.8]` **and**
`BETA_DEMEAN_W∈[400,600]` simultaneously — this repo's own neighbor-stability convention treats an
isolated spike like this one as noise, not a finding, and it's discarded on that basis.

## Reduced-rank regression on the idio ridge ensemble — rejected, 0/14, clean and decisive
The idio ridge (`_ewls_ridge` in `SAFE_llboost_v8.py`) fits a 51×50 coefficient matrix `B` per
half-life (all 51 instruments' current returns → the 50 idio names' next-day returns) and shrinks
**every one of the 2550 coefficients uniformly toward zero** via a single scalar `RIDGE_A=0.1`. Since
this repo's own exhaustively-mapped finding is that the market is a one-factor model (ALGO/PC1 +
lead-lag + idiosyncratic noise), uniform shrinkage-to-zero looked like the wrong prior for a
genuinely low-rank relationship — reduced-rank regression (RRR) shrinks toward the correct low-rank
**subspace** instead. Confirmed via a fresh repo search this was genuinely untested here (the
existing "PCA pre-reduction" item in `test_q20_items01_04_ridge_variants.py` projects the *predictor*
panel onto top-K components before fitting — a different technique from constraining the *fitted*
coefficient matrix's rank).

**The estimator** (`test_v9cand_rrr.py`, derived and verified by hand before trusting it — do not
naively SVD-truncate raw `B`, and do not weight by unregularized `X'X`; both give a statistically
different, inferior estimator): given the existing fit's `S = XtWX + (eps+a)*I` and `B = S⁻¹XtWY`,
the ridge loss reduces by completing the square to `L(C) = const + tr((C−B)'S(C−B))` — minimizing
this subject to `rank(C) ≤ r` is a **weighted-by-S** low-rank approximation of `B`, with closed form
`C_r = B·V_r·V_r'`, `V_r` = top-r right singular vectors of `S^{1/2}B` (via Cholesky, numerically
stable). At `r=50` (=min(p,q)) this must reproduce `B`, and hence the shipped v8 baseline, exactly —
confirmed: the sweep's `r=50` row reproduces v8's docstring numbers (847.4/888.9/886.2/674.4) to the
decimal, so the implementation is trusted before looking at any smaller-rank result.

**Every rank tested loses, monotonically, with no exception:**

| rank | OLD | NEW | rmean | rfloor |
|---|---|---|---|---|
| 1 | 377.8 | 559.2 | 378.3 | 267.3 |
| 5 | 723.7 | 883.4 | 725.8 | 348.7 |
| 15 | 823.4 | 939.4 | 833.8 | 418.2 |
| 25 | 687.9 | 873.9 | 829.9 | 666.2 |
| 35 | 816.4 | 851.5 | 869.4 | 689.0 |
| **50 (full rank, = v8)** | **847.4** | **888.9** | **886.2** | **674.4** |

**0/14 ranks beat v8 on OLD+NEW+rmean jointly**, and the approach to the baseline as `r→50` is
monotonic with no local bump anywhere in between — the strongest possible form of this rejection:
full rank isn't just the best config tested, it's the top of a curve that never turns over. This
directly contradicts the working hypothesis: an honest re-check of the "second factor" evidence
(re-running `test_pc2_probe.py` rather than trusting a cached memory of it) found PC1 (~ALGO) genuinely
predictive (p=0%) but **PC2 clearly null (p=25%) and PC3 only borderline (p=9%)** — no second common
factor beyond ALGO. That's consistent with there being a real dominant direction, but says nothing
about whether the *other* 49 individually-weak, largely-independent stock-level predictive
relationships in `B` are compressible — and empirically, they are not: the ridge ensemble apparently
needs close to its full 50-dimensional response space to capture the real (if individually small)
signal spread across many stock pairs, not concentrated in a handful of common directions. The
4-half-life averaging the ensemble already does may also already be capturing whatever
noise-suppression benefit RRR could offer, the same way it's absorbed several other prior rejected
ridge/ensemble ideas (per-half-life `RIDGE_A`, trimmed-mean ensembling, extra half-lives).

No `RIDGE_A` confirmatory grid, no `SAFE_llboost_v9.py` — the primary sweep's rejection is too clean
and too monotonic to warrant it (the plan's own stopping condition: only run the confirmatory grid if
the rank sweep looks promising).

**A result this clean (no local bump anywhere) is exactly the shape a bug would also produce, so it
was independently audited before trusting it** (not kept as a file — a one-off correctness check, not
a candidate): (1) on synthetic data with a KNOWN rank-2 ground truth, the same machinery correctly
recovers low rank as the best out-of-sample fit, confirming it works on a problem designed for it;
(2) `B_r`'s actual numerical rank exactly matches the target rank at every test; (3) in-sample
weighted loss decreases monotonically with rank at every half-life, as the constrained-optimum
construction requires; (4) recomputing `B_r` via a totally different route (eigendecomposition of
`B'SB` instead of SVD of the Cholesky-whitened matrix) agrees with the original to ~1e-15. The
real-data score drop is much larger than the underlying regression-loss drop (rank 1 more than halves
the traded score but only costs ~5-10% in-sample loss) — consistent with, not contradicted by, this
book's full-conviction sign sizing: a fixed $10k stake pays for wrong signs in full, not a damped
fraction, so a modest accuracy loss shows up amplified in traded score. The rejection is genuine.

## Post-v6 test queue: 5 genuinely-untested hypotheses, 1 validated
Five structurally new (not re-runs of anything in the 80-idea queue above) hypotheses were tested
this session, each held to the same causal-only, 61-window rolling `n_worse` bar as everything
else in this file. Only the `COMBINE_GAIN` resweep above survived; the other four are documented
dead ends, same policy as every rejected idea in this file:
- **Self-adaptive `BOOST_K`** (`test_v7cand_adaptive_boostk.py`): scale the pairwise boost's
  strength by its own pooled trailing realized IC, mirroring the ALGO leg's validated adaptive-gain
  philosophy — never applied to the boost itself. Rejected: the boost's realized edge is too
  stable/uniform across this file (trailing IC mean 0.085, never meaningfully negative) for a
  trailing-performance gate to find a genuine regime to exploit — it just uniformly de-rates `K`
  and costs the NEW window (0/64 configs tested cleared v6 jointly).
- **Regime-adaptive lookback generalized beyond `MOM_LB`** (`test_v7cand_regime_lookbacks.py`):
  the vol-regime short/long switch validated for momentum (v2) applied to `REV_W`, `BOOST_IC_L`,
  and `IC_EW_W` instead. Rejected for all three — no combination clears v6 on OLD+NEW+rolling-mean
  jointly (the `IC_EW_W` version is additionally near-degenerate: only 4% of sampled days differ
  from a fixed window at all).
- **Pair-correlation trend as a boost confidence multiplier** (`test_v7cand_corr_trend_boost.py`):
  distinct from H3's existing leader-*identity*-stability gate — this tests whether a pair's
  correlation *magnitude* is currently strengthening or weakening. Rejected — no ratio- or
  gate-based design (across window/threshold sweeps) beats v6 on OLD+NEW+rolling-mean jointly.
- (`COMBINE_GAIN` resweep — validated, see `SAFE_llboost_v7.py` section above.)

## Porting the ALGO leg's double-IC veto to the idio book — rejected, 0/29
`_side()` in the ALGO leg takes its sign from a fast 90-day simple IC and then **refuses to trade
at all** unless a structurally different second estimator — the mean of two EW ICs at half-lives
(20, 45) over 200 days — agrees on the sign. Tested whether that same two-estimator-agreement veto
helps the idiosyncratic book (`test_v7cand_double_ic_idio.py`, 29 variants at three placements;
ALGO leg held identical throughout, so every difference is idio-side). Distinct from the
previously-rejected single-estimator trailing-IC ideas (adaptive `BOOST_K`, gated-pair-boost,
partial pooling, margin scaling) — the content here is *estimator disagreement* as a stand-down
signal, not an IC level or a strength dial. **Every variant lost; nothing came within 1 point of v7
(830.3 / 888.5 / 876.8 / 674.4).**

| placement | best variant | OLD | NEW | rmean | n_worse/61 |
|---|---|---|---|---|---|
| A — pair (confirm the boost's `ic>0` gate) | `A-fast(L=180)`, 99% of boosts kept | 830.3 | 887.6 | 875.9 | 15/61 |
| B — per-stock (literal `_side` port on `wz[j]`) | `B-veto(fast60)`, flat 12.8% of stock-days | 767.7 | 760.7 | 802.6 | 61/61 |
| C — whole book (pooled IC) | all 8 variants | 830.3 | 888.5 | 876.8 | 0/61 (**inert**) |

The mechanism, measured directly (`test_v7cand_double_ic_diag.py`) — the veto needs an edge whose
sign is genuinely unstable, and only ALGO has one:
- **ALGO vol feature**: fast IC mean +0.071 but **sd 0.101**, negative on 22% of days, crossing zero
  13 times. The veto fires on **5.4%** of days — it is catching real sign flips in a
  regime-dependent relationship, which is exactly why it earns its keep there.
- **Idio book, pooled**: fast IC mean +0.0675 with **sd 0.0255**, range `[+0.0096, +0.1185]` —
  **never negative on any of 704 days**, on either estimator. The two estimators disagree on **0**
  days, so the book-level gate is inert by construction. This is a different animal from ALGO's
  feature: the ridge refits daily and its edge is stably positive (consistent with the champion
  staying IC-positive even in a synthetic momentum regime).
- **Idio, per-stock**: ~90 observations per estimate gives SE ≈ 0.105 against a mean IC of +0.072 —
  signal-to-noise **0.69**. A single name's trailing IC cannot resolve its own sign, so the 17.7%
  disagreement rate is pure estimator noise; acting on it flattens ~13–20% of a full-conviction
  book at random (−74 rmean), and the sign-flipping version is a catastrophe (−400 rmean).
  Pooling across the 50 names is what raises signal-to-noise to 4.53 — and then there is nothing
  left to veto.
- At the pair level the veto can only *subtract* boosts, and subtraction is monotonically bad: 99%
  kept → −0.9 rmean, 94% → −2.8, 91% → −6.1, 85% → −10.0. Same conclusion as the adaptive-`BOOST_K`
  test — every pair surviving the shipped Bonferroni + `ic>0` + 480-day-history gates is worth
  trading, so a second opinion only removes good boosts.

**General rule this establishes:** a two-estimator agreement veto pays only where the edge's *sign*
is regime-dependent AND each estimate has enough data to resolve that sign. The idio book fails both
tests — its edge is stably positive, and per-name estimates are noise-dominated.

### Follow-up: is the EW (20, 45) estimator itself any good on the idio side? — rejected, 0/19
The test above only ever used `IC_EW_HL=(20,45)` as a *second opinion* (it could remove boosts,
never add one), which says nothing about the estimator's own quality. `test_v7cand_ew_idio.py` asks
the separate question — is exponential recency weighting at 20/45-day half-lives a better **primary**
estimator than the flat equal-weighted windows the idio book uses? Replacements can go both ways.
Four placements, 19 variants; **0 beat v7**, and the failure is *monotone in half-life* in all four:

| placement | shipped | fastest tried | ← rmean → | slowest tried |
|---|---|---|---|---|
| **E1** boost IC gate (leader selection untouched) | flat 250d = **876.8** | `ew(20,45)` 865.4 | `ew(45,90)` 869.3 · `ew(87)` 874.2 · `ew(125,250)` 875.4 | `ew(87,174)/500` **876.9** (+20/−0 boosts = 0.4% changed; a tie, not a pass) |
| **E2** leader-selection correlation (Bonferroni bar at n_eff) | flat, n=998 → **876.8** | HL=20, n_eff 58 → 789.2 | HL=45 782.3 · HL=90 799.4 · HL=250 826.3 | HL=500 864.0 |
| **E3** ridge half-lives | (250,1000,500,2000) = **876.8** | (20,45) only → **392.5** | +(20,45) → 806.5 | +45 → 849.1 |
| **E4** reversal leg | flat `REV_W=10` = **876.8** | HL=5 → 821.7 | HL=10 832.4 · HL=20 824.9 | HL=45 785.2 |

Every idio estimator wants **more** memory, not less, and 20/45 is off by roughly an order of
magnitude. The mechanism is the same one that sank the veto, seen from the other side: the ALGO
leg's IC is genuinely non-stationary (sd 0.101, negative 22% of days, 13 zero-crossings), so a fast
taper is tracking something real; the idio book's IC is stationary (pooled sd 0.0255, never negative
in 704 days), so a fast taper adds variance to estimate a constant. Corollaries worth keeping:
- The **COM-matched control** isolates taper shape from lookback length — `ew(87)` has the same
  centre of mass as v7's flat 250-window and still loses 2.6 rmean. Where the underlying quantity is
  stationary, a hard window is not just adequate, it is *better* than a taper of equal average age:
  equal weighting is the minimum-variance estimator of a constant.
- **E3 (20,45)-only collapsing to 392.5** is a sample-size wall, not a tuning miss: HL=20 leaves
  ~29 effective observations to fit a 50-predictor ridge.
- **E2 HL=500 posts NEW=905.9, the highest NEW anywhere in this file** — and OLD=797.8, rmean 864.0,
  n_worse 45/61. Textbook single-window artifact, same shape as the rejected Spearman boost pool
  (NEW 862.5 / OLD 755.4). Do not chase it.
- The reverse direction (E1 half-lives *longer* than shipped) flattens out at v7's number rather
  than beating it, so the flat 250-day gate sits at the top of the curve, not on a slope.

## 80-idea follow-up test queue: what else was tried and rejected
After v3/v4 shipped, a further 80-idea batch was tested across four categories (parameter/mechanism
refinements of the new boost pool, new signal features, portfolio-construction/sizing schemes,
ridge/ensemble variants + cheap checks) — see `test_batch80_*.py`. Besides the v5/v6 refinement
above, everything else was rejected, reinforcing patterns established earlier this session:
- **Sizing/smoothing schemes uniformly lose** (Kelly sizing, drawdown throttles, vol-targeting,
  rank-based sizing, confidence ramps, persistence bonuses) — full-conviction sign-based sizing
  keeps winning, consistent with every prior sizing idea rejected this session.
- **Ridge/ensemble variants uniformly lose** (per-half-life `RIDGE_A`, median/trimmed-mean
  ensembling, extra half-lives at 100 or 4000) — the existing 4-half-life simple-mean ensemble is
  a robust, hard-to-improve-on estimator.
- **Alternative boost-pool mechanics uniformly lose** relative to v5's clean profile (EWMA/short-
  window vol ranking, FDR correction, Spearman correlation, follower-side restriction) — some show
  an attractive number on one metric but always trade it off against another (e.g. Spearman:
  NEW=862.5 but OLD drops to 755.4, n_worse=45/61).
- **Two new signals cleared Stage 1 significance** (idio beta-to-ALGO stability, p=0.000; cross-
  sectional return dispersion, p=0.003) but both failed Stage 2: the beta-stability tilt degrades
  other metrics once it's large enough to matter, and dispersion turned out to predict the
  *average* next-day return across all stocks (a market-timing signal), not which stocks to
  prefer — a uniform per-stock tilt built from it has essentially zero effect on sign-based
  positions.

## Where the score actually goes — the v7 budget (read this before proposing anything)
`test_v7_leak_diagnostic.py` and `test_v7_algo_headroom.py` decompose the shipped book instead of
testing a candidate. Four measurements that should constrain every future idea:

**1. Score ≈ mean daily PnL. Variance is very nearly irrelevant.** v7 runs at annualised Sharpe
**7.5**, so `frac = sr²/(sr²+1) = 0.983` — the Sharpe penalty costs only ~$15/day. Elasticities at
this operating point: **+1% mean = +1.03% score, −1% stdev = +0.03% score. Mean is worth 30× a
variance cut of the same size.** This retroactively explains every rejected sizing/risk idea in this
file (Kelly, vol-targeting, drawdown throttles, cluster-neutral, confidence ramps): they all trade
mean for variance at a 30:1 disadvantageous exchange rate. *Do not propose risk control here.*

**2. The daily budget (NEW window, 750–1000):** idio gross **$726** (76%), ALGO gross **$225** (24%),
commission **−$47** (idio $46.3, ALGO $0.4), net $904 → score 888.5. Zero-commission score is 936.0,
so **any turnover-reduction idea whatsoever is capped at +47 (+5.2%)** — and only if it gives up
literally no edge.

**3. Both legs are already at maximum deployment.** ALGO runs at **95.1%** of its $100k cap and is
**never flat** (0 idle days in 500); the idio book is at the $10k cap on all 50 names every day. Under
a per-name dollar cap, sign-sizing is *provably* mean-optimal, so position construction is finished —
there is no capital-allocation or sizing idea left with positive expected value. Confirmed
empirically: ALWAYS-CAP on ALGO scores 869.4 (−7.4) and `COMBINE_GAIN` past 16 decays monotonically
(20 → 875.5, 25 → 874.0, 40 → 871.0, 100 → 868.5), so 16 is the top of the curve, not a slope.

**4. Per-dollar productivity is lopsided.** ALGO earns **23.6 bp/day** per deployed dollar against the
idio book's **13.6 bp** (1.74×), at **1/116th** the commission ($0.4 vs $46.3/day), with correlation
−0.07 between the legs. ALGO is the best capital in the book — it is simply capped by the rules at
$100k, and it is already full.

**Consequence: only three things can move the score at all** — idio sign accuracy (76% of gross),
ALGO sign accuracy (24% of gross, ~free to trade), and commission (hard ceiling +5.2%). Nothing else.

**Where adaptive machinery can pay.** Combining this with the double-IC findings above gives a rule
for the whole file: an adaptive/gating mechanism needs *both* non-stationarity to track *and* enough
data per estimate to resolve it. Measured on this data —

| | non-stationary? | data per estimate | adaptive machinery? |
|---|---|---|---|
| ALGO leg | **yes** — IC sd 0.101, negative 22% of days, 13 zero-crossings | 1 instrument, full history | **yes** — this is where `_side`, the vol/momentum switch and `COMBINE_GAIN` all paid |
| idio, pooled | sign-stable, low-variance (IC sd 0.0255, never negative in 704 days) — see caveat below | ~4500 obs, SNR 4.5 | nothing to adapt to |
| idio, per-name | unresolvable | ~90 obs, SNR **0.69** | can't resolve its own sign |

That is the whole reason the same gate works on ALGO and fails on the idio book, and it predicts
which future ideas are worth the compute.

**Formal check (`test_ic_stationarity_formal.py`), and a correction to the language above:** "non-
stationary?" was originally called from summary stats alone (mean/sd/negative-day-count). Running
actual ADF (H0: unit root) and KPSS (H0: stationary) confirms ALGO's IC is non-stationary — KPSS
rejects stationarity outright (p=0.01), ADF can't reject the unit root either (p=0.20), both point
the same way, consistent with the direct 22%-negative/13-crossing count. **Idio's pooled IC is
genuinely ambiguous, not confirmed stationary**: ADF fails to reject a unit root (p=0.81, expected)
but KPSS *also* fails to reject stationarity, only barely (p=0.089) — the two tests disagree rather
than agreeing, so "stationary" oversold it. Likely a power artifact, not a wrong call: the IC is a
90-day rolling estimate, so consecutive values share 89/90 of their underlying data, which mechanically
induces heavy serial correlation regardless of the true process and is known to starve ADF of power
against a highly persistent alternative — it hits both series, but only ALGO's swings are large enough
to read as non-stationary despite it. Doesn't change any conclusion above (those rest on the direct
sign-flip/disagreement counts, not on a unit-root classification) — only the label: idio's IC is
"empirically sign-stable over this sample," not "confirmed stationary."

## Lowering `BOOST_MIN_DAY` below 480 on the current v7 book — re-confirmed, still essential
The original `SAFE_llboost.py` docstring diagnosed why the 480/500-day minimum-history gate is
necessary against the *original* book (N=49 candidates, `IC_L=220`, `SCALE_W=500`). Since v3's
candidate-pool restriction (39 names) and v7's retuned `IC_L=250`/`SCALE_W=1000` both change how the
gate behaves, `test_v7_boost_min_day_200.py` re-ran the same question directly against the shipped
v7 book (monkey-patching the real `V7.BOOST_MIN_DAY` global so `_pairwise_boost`'s own gate fires —
no reimplementation) instead of assuming the old finding still applies.

**It does, unchanged.** OLD (500–750) and NEW (750–1000) score identically (830.3/888.5) at every
`min_day` from 150–500 — the grader's window starts after day 480 regardless, so unlocking earlier
days can only ever hurt the rolling floor, never help the graded score:

| min_day | rmean | rfloor | n_worse/61 |
|---|---|---|---|
| 480 (shipped) | 876.8 | **674.4** | — |
| 300 | 863.7 | 652.1 | 34/61 |
| 200 | 858.3 | 624.2 | 34/61 |
| 150 | 857.2 | 607.9 | 34/61 |

Worst single window (end_day=490): −61.4. Directly measured why: the Bonferroni bar shrinks from
`|corr|>0.227` at day 198 to `0.147` by day 478 (more samples → looser bar relative to noise), so the
earliest "significant" leaders are found on the thinnest, most permissive samples. An out-of-sample
check (next-60-day same-sign hit rate for every stock-day boosted only because of the lower gate,
never used inside the traded signal) confirms these are pure false positives: **49.6% — a coin flip**
— over 52,680 observations, 1055 stock-days, 15/50 names. Same mechanism as the original diagnosis
(Bonferroni controls false-discovery within one re-estimate, not across the ~15 sequential
re-estimates made walking forward); v3/v7's other refinements don't touch it. `BOOST_MIN_DAY=480` is
not a stale artifact of the old book — it is still exactly as load-bearing on the current one.

## Why Bonferroni alone isn't enough, and why the fix isn't a better correction (H3, re-confirmed)
`BOOST_MIN_DAY`'s docstring diagnosis and the `test_v7_boost_min_day_200.py` re-confirmation above
both point at the same gap: Bonferroni correctly controls the false-positive rate of the **39
simultaneous candidates on one day's search**, but says nothing about **repeated testing over time**
— the same search re-run day after day as the file walks forward. A false-positive leader found on a
thin sample doesn't just misfire once either: the correlation is estimated over a long trailing
window, so it barely moves day to day, and a lucky discovery tends to keep clearing the bar (and
trading with real size) for dozens of subsequent days.

The natural statistically-motivated fix — require a leader relationship to persist for N consecutive
days before trusting it, directly targeting the repeated-testing problem — was tested this session
(`test_h3_stage2_backtest.py`, "H3 leader-identity-stability") and **rejected**: every `min_stab`
threshold from 10–200 days *worsens* rmean monotonically (811.4 → 806.7 → 788.6 → 780.8), and,
tellingly, **the rolling floor doesn't move at all** (563.8 unchanged at every threshold) — so a
persistence gate doesn't even fix the specific failure mode it targets. A soft confidence-multiplier
version (scale the boost by a pair's stability instead of a hard cutoff) fares no better.

**Conclusion:** Bonferroni isn't the weak link — it's correctly solving the problem it was designed
for. The gap is an orthogonal one (sequential/repeated testing over time), and the one principled
attempt to patch it directly failed. The gate that actually works, `BOOST_MIN_DAY`, doesn't try to
statistically distinguish good early discoveries from bad ones at all — it just refuses to search
until the sample is already large enough that thin-sample false positives stop dominating, sidestepping
the high-risk region rather than correcting within it.

## Synthetic stress test: how much does the 250-day score vary from pure sampling luck?
`stress_test_synthetic.py` fits a one-factor market model + a parametric-bootstrap idiosyncratic
process (SAFE's own fitted ridge coefficients, assumed to be the true generator) + a calibrated ALGO
vol-continuation process to `prices.txt`, then draws 60 independent fresh 1000-day panels and runs
the *actual* `SAFE_llvol.getMyPosition` walk-forward on each, scored with the exact official
convention (sanity-checked to reproduce 761.78 on the real file first).

| | value |
|---|---|
| mean | 922.3 |
| median | 908.3 |
| std | 118.0 |
| p5 / p95 | 739.0 / 1110.4 |

The real file's actual score (761.78) sits at just the **8th percentile** of this distribution; 92%
of synthetic draws beat it, and 57% land above 900 (the range this session's discussion cited for the
top leaderboard teams). Read narrowly: this quantifies sampling variance *conditional on* our fitted
model being exactly the truth — a legitimate but circular-in-our-favor test, since the synthetic
world is built from our own coefficients and holds them fixed rather than also resampling the
estimation uncertainty in them. It does not prove the top teams aren't also genuinely better; it
shows the noise floor at this sample length is large enough that the gap alone isn't strong evidence
either way.

## Evals
`eval_safe.py` / `eval_swing.py` / `eval_qual.py` / `eval_llmatch.py` / `eval_llvol.py` / `eval_llmeta.py`
/ `eval_llboost.py` / `eval_llboost_v2.py` / `eval_llboost_v3.py` / `eval_llboost_v4.py` /
`eval_llboost_v5.py` / `eval_llboost_v6.py` / `eval_llboost_v7.py` — official accounting, last 250
days. Edit `MATCH_K` in `SAFE_llmatch.py`, `VOL_GAIN`/`IC_LOOKBACK` in `SAFE_llvol.py`, `META_L` in
`SAFE_llmeta.py`, `BOOST_K`/`BOOST_MIN_DAY` in `SAFE_llboost.py`, `MOM_LB_SHORT`/`MOM_LB_LONG` in
`SAFE_llboost_v2.py`, `BOOST_N_CANDIDATES` in `SAFE_llboost_v3.py`/`SAFE_llboost_v4.py`,
`BOOST_IC_L`/`BOOST_MIN_DAY` in `SAFE_llboost_v5.py`/`SAFE_llboost_v6.py`, or `COMBINE_GAIN` in
`SAFE_llboost_v7.py`, then rerun the matching eval.

## `MATCH_K` robustness (rolling 250-day, 61 draws)
| k | OLD 501–750 | NEW 751–1000 | rolling mean | rolling floor |
|---|---|---|---|---|
| 0 (leg off) | 585 | 586 | 651 | **493** |
| **1.0 (shipped)** | 564 | 600 | **657** | 482 |
| 1.5 | 539 | 608 | 655 | 471 |
| 2.0 | 512 | 618 | 649 | 447 |

k=1.0 = the 1:1 match (index takes exactly the book's predicted net-$ tilt): best rolling mean, high
floor. Higher k trades old-window score for new-window score and erodes the floor — pick k on the floor.
See panel 5 of `diagnostics.html`.

## ALGO as an extra pairwise-boost leader candidate — rejected, small and net-negative where it fires
`_pairwise_boost`'s leader search has only ever considered other idio names as candidates — ALGO is
excluded even though it already sits inside the linear ridge as one of 51 predictors. The ridge only
captures a *linear* ALGO→name relationship; the boost's own convex `sign(x)*(|x|/scale)^P` transform
(validated elsewhere for idio-vs-idio pairs, e.g. the DUCT→AMRP relationship) is a structurally
different hypothesis — a crash-beta/tail-sensitivity effect, untested here. Distinct from the
rejected v17 ALGO crossover (which tested ALGO's own price pattern predicting ALGO's *own* next
return, time-series); this tests ALGO's move predicting *other stocks'* next moves, cross-sectionally.

`test_v18cand_algo_leader_boost.py`: added ALGO unconditionally as a 40th candidate leader
(Bonferroni divisor 39→40 to match), every other boost mechanic identical. **Doesn't clear the bar**
(against real SAFE_llboost_v10, sanity-checked to reproduce 871.0/912.6/909.8/709.7 first):

| | OLD 501–750 | NEW 751–1000 | rolling mean | n_worse/61 |
|---|---|---|---|---|
| SAFE_llboost_v10 (baseline) | 871.0 | 912.6 | 909.8 | — |
| stricter-divisor-only (control: divisor 40, no ALGO) | 869.8 | 915.5 | 909.1 | 32/61 |
| ALGO-as-40th-leader | 869.8 | 912.2 | 908.6 | 41/61 |

The control isolates the mechanism cleanly: OLD is *identical* between the control and the ALGO
variant, meaning ALGO is never actually selected as a leader on that window at all — the entire OLD
move is just the one-extra-simultaneous-test Bonferroni tightening, nothing to do with ALGO. ALGO
*is* picked on ~10% of days from day 480+ (52/520), concentrated in the NEW window, and where it
fires it costs **−3.3** relative to the control (915.5 → 912.2) — mildly net-negative, not neutral.
**Conclusion:** ALGO's own move has no additional convex/nonlinear predictive value for other idio
names beyond what the linear ridge already extracts from it. Consistent with the standing finding
that the idio book's pooled IC is stable and already well-saturated (double-IC-veto section above) —
there was no obvious missing piece here to find.

(The stricter-divisor-only control's own OLD/NEW trade-off, +2.9 NEW for −1.2 OLD, is a separate,
very marginal side observation — not investigated further since it doesn't clear the joint bar either
and isn't the hypothesis under test.)

## Re-sweeping `BOOST_N_CANDIDATES` against the current best — still optimal at 39, but the
## neighbor-stability story that justified it no longer holds
`BOOST_N_CANDIDATES=39` (the pairwise boost's leader-pool size) was chosen in `SAFE_llboost_v3`/`v5`,
swept only against the original `SAFE_llboost` baseline — before the beta-adjusted ridge target (v9)
and the rank-stability blend (v10) existed. `_pairwise_boost` is unchanged code since v7 and operates
only on raw idio returns, independent of `wz`, so there was little mechanistic reason to expect the
optimum moved — but that's an assumption, not a result. `test_v19cand_boost_ncandidates.py` checks it
directly against `SAFE_llboost_v10` (sanity-checked to reproduce 871.0/912.6/909.8/709.7 exactly at
N=39 before trusting anything else): expensive precompute (ridge WZ, BLEND reversion, ALGO leg,
rank-stability signal) is cached once since none of it depends on N; only the boost itself is
recomputed per candidate value, swept N=15–50.

**Still the best value — 0/21 alternatives beat v10 jointly, N=39 is the only one with n_worse=0/61:**

| N | OLD 501–750 | NEW 751–1000 | rolling mean | n_worse/61 |
|---|---|---|---|---|
| 35 | 863.9 | 902.7 | 901.7 | 50/61 |
| 37 | 855.9 | 882.0 | 899.6 | 44/61 |
| 38 | 861.4 | 888.2 | 901.0 | 44/61 |
| **39 (shipped)** | **871.0** | **912.6** | **909.8** | **0/61** |
| 40 | 863.0 | 916.9 | 904.6 | 41/61 |
| 41 | 859.0 | 915.1 | 902.3 | 50/61 |
| 42 | 859.3 | 924.9 | 904.1 | 46/61 |

So the direct answer to "does changing the boost pool size help" is no — nothing in a 21-point sweep
from 15 to 50 improves on the shipped value.

The more important finding is what this reveals about the *shape* around that optimum. The original
v3 writeup described N=29→39 as "a genuine, monotonically improving region... unlike an isolated
spike," with N=35–38 "also solid (n_worse 7–16/61)" and flagged only N=40's cliff as a known,
accepted risk. Under the current v9/v10 baseline that plateau is gone: every neighbor from 35–42
(except 39 itself) now sits at n_worse 41–50/61 — clearly worse, not "also solid." N=39 is no longer
sitting on a gentle slope; it's an isolated spike surrounded by a cliff on both sides, the same
overfitting-shaped pattern this repo rejected the Huber candidate and the `IC_EW_W=150` spike for
elsewhere. It happens to be a spike at exactly the value already shipped, so there's nothing to change
— but the original justification for trusting 39 (a stable, structural plateau) has quietly stopped
being true as the rest of the book evolved around it, and should be read as a standing, unresolved risk
rather than a settled one. Mechanistically this tracks the book's full-conviction sign-based sizing:
adding or removing one marginal high-vol candidate from the leader pool occasionally changes *which*
name is selected as the most-significant leader for a given idio name on a given day, which can flip
that name's position sign outright — a small, continuous change in the candidate pool producing a
discrete P&L jump on the days it matters, rather than a smoothly-varying effect that would average out
across neighboring N.

## An idio-side analog to the ALGO min-conviction deadband — rejected, 0/14, clean and decisive
`SAFE_llboost_v8`'s ALGO deadband (hold yesterday's shares instead of resizing into a small,
uncertain-sign combine target) was validated because ALGO's IC is genuinely non-stationary and its
low-magnitude days were shown to be actually loss-making (−$81/day vs +$309/day elsewhere). The "v7
budget" diagnostic argued the same trick shouldn't work on the idio book — pooled idio IC is
sign-stable ("nothing to adapt to") and per-name IC is unresolvable (SNR 0.69) — but that was an
inference from aggregate IC statistics, never a direct test of this specific mechanism. Tested it
directly against `SAFE_llboost_v10` rather than trusting the inference (`test_v20cand_idio_deadband.py`,
sanity-checked to reproduce 871.0/912.6/909.8/709.7 exactly with the gate off).

Mechanism: per idio name per day, if `|wz_i|` falls under some fraction of that day's cross-sectional
mean `|wz|` (a near-coin-flip combine target for that one name), hold or flatten instead of resizing.
Two treatments (HOLD, FLATTEN) × 7 thresholds (0.05–0.50), gated off before day 480 (matching
`BOOST_MIN_DAY`):

| | OLD 501–750 | NEW 751–1000 | rolling mean | n_worse/61 |
|---|---|---|---|---|
| SAFE_llboost_v10 (baseline) | 871.0 | 912.6 | 909.8 | — |
| HOLD, thresh=0.05 (mildest) | 788.8 | 866.1 | 861.4 | 52/61 |
| FLATTEN, thresh=0.05 (mildest) | 800.6 | 865.2 | 868.0 | 50/61 |
| HOLD, thresh=0.25 | 614.1 | 822.9 | 758.7 | 52/61 |
| FLATTEN, thresh=0.50 (widest) | 680.7 | 722.9 | 766.0 | 52/61 |

**0/14 configurations beat v10**, and every metric degrades monotonically as the threshold widens —
even the mildest, narrowest gate (thresh=0.05, touching the fewest name-days) is already clearly
worse. This isn't a close call.

**Why, quantified the same way the ALGO writeup was:** at thresh=0.25 the gate touches 21.2% of
name-days (5,511 of 26,000, day 480+) — not a rare edge case. Split by the shipped book's own realized
$ PnL, low-conviction name-days earn **$13.79/name-day**; the rest earn **$16.36/name-day**. Both
solidly positive. This is the mechanistic difference from ALGO: ALGO's low-conviction days were a
genuinely different, loss-making regime (sign flipped, not just smaller). Idio's low-`|wz|` name-days
are just a slightly weaker slice of the *same* positive edge, not a different regime — there's no bad
subset to cut, so holding or flattening it only discards real edge for nothing in return. Directly
confirms, rather than just infers, the "pooled idio IC: nothing to adapt to" line in the v7 budget
table, and closes out the one gap in that table that hadn't been tested head-on.

## 100-idea sweep against the current best — 0/100 shipped, one caught red-handed as an isolated spike
Ran a workflow-orchestrated batch of 100 candidate ideas against `SAFE_llboost_v10`, split across ~20
parallel subagents in two passes (the first pass lost 60 ideas to BLAS thread-contention under heavy
concurrent load — those subagents backgrounded their scripts and returned a non-answer instead of
waiting; re-run with single-threaded BLAS and no backgrounding, all 60 completed cleanly on the
second attempt). Every idea followed this repo's standard convention: sanity-check reproduction of
v10's exact numbers (871.0/912.6/909.8/709.7) before trusting any variant, pass bar = beats v10 on
OLD+NEW+rolling-mean jointly, `n_worse`/61 reported. **Result: 0/100 beat v10.**

**The one Stage-1 "pass," caught by Stage 2:** a `BETA_DEMEAN_W` resweep (350-650, 5 points) found
W=550 clearing the joint bar (n_worse=8/61). A denser 23-point follow-up sweep (`test_batch100_A23_full.py`)
found only 2/23 points pass at all (W=550 and W=575), both flanked on *every* side by worse-than-shipped
neighbors — a jagged, non-monotonic response surface with no plateau, the same isolated-spike signature
already rejected for Huber and `IC_EW_W=150` earlier this session. **Rejected** — exactly the failure
mode the two-stage screen-then-verify process exists to catch.

**Closest legitimate near-misses (not shipped, but worth remembering if revisited):**
- **Two-hop transitive boost** (C43: if A leads B leads C, add A as a candidate leader for C): NEW+1.7,
  rmean+0.3, rfloor unchanged, n_worse=0/61 — missed only because OLD was an *exact tie*, not strictly
  beaten.
- **Ridge interaction terms** (D55: name×ALGO features) is a genuine wash (n_worse=36/61); D56
  (top-10 pairwise name×name features) gets the single largest NEW gain of the whole sweep
  (912.6→923.9) but clearly overfits OLD.
- **Regime-conditional `COMBINE_GAIN`** (E66) and a **learned per-name boost/rank-stability blend
  weight** (G84) both missed by a single leg (rmean by 1.8, or NEW by 1.3) — flagged by their own
  test agents as worth a denser follow-up sweep, not investigated further here.

**Other findings worth keeping, independent of the shipping bar:**
- **Statistically confirmed**: a paired bootstrap over trading days puts v10's total improvement over
  the original `SAFE_llboost` at +$90.33/day mean, 53.0% win rate, 95% CI [22.61, 158.66] — excludes
  zero, p=0.0095 (I92).
- **Strict walk-forward re-check** (I88): refitting `RIDGE_A`/`BOOST_K`/`RS_WEIGHT` using only days
  1-750 picks the exact combo already shipped, and forward-applying it reproduces the full-range
  numbers exactly — reassuring against overfitting on the tested grid (caveat: only 2 `RIDGE_A` values
  were tried in this quick check, not a full re-derivation).
- **Edge concentration is not a growing trend** (I91): contradicts the natural-sounding guess from
  earlier this session. v7→v8 (the ALGO deadband) is the lumpiest transition by far at 97.8%
  concentration in its top-10 days, versus v8→v9's 19.0% and v9→v10's 40.2% — mechanism-specific, not
  monotonically worsening.
- **N=39 re-confirmed as a genuine isolated optimum, not a fixable artifact** (J99): averaging the
  boost computed at N={35,39,43} to smooth over the spike found in the earlier `BOOST_N_CANDIDATES`
  re-sweep makes things *worse*, not better — the spike is real, not a smoothing opportunity.
- **18/50 idio names are structural "orphans"** (J98) — never selected as a boost leader or
  follower-recipient across the graded window — and they underperform ($7.24/name-day vs $20.63 for
  the rest). A real, unexploited structural fact, though no mechanism tested here successfully turned
  it into edge.
- **Adjacent-day flip-flops are common (22.8% of idio name-days) and net *profitable* as traded**
  (H87: +$262,579 vs a −$40,211 counterfactual hold-steady) — directly answers "would a same-direction
  settle filter help" with no, consistent with every other turnover-suppression idea this session.
- **Genuine secular drift detected but not exploitable** (J100): a linear day-index feature survives a
  vol-regime partial-correlation control (partial corr=0.785), but the naive mechanism built to trade
  it (a uniform book-wide tilt) fails jointly — robs OLD/rmean to sometimes pay NEW. A real lead,
  no working mechanism yet.
- **Cross-sectional dispersion tilt is a mathematically enforced no-op** (B38, sharper than the
  earlier "failed Stage 2" characterization): z-scoring a uniform, non-stock-differentiating vector
  collapses it to exactly zero at every weight tested.

**The other ~85 ideas** — parameter resweeps of every shipped constant (`RIDGE_A`, `HALF_LIVES`,
`BLEND`, `REV_W`, `VOL_WIN`, `VOL_Z`, `IC_FAST`, `SWITCH_GAIN`, `IC_EW_HL`, `IC_EW_W`, `MOM_LB_SHORT/LONG`,
`COMBINE_GAIN`, `DEADBAND_THRESH_FRAC/MIN_DAY`, `BOOST_K/IC_L/MIN_DAY/P/SCALE_W/ALPHA`,
`BETA_DEMEAN_LAM`, `RS_SHORT_W/LONG_W/WEIGHT`); re-tests of previously-rejected mechanisms against the
current baseline (Huber, RRR, predictor-shrink, Elastic Net, Kalman/RLS, signed boost, leader-stability,
cluster-neutral, confidence-ramp sizing, GBM confirm-gate, beta-to-ALGO stability, PC2/PC3); new
lead-lag variants (multi-leader averaging, cluster-restricted pools, distance correlation, graphical
lasso, Granger causality, partial correlation, lag-2, time-decay); new model classes (logistic
sign-classification, Student-t MLE, SVR, random forest, quantile regression, wavelet inputs,
hierarchical ridge); and several more signal/portfolio ideas (decile momentum, kurtosis, cointegration
pairs from `SAFE_combined.py`, continuous rank-stability, model-averaging v9/v10, flip cooldowns,
commission-rounding sensitivity) — **all rejected cleanly**, most by a wide margin, several with the
same "sharp isolated peak at the shipped value" shape recurring across nearly every parameter (RIDGE_A,
BLEND, REV_W, VOL_WIN, IC_FAST, BOOST_P, BOOST_K, RS_LONG_W all showed this). Full per-idea numbers
are in the `test_batch100_*.py` scripts and workflow journals; not reproduced here in full given the
volume.

**Bottom line: the book is currently sitting in a narrow, well-tuned local optimum on essentially
every axis tested.** Combined with the earlier finding that another team scores ~1030 on the same
window at similar variance, and that this gap is within one noise-floor standard deviation of pure
sampling luck (per the synthetic stress test), the honest read is that closing further ground likely
needs either a genuinely different signal family this 100-idea sweep didn't happen to hit, or accepting
that some of the visible gap may not be closable at all.

## A user-raised structural concern: no pooled detector, and a controlled change-point experiment
A user pointed out, correctly, that `SAFE_llboost_v10` has **no pooled detector or shutdown mechanism**:
the ridge ensemble re-learns purely from price history (equal-weight blend of 4 EWLS half-lives,
250/500/1000/2000), and the pairwise boost's leader-*validation* step (trailing-250-day IC) can
suppress a broken relationship but its leader-*selection* step uses a full, undecayed price history.
Unlike `SAFE_lldollar.py`/`SAFE_combined.py` in this same directory (which already carry a champion-
health validator + kill switch, see [[algothon-protection-stack]]), the llboost lineage never got that
treatment.

**Built a controlled change-point experiment to test this concretely** (`changepoint_synthetic.py` +
`test_v11_changepoint.py`): calibrated a synthetic continuation from real `prices.txt` (beta, idio
noise, ALGO's stochastic-vol process) with an EXPLICIT, KNOWN 20-pair lead-lag structure (rho=0.25)
running for 1000 days, then broke it at day 1001 two ways -- **reverse** (same pairs, sign flipped)
and **rotate** (same followers, genuinely new random leaders) -- and ran the ACTUAL
`SAFE_llboost_v10.getMyPosition` walk-forward across both, checked across 4 seeds each:

- The ridge ensemble is **more adaptive than the raw half-life-weight arithmetic implies** (a
  re-fit weighted regression each day, not a frozen blend): it flips sign and holds within
  **~15-40 days** after a clean reversal. Adapting to a genuinely NEW relationship (rotate) is
  slower and less reliable -- some seeds never cleanly recovered within the 600 days tested.
- The boost's trailing-250-day IC gate **suppresses** a broken pair reliably (18-20/20 tracked
  pairs, within 600 days) -- but its full-history candidate-**selection** step essentially never
  finds the new correct leader within 600 days (0-1/20 pairs, across every seed tested) -- confirmed
  exactly as the user suspected: linear (sample-count) dilution, not exponential decay, so a new
  regime needs a comparable day-count to the ENTIRE pre-change history before it can win an
  unweighted argmax.

**CORRECTION (found while stress-testing the fix below):** the PnL/oracle numbers first reported
here were computed with an off-by-one indexing bug in the test harness (`walk_pos_idio` stored each
day's position one column earlier than the scoring loop expects) -- verified concretely: real
`prices.txt` only reproduces the documented 871.0/912.6 with the corrected indexing; the buggy
version silently produced deeply negative garbage scores that happened to still look directionally
plausible. **Re-run with corrected indexing** (`test_v11_changepoint.py`, current version):
- Reverse: the idio-only book DOES lose real money post-change, ~$180k-$280k cumulative over the
  post-change window (smaller than first reported, but still a genuine loss); oracle gap ~$680k-$840k.
- **Rotate: the idio-only book stays net POSITIVE post-change (+$15k to +$121k), not negative.** The
  ~$360k-$400k gap to the oracle is pure opportunity cost (the book still trades profitably, just
  worse than an unrealistic perfect-foresight comparison) -- not an active loss the way reverse is.
  This is a materially different, more nuanced picture than "the book loses $400k-$1.1M in both
  scenarios" as first stated, and changes what kind of fix rotate-mode actually needs (see below).

## SAFE_llboost_v11.py: an idio kill switch, and an honest account of what it does and doesn't fix
Built `SAFE_llboost_v11.py` = v10 + a kill switch on the final traded idio signal, in response to the
change-point finding above. Two trigger designs were tried, not just the first idea kept:

1. **First attempt**: ported `SAFE_lldollar.py`'s `_kill` verbatim (IC t-stat < -3.0, sustained 10
   consecutive days, ROT_W=60). Verified safe (0/904 real-data false-positive kill days) but weak.
   Diagnosis: a rotation degrades the old relationship to near-zero NOISE, not a confidently negative
   IC -- a significance test structurally can't catch "edge is gone", only "edge is actively hostile".
2. **Adopted instead**: a trailing-summed realized-PnL-sign trigger (ROT_W=60, KILL_P=1, no
   persistence delay -- flatten today if the trailing 60-day sign(forecast)·realized-return sum is
   negative, re-evaluated fresh every day). This directly reuses the lesson already shipped in
   `algopart2/SAFE_rotate.py` + `SAFE_live.py` (see [[algothon-protection-stack]]: "a more sensitive
   switch than the old IC-significance gate... captures a real regime far faster"). A narrower
   ROT_W=40 was also tried and rejected -- it introduces 7 real-data false-positive kill days, unlike
   ROT_W=60's clean 0.

**Validated** (`test_v11_changepoint.py`): real prices.txt (1000 days) -- byte-identical positions to
v10, 0/904 kill days. **Numbers below are corrected after the indexing-bug fix above** (superseding
an earlier, wrong report of 34-42%/3-23%):

| | reverse (sign flip) | rotate (new leader) |
|---|---|---|
| frac. of transition loss recovered, per seed | 22.5-28.6% | -5.5% to +1.7% |
| mean across 4 seeds | ~25% | ~-1.5% |

**Honest reading, corrected and less flattering than first reported:** the kill switch is a real,
validated improvement for the reverse (actively-wrong-signed) failure mode -- recovers roughly a
quarter of that loss, cleanly, on real data. **For the rotate (edge-decays-to-noise) failure mode it
is a wash and occasionally mildly harmful** (2 of 4 seeds: v11 worse than plain v10) -- because, per
the correction above, v10 alone isn't actually losing money in that scenario; a PnL-sign trigger
flattens on trailing-negative windows that occur even in a still-net-profitable book, forfeiting real
edge without preventing a real loss. This makes sense mechanistically once you see the correction:
a defensive flatten is the right tool for "actively hostile," not for "quietly underperforming its
own potential." It does **not** address the boost's slow leader-reselection at all -- that remains
open, and is the more appropriate fix for the rotate scenario specifically (see below). Only one
synthetic calibration (rho=0.25, 20 pairs) was checked. Numbers can vary by a couple percentage
points run-to-run (floating-point/BLAS-threading noise, already noted elsewhere in this file) but the
qualitative reverse-helps/rotate-doesn't split reproduced consistently across both the original and
corrected runs.

## Stress-testing the boost's slow leader-reselection, and a candidate fix with a real tradeoff
Two follow-up questions, given the rotate-mode finding above shows the kill switch isn't the right
tool there: (1) is the reselection gap actually as bad as it looked, or does it resolve given more
runway? (2) if real, can the candidate-selection mechanism itself be fixed?

**(1) Severity, confirmed via extended runway.** The 600-day window used above wasn't long enough to
see the eventual outcome. Extending to up to 4000 post-change days (cheap: just the raw correlation
dynamics, no ridge/boost machinery needed) shows the new true leader's full-history correlation
*does* eventually overtake the old one for all 20/20 tracked pairs in both seeds tested -- but the
median crossover lands around **~1000-1200 days post-change**, with a long tail out to 3000-4000,
matching the dilution math almost exactly (a new regime needs a day-count comparable to the ENTIRE
pre-change history, here 1000 days, before an unweighted full-sample argmax can flip). For any
tournament window in the 500-2000 day range, this is effectively "too slow to matter" even though it
technically isn't "never" -- confirming the severity, not just the existence, of the gap.

**(2) A fix was tried: exponentially-decayed candidate-selection correlation** (same half-life
weighting style as the ridge ensemble's own `_ewls_ridge`, applied to the boost's leader-selection
step only -- the trailing-250-day validation/suppression gate is untouched). A hard trailing WINDOW
was tried first and rejected outright: real-data performance degrades monotonically below ~750 days
(e.g. rmean 742→670 at a 250-day window) -- a hard cutoff throws away exactly the long-run evidence
that makes genuine, stable relationships significant in the first place. Exponential decay is a
meaningfully better mechanism -- it doesn't discard old evidence, just downweights it -- and:
- **Reselection speed improves substantially.** At half-life=1000 (matching the ridge ensemble's own
  longest half-life), median reselection time drops to roughly **~700-900 days** (from ~1100+ for the
  undecayed baseline) and 1-2 more of the 20 tracked pairs resolve within any given horizon.
  Half-life=500 is faster still (~400-500 day median) but costs more real-data quality (below).
- **But real-data cost is NOT free, and the initial 3-point summary (OLD/NEW/rmean) hid it.** Checked
  against this file's own `n_worse`-of-61-rolling-windows bar: half-life=2000 shows OLD/NEW/rmean all
  at-or-above the undecayed baseline (724.6/693.0/745.6 vs 717.4/688.1/742.0) *but* **14/61 rolling
  windows are worse**; half-life=1000 similarly has better-looking averages but **27/61 windows
  worse** (a near coin-flip); half-life=750 is clearly worse (46/61). None of this shows up in the
  averaged OLD/NEW/rmean view -- it only appears once every rolling window is checked individually,
  the same trap this file has flagged before ([[ic-vs-score-lesson]]-adjacent: aggregate metrics can
  mask real per-window inconsistency).

**Not shipped -- this is a genuine tradeoff, not a clean win, and is left as an open decision
rather than a unilateral call:** faster rotate-mode recovery is real and substantial, but it comes
with meaningfully more real-data rolling-window volatility than anything else shipped in this file
(every prior version's headline change was either `n_worse=0/61` or, at worst, a minority of windows
worse with every other metric improving -- 14-27/61 is a different, larger scale of inconsistency).
Whether that trade is worth taking depends on how much weight to put on a tail-risk scenario (a
genuine rotation-type regime break) that hasn't been observed in the real data at all, versus a
known, immediate cost to the already-validated real-data score's consistency. (`SAFE_llboost_v12.py`
was later taken by an unrelated, independently validated change -- v11's kill switch + a post-jump
fade, see below -- so if this decayed-reselection idea is picked up later, it should ship as `v13`
or later, not `v12`.)

## A second, independent ~50-idea signal search — one validated, shipped as `SAFE_llboost_v12.py`
Run in parallel with (and without visibility into) the 100-idea sweep above -- same spirit, different
organization: ~50 hand-designed hypotheses grouped into 7 batches by family, each swept through a
shared, verified backtest harness (`_v10_harness.py`, asserted on import to reproduce v10's exact
871.0/912.6/909.8/709.7) before anything was trusted. Bar and convention identical to the rest of this
file (beats v10 on OLD+NEW+rolling-mean jointly, `n_worse`/61 reported).

**Rejected, batch by batch:**
- **Leader-*identity* extensions** (two-hop lead-lag, chain-length confidence multiplier, mutual/
  reciprocal leader pairs -- 8 configs): 0/8. Notable: requiring a leader relationship to be mutual
  AND independently Bonferroni-significant in both directions extinguishes literally every candidate
  pair in this data (0/26,000) -- "mutual-only" degenerates to the boost-off ablation.
- **Peer-*aggregation* extensions** (weighted top-3 multi-leader blend, peer-consensus broadcast,
  leader-surprise, correlation-cluster momentum -- 10 configs): 0/10. Near-miss: the multi-leader
  blend improved OLD and rmean with n_worse=12/61, but NEW dropped too far to pass -- diluting the
  boost across several leaders trades NEW-period edge for OLD-period robustness.
- **Alternative leader-*selection* dependence metrics** (asymmetric-by-direction, decayed multi-day,
  Granger-style, split-sample validation, distance correlation, mutual information, Kendall's tau,
  tail-dependence, partial-correlation-vs-ALGO -- 9 configs): 0/9, none close. Plain Pearson
  correlation (shipped) beats every alternative tried; mutual information is the clear worst
  (NEW collapses to 580.7).
- **Own-series lag/autocorrelation structure** (point autocorrelation at lags 2/3/5/7, a genuine
  **VAR(2) ridge extension** -- doubling the predictor set to include lag-2 returns, streak length,
  5-day acceleration, a variance-ratio reversion-strength gate, an AR(1)-half-life sizing gate -- 27
  configs): 0/27. The VAR(2) result is a clean, structurally meaningful negative: adding lag-2
  predictors makes the *raw* ridge forecast markedly worse (rmean 536.0 vs the lag-1-only 746.8,
  before boost/reversal/rank-stability are even added) -- doubling the parameter count without enough
  data to estimate it hurts, not just fails to help.
- **Higher-moment and cross-sectional relative-value signals** (kurtosis, vol-of-vol, post-jump drift
  as a continuous z-score, tail-co-exceedance, forecast-self-relative z, correlation-graph eigenvector
  centrality, an ordinal-rank version of rank-stability, volatility dispersion, same-day consensus
  deviation, a multi-horizon rank-averaged momentum composite -- 10 ideas, several screened out at a
  cheap raw-IC check before backtesting): 0/10. `forecast-self-relative-z` had a genuinely significant
  raw IC (+0.0232, p<1e-6) but still lost the backtest -- another confirmation of this file's own
  30:1 mean-vs-variance elasticity finding. Centrality-as-boost-multiplier was the closest miss
  (n_worse=0/61) but didn't clear OLD/rmean strictly.
- **Rank-stability mechanism variants + ALGO-regime cross-leg coupling** (triple-timeframe
  confirmation, trading the trend-*agreement* case as momentum instead of only ever fading
  disagreement, severity-scaled fade, a residual-based crossover; `BOOST_K`/`RS_WEIGHT` scaled by
  ALGO's own vol regime, an ALGO fast-IC sign bias nudging idio `wz`, idio's own pooled IC fed back
  into the ALGO leg's gain -- 88 configs): 0/88. The idio→ALGO reverse-coupling's "switch" placement
  turned out to be inert by construction (`SWITCH_GAIN`'s branch is dead code once `VOL_COMBINE=True`
  reliably supplies a momentum sub-signal); the ALGO-sign-bias idea fails for a clean mechanistic
  reason -- it directly fights the beta-demean step, which exists specifically to strip common-mode
  market exposure back out of the idio target.
- **Volatility/event-conditional signals and confirmatory diagnostics**: a static per-name
  vol-tercile trading filter (decisively rejected, rmean as low as 301 when isolating the bottom
  tercile -- concentrating by name-level vol destroys diversification value); a boost-leader-vol-level
  split (diagnostic only -- next-day hit-rate is flat at ~52% across low/mid/high leader-vol
  terciles, no exploitable split); up/downside volatility asymmetry (rejected at every weight); a
  multi-name co-crash trigger (only 34 qualifying response-days fall in the graded window -- too rare
  to matter, and the one response tried was net negative anyway); a boost/rank-stability sign-
  disagreement damping gate (disagreement is only 3.2% of eligible stock-days here, vs. 48% for the
  analogous, already-rejected ALGO-side `sig`/`msig` gate -- genuinely much rarer, but still net
  negative); a half-life re-verification (confirmed `HALF_LIVES=(250,500,1000,2000)` is still exactly
  optimal, nothing drifted); a day-of-week check (no signal, as expected on a synthetic panel).

**The one validated idea: a post-jump fixed-size fade.** On any idio name whose most recent daily
return exceeds `FADE_K_SIGMA=2.0` times its own trailing `FADE_W=40`-day realized stdev (computed
strictly *before* that return -- fully causal), add a fixed-size fade against the move,
`FADE_EXTRA_W=0.06 * (-sign(that return)) * mean(|wz|)` that day -- a discrete, event-triggered
overlay, distinct from the existing *continuous* 10-day reversal leg (`BLEND=0.3, REV_W=10`). Fires
on ~5% of all name-days -- broadly distributed, not a handful of lucky days. **56/140 neighbor
configs** in a `W∈{30..50} × K_SIGMA∈{1.75..2.5} × EXTRA_W∈{0.02..0.08}` grid clear the strict bar --
a genuine plateau, the same standard this file already holds every other shipped mechanism to (and
the opposite shape from the `BETA_DEMEAN_W=550` isolated spike caught above).

| | OLD 501-750 | NEW 751-1000 | rolling mean | rolling floor | n_worse/61 |
|---|---|---|---|---|---|
| SAFE_llboost_v10 (real `getMyPosition`) | 871.0 | 912.6 | 909.8 | 709.7 | -- |
| **+ post-jump fade only** | **885.8** | **913.8** | **917.3** | **720.7** | **0/61** |

Confirmed through the *real*, sequential `getMyPosition` walk-forward (`validate_postjumpfade_full.py`),
not just the sweep -- numbers match the harness to the decimal. Positions change on only 40/852
graded-eligible days; NEW-window commission moves by +$12 on ~$11,624 (negligible). Honest caveat:
the *graded-window* (NEW) gain is modest, +1.2 -- the larger gains are OLD (+14.8) and the rolling
floor (+11.0). Real and useful, not a blowout.

**Shipped as `SAFE_llboost_v12.py` = `SAFE_llboost_v11.py`'s kill switch + this fade, combined.** The
fade is wired into `v11`'s own `_idio_signal` helper (the factored-out "full traded idio forecast"),
immediately after the rank-stability blend -- so the kill switch's trailing-PnL trigger automatically
evaluates the PnL of the signal *including* the fade, no separate wiring needed. On real
`prices.txt` the two mechanisms don't interact at all: v11's kill switch already fires 0/904 days
(confirmed again here), so v12 == "v10 + fade" exactly, to the decimal, on every metric
(`validate_llboost_v12_full.py`). The two remain genuinely orthogonal mechanisms -- one watches
trailing realized PnL sign at the whole-book level, the other watches same-day per-name return
magnitude -- and combining them doesn't change either one's own documented properties (the kill
switch's reverse-helps/rotate-doesn't split from the section above is unaffected; it was never
re-validated under the synthetic change-point experiment with the fade present, only confirmed to
compose cleanly on real data).

**Everything else** (parameter resweeps and mechanism variants not itemized above -- several dozen
more configs across the same 7 batches) was rejected the same way, most by a wide margin. Full
per-idea numbers are in `test_batch100_catA_leader_identity.py`, `test_batch100_catA_peer_aggregation.py`,
`test_batch100_catB_altmetrics.py`, `test_batch100_catC_lag_structure.py`,
`test_batch100_catDE_moments_relval.py`, `test_batch100_catFG_rankstab_regime.py`, and
`test_batch100_catHIJK_vol_event_misc.py`.

## SAFE_llboost_v13.py and v14.py: gated boost fallback + momentum/xsac insurance -- a mixed result, NOT a clean ship
Two more files built on top of v11's kill switch (2026-07-29): `SAFE_llboost_v13.py` adds a GATED
fallback to `_pairwise_boost` -- for any follower whose full-history candidate-selection path
contributes zero that day, try an exponentially-decayed candidate search (`BOOST_SEL_FALLBACK_HL=1000`)
before giving up, targeting the boost's slow full-history leader-reselection gap identified in v11's
own change-point work above. `SAFE_llboost_v14.py` = v11's kill switch + v12's post-jump fade + v13's
gated boost fallback merged, PLUS a NEW momentum/xsac insurance layer ported from
`SAFE_lldollar.py`/`SAFE_rotate.py`'s `_pick_at`/`_choose`/`xsac` validator (PnL-sum "champ sick" check
+ `FALLBACKS=(mom, momJT, residMom)`, no `tsrev`/`pairs` per the already-established negative results
in [[algothon-protection-stack]]).

**Real prices.txt (`compute_diagnostics.py`, OLD 501-750 / NEW 751-1000):**

| | OLD | NEW | rmean (61 windows) | rfloor | n_worse/61 vs v11 |
|---|---|---|---|---|---|
| v10 | 871.0 | 912.6 | -- | -- | -- |
| v11 (kill switch) | 871.0 | 912.6 | 909.8 | 709.7 | -- (identical to v10, 0/904 fires) |
| v12 (+ fade, shipped) | 885.8 | 913.8 | -- | -- | -- |
| **v13 (+ gated boost fallback)** | 872.7 | 920.3 | **910.6** | 709.7 | **25/61 worse, 27/61 better** |
| **v14 (v11+v12+v13 + insurance)** | **887.4** | **921.5** | -- | -- | -- |

**v13 does NOT clear this repo's own bar, despite attractive headline numbers.** The fallback engages
on 28/904 real days (3.1% -- NOT the "never triggers" case the file's own docstring flagged as the
easy, provably-safe outcome), and OLD/NEW/rmean all nominally beat v11 -- but **n_worse=25/61,
n_better=27/61 is a coin flip**, the identical failure signature already rejected twice earlier in
this file (the `BETA_DEMEAN_W=550` isolated spike, and the exponentially-decayed candidate-selection
idea at 14-27/61 worse): an aggregate-metric improvement that dissolves into noise-level
per-window inconsistency. **Rejected as a clean win** by this file's own standard, even though nothing
here is a bug -- it's exactly the trap the two-stage screen-then-verify convention exists to catch.

v14's real-data numbers are the best of all 14 versions (887.4/921.5), and combine v12's and v13's
real-data deltas almost exactly additively (871.0+14.8+1.7=887.5≈887.4; 912.6+1.2+7.7=921.5 exactly)
-- confirming the momentum/xsac insurance layer is silent on real data as designed (validator picks
`champ` every day), and that v12's fade and v13's boost fallback don't interact. But v14 inherits
v13's coin-flip rolling-window problem wholesale (same underlying mechanism), so it inherits the same
"not a clean win" verdict on real data.

**Change-point experiment (`changepoint_synthetic.py`, reverse=sign-flip / rotate=leader-reassignment,
4 seeds, frac. of the transition's oracle-gap PnL recovered vs plain v10):**

| | reverse (mean) | rotate (mean, v13/v14's actual target) |
|---|---|---|
| v11 (kill switch only) | 25.1% | -1.5% |
| v13 (+ gated boost fallback) | 25.0% (wash vs v11, not its target) | -0.7% (nudges positive, stays net negative) |
| v14 (+ momentum/xsac insurance) | **28.0%** | -0.2% (still net negative) |

Rotate is v13's actual target scenario (slow full-history leader-reselection) and it's still a wash
there for both v13 and v14 -- a small, consistent-direction nudge (every v14 seed but one improves
over v13), never flipping to a net positive. Neither file fixes the rotate-mode gap identified in
v11's own change-point section above.

**The reverse-mode number is a real surprise, investigated rather than taken at face value**
(`test_v14_reverse_mechanism.py`): v14's docstring explicitly pre-registered an expectation that the
momentum/xsac insurance layer should be a WASH on this harness (a plain-numpy pre-check found
mom/momJT/residMom statistically indistinguishable from noise against this exact pairwise-break
generator) -- but `test_v14_changepoint.py` showed the layer picking a non-champ fallback on 39-47%
of post-change reverse-mode days, clearly not inert. Built a counterfactual that is IDENTICAL to
v14's actual traded position on every day except days where `_choose` picked a fallback and `_kill`
hadn't already flattened it -- on those specific days only, flatten instead of trading the fallback,
isolating "the fallback signal has real edge" from "any departure from a known-bad champion helps."

| seed | v14 actual | flatten-fallback-only counterfactual | delta |
|---|---|---|---|
| 123 | -9,460 | -40,045 | +30,585 |
| 124 | -20,079 | -18,632 | -1,447 |
| 125 | -15,383 | -39,070 | +23,687 |
| 126 | 1,453 | -37,604 | +39,057 |
| **mean** | **-10,868** | **-33,838** | **+22,970** |

3/4 seeds show the traded fallback beating flatten-only by tens of thousands of dollars; the 4th is
roughly neutral (noise-level relative to the ~500k oracle scale). **Conclusion: the reverse-mode gain
is real, not a mirage** -- the fallback signals DO carry usable edge here, contradicting the earlier
unconditional pre-check. Mechanism: `_choose`/`_kill` only ever trade a fallback once the PnL-sum/xsac
detector has confirmed champ is sick, which apparently selects for exactly the windows where the
fallback edge is live -- an unconditional IC/PnL probe over the whole post-change period dilutes that
same edge with noise from periods where it isn't needed. **Same family of trap as
[[ic-vs-score-lesson]]** and the leader/follower pair-signal-IC re-derivation earlier in this
directory's memory: the conditional quantity a mechanism actually trades can look completely
different from an unconditional probe of the "same" relationship.

**Trend-regime test** (`test_v14_trend_regime.py`, momentum/flip/noise injection ported from
algopart2/`stress_momentum.py`, idio-only cumulative PnL over a 150-day injected window, v10 vs v14 vs
a pure-cross-sectional-momentum reference book):

| regime (injected-window lag-1 xsectional autocorr) | v10 (none) | v14 (insurance) | pure_mom (upper bound) |
|---|---|---|---|
| momentum (+0.372) | 210,281 | **672,286** | 708,939 |
| flip/whipsaw (+0.095) | 193,893 | **144,715** | 115,648 |
| noise (-0.014) | -16,174 | -16,161 | 8,847 |

**Momentum: a real, large win** -- v14 captures ~95% of the theoretical upper bound (672k vs 709k),
massively ahead of doing nothing (210k); switches off champ 140/150 days, 0 kill days. **Flip/whipsaw:
a genuine cost, not a wash** -- v14 is WORSE than plain v10 (144,715 vs 193,893, giving up ~$49k)
despite beating the naive pure-momentum reference; this is the honest tradeoff the file's own docstring
flagged to watch for. **Noise: statistically flat** between v10/v14, but the kill switch fires 43/150
days here -- far above the 0/904 real-data bar this repo holds every gate to (synthetic-only regime,
but a real flap-rate signal, not nothing).

**Flap rate** (state transitions between {champ, mom, momJT, residMom} x {killed, not-killed} per
day, `test_v14_changepoint.py`): 35-55/599 days in reverse mode (6-9%), 8-36/599 in rotate mode
(1-6%) -- the tension the docstring flagged between `_choose`'s ROT_P=5-day persistence and `_kill`'s
no-persistence trigger is real, not negligible, though it didn't visibly hurt the reverse-mode PnL
result above.

**Bottom line: neither v13 nor v14 is a clean ship by this file's own established bar.** v13's
real-data gain is coin-flip-inconsistent (25/61 worse) -- the same rejected failure mode as two
earlier ideas in this file. v14 adds a genuine, now mechanistically-confirmed momentum-regime win and
a real (not spurious) reverse-mode improvement, but carries a real whipsaw-regime cost, an unresolved
rotate-mode wash, and a non-trivial flap rate. Not rejected outright either -- the momentum-regime and
reverse-mode results are real, validated gains against threats this repo hadn't specifically tested for
the llboost lineage before ([[algothon-protection-stack]]'s protection stack was built for the
lldollar/rotate lineage only). Left as an open decision rather than shipped unilaterally, consistent
with how v11's rotate-mode decayed-reselection tradeoff was handled above: real upside on one axis,
real cost on another, no single clean number to point to.

## SAFE_llboost_v15.py: v12 + the momentum/xsac insurance layer, WITHOUT v13 -- the strongest validated candidate so far
Since v13's real-data coin-flip problem is its own mechanism (not an interaction with the insurance
layer -- v14's real-data numbers combine v12's and v13's deltas exactly additively, see above), built
`SAFE_llboost_v15.py` = `SAFE_llboost_v12.py`'s exact ridge+plain-boost+fade, PLUS v14's Part B
momentum/xsac insurance layer verbatim, DELIBERATELY dropping v13's gated decayed-selection boost
fallback entirely. Hypothesis: keep the insurance layer's two genuine wins (momentum-regime survival,
the now-confirmed-real reverse-mode edge) without inheriting v13's real-data inconsistency. Validated
(`test_v15_insurance_only.py`) against v10/v12 (recorded) and v14 (recorded):

| check | v15 result | vs v14 (recorded) |
|---|---|---|
| Real data (OLD/NEW/rmean/rfloor) | 885.8 / 913.8 / 917.3 / 720.7 | byte-identical to v12: **0/904 days differ, n_worse=0/61** |
| Change-point reverse (mean frac saved) | **28.2%** | v14: 28.0% -- unchanged |
| Change-point rotate (mean frac saved) | -0.9% | v14: -0.2% -- slightly worse (missing v13's ~1%-of-days contribution), still in the same near-zero bucket as v11/v13/v14 |
| Trend-regime momentum (150d) | 672,934 | v14: 672,286 -- unchanged |
| Trend-regime flip/whipsaw (150d) | 145,981 | v14: 144,715 -- unchanged, same known cost |
| Trend-regime noise (150d) | -22,234 | v14: -16,161 -- **worse** |

**The hypothesis held on the two axes that mattered most.** Real data is byte-identical to v12 --
this is the clean win v13 never achieved (n_worse=0/61, not 25/61), inherited for free rather than
re-swept. The reverse-mode gain (28.2% vs v14's 28.0%) is fully preserved, confirming (independently
of the counterfactual analysis above) that it comes entirely from the insurance layer, not v13's
fallback. The momentum-regime win and the whipsaw cost both reproduce unchanged, since neither
involves the pairwise-boost mechanism v15 removed.

**One real, if minor, new finding: the noise-regime result is worse (-22,234 vs v14's -16,161), not
just noise** -- same seed, same generator, so the difference is mechanistic: the champion's own PN
series shifts slightly on the ~1% of days where v13's fallback would have engaged, which perturbs the
path-dependent kill-switch trigger timing (recall the kill switch is a threshold on a trailing PnL
sum, re-evaluated fresh every day with no persistence) enough to land on a worse day in this
already-flagged-fragile scenario (v14 itself over-fires its kill switch 43/150 days here, far above
the real-data 0/904 bar). Not a new failure mode -- a different realization of the one already known.

**Verdict: the best-validated candidate from this session.** Real data exactly as safe as the
already-shipped v12 (not a re-sweep, a mathematical consequence of dropping the one problematic
piece), plus both of v14's genuine synthetic-scenario wins, at the cost of the same two tradeoffs
already documented above (whipsaw cost, rotate-mode still open) and a marginally worse noise-regime
number. Not shipped as SAFE_llboost's default here -- still a decision about how much weight to put on
protecting against regime types (momentum, edge-decay) that have never been observed in this repo's
real data, same framing as the rest of this section.

## Two follow-up investigations into what's still NOT working
**Why does the rotate-mode gap stay open across v11/v13/v14/v15?** Instrumented `_pairwise_boost`
(`investigate_rotate_gap.py`, verbatim copy of v13's logic with a fill-source counter, NOT a
modification of the shipped file) to directly count, per follower-day, whether the full-history path,
the decayed fallback path, or neither filled the boost. **The decayed fallback path engages on only
0.9-1.3% of follower-days in rotate mode and 0.0-0.9% in reverse mode** (4 seeds each) -- it almost
never gets a turn, which is the real reason it barely moves the rotate-mode number regardless of
whether it would help when it fires. Root cause is structural, not a tuning slip: the decayed path's
significance threshold is computed from the *effective* sample size under exponential decay-weighting,
which is always smaller than the full-history path's raw sample count -- a smaller effective N means a
*stricter* Bonferroni-corrected bar, so decay-weighting's recency benefit is partly cancelled by a
harder significance test working against it. A real fix would need to loosen the decayed path's own
threshold specifically (trading more false positives there for more chances to fire), not something
tried live in this session.

**Why does the insurance layer lose money in the flip/whipsaw regime?** Tracked day-by-day
chosen/killed state against the known 25-day flip schedule (`investigate_whipsaw.py`, single version,
single regime). Broken down by 25-day block:

| block | regime | v10 PnL | v14 PnL | delta | non-champ% |
|---|---|---|---|---|---|
| 0 | momentum | 33,715 | 76,115 | +42,400 | 60% |
| 1 | reversion | 42,488 | -34,990 | **-77,478** | 72% (still majority `mom`) |
| 2 | momentum | 31,665 | 33,058 | +1,394 | 0% |
| 3 | reversion | 43,769 | 42,807 | -961 | 0% |
| 4 | momentum | 8,776 | 19,269 | +10,493 | 16% |
| 5 | reversion | 33,480 | 8,455 | -25,025 | 32% |

**Confirmed exactly as suspected: the insurance layer's own detection lag (ROT_W=60/XSAC_W=40
trailing windows, ROT_P=5-day persistence) is slower than this generator's 25-day flip period.** Block
1 is the clearest case -- the regime has already flipped back to reversion, but the layer is STILL
majority-trading the `mom` fallback (72% of days) because its trailing window is still dominated by
the just-ended momentum block, actively fighting the new regime. That single block accounts for
-$77,478 of the total -$49,178 loss on its own. Every correctly-timed momentum block is a genuine win
(+42,400/+1,394/+10,493 = +$54,287 combined) -- the layer isn't broken, it's mistimed relative to a
regime that oscillates faster than its own memory. Faster windows would fix the timing but at the cost
of more real-data false positives, which is exactly why those windows were sized the way they are --
a real tradeoff, not a bug to patch.
