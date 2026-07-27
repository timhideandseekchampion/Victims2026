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
| `SAFE_llboost_v7.py` | **SAFE_llboost_v6 + re-tuned `COMBINE_GAIN`**: ALGO leg re-swept against the TRUE, final v6 idio book (never checked before) — every parameter confirmed still-optimal except `COMBINE_GAIN` (3.5→16.0), independently confirmed by a 720-combo joint grid search — **best result overall, see below** |

The idio book (instruments 1–49) is identical across the first seven; `SAFE_llboost.py` and its v2–v6 variants are the exceptions (they extend the idio book with the pairwise boost described below).

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
  circular-shift surrogate test (p<0.001 full, <0.00025 new). Plausibly an index-level vol risk premium.
- **Caveat:** it is index-specific, so persistence to the finals draw depends on the generator. The adaptive
  gate is the safety net — it sizes the leg to zero if the effect is absent. Combining with lead-lag hurts.

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

## `SAFE_llboost_v7.py` — re-tuned `COMBINE_GAIN` (best result overall)
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
