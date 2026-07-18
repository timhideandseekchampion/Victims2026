# DECISION.md — submission plan & validated findings (2026-07-18 session)

Consolidation of what we established this session. Complements `STRATEGY.md` (how the book works)
and `FINDINGS.md` (the original research trail). Everything below is verified on a no-look-ahead
engine that reproduces the official `eval_safe.py` score **to the cent** (612.98 on 500-750).

---

## 0. Tournament structure (as understood)

- **Preview phase (NOW):** graded on days 750→, a fixed-start window that grows +50 days at a time
  up to 750-1000. **This does NOT gate anything** — it's a live preview. Grading is path-dependent
  (positions locked as days accrue), so our 12th-place standing here is short-window noise, not a
  verdict.
- **Qualifier:** scored on **days 1000-1500** (a 500-day window). **Top-10 → the final.**
- **Final:** scored on **days 1500-2000** (a 500-day window). Prizes.

**Key implication:** the grades that matter are 500-day windows. Over a window that long, day-to-day
variance washes out and **Score ≈ the config's true mean PnL** (the `SR²/(SR²+1)` term is already
~0.96 at Sharpe ~5, so mean dominates). Short-window variance is NOT a catch-up tool here.

---

## 1. THE PLAN

| phase | window | goal | **ship** | why |
| :-- | :-- | :-- | :-- | :-- |
| Preview | 750-1000 | (nothing gated) | **SAFE** (leave running) | don't chase the preview; switching risks nothing gained. Switch happens before day 1000 anyway. |
| **Qualifier** | 1000-1500 | make **top-10** (survive) | **SAFE.py** | survival = clear a bar = highest floor / lowest variance. SAFE is purpose-built for this. |
| **Final** | 1500-2000 | place for a **prize** | **QUAL.py** if you need EV to reach podium; else SAFE | 500-day window rewards mean; QUAL has the highest robust mean. Only take the variance if a steady score wouldn't win. |

**One-line rule:** *SAFE to survive, QUAL to win.* Chasing the 750-1000 preview leaderboard is a trap.

---

## 2. The books (all self-contained, `getMyPosition(prcSoFar)`, numpy only)

| file | HL | BLEND | role |
| :-- | :-- | :-- | :-- |
| `SAFE.py` | ensemble | 0.30 | highest floor → **qualifier / survival** |
| `QUAL.py` | ensemble | **0.20** | highest robust MEAN → **final / catch-up** (NEW this session) |
| `SWING.py` | 1000 | 0.15 | higher mean, higher variance, lower floor — superseded by QUAL for the long-window objective |

Ranked by MEAN on **500-day windows** (the grade length), CONTRA_DOL=1M, HEDGE=False:

```
config            500d mean   floor
QUAL (ens b.20)      618        517     <- best mean AND best floor of the top group
SWING (hl1000 b.15)  613        503
SAFE (ens b.30)      600        501     <- lower mean, but see caveat below
```

**Caveat that keeps SAFE as the qualifier pick:** on a *lead-lag-weak* draw (like the real 500-750
leg) the heavier blend wins — SAFE 613 vs QUAL 549 on that leg. QUAL's higher *average* comes with
more regime risk. For top-10 survival on an unseen window, SAFE's robustness is the safer bet.

---

## 3. Validated knob settings (don't re-tune these)

- **CONTRA_DOL = 1,000,000.** Tested {0, 250k, 500k, 1M, 2M}. 1M beats 500k on mean, floor, AND the
  clean 500-750 leg. Turning the ALGO fade off (0) is by far the worst (−80 on 500d mean) — the
  fade's alpha has *weakened* but is NOT zero. 2M is marginally higher but the position is
  cap-clipped so 1M already deploys full ALGO capacity. → keep 1M. (`contra_sweep.py`)
- **BLEND: 0.30 (SAFE) / 0.20 (QUAL).** Turning reversion OFF (b.00) gives the LOWEST mean and floor
  — the weak (IC~0.02) reversion signal still helps via diversification (it's uncorrelated with
  lead-lag). Mean peaks ~0.20, then erodes past it. (`qualifier_sweep.py`)
- **HALF_LIVES: ensemble (250/500/1000/2000).** A LONGER single half-life does NOT help — across-window
  mean *falls* monotonically with HL (short 250-500 has higher mean). Longer only won on the recent
  500-750 leg (regime bet). No single HL dominates → the ensemble is the robustness hedge. Keep it.
  (`hl_sweep.py`)
- **HEDGE = False.** The ALGO fade pins the $100k cap, leaving `room ≈ 0` for a beta hedge, so
  HEDGE=True applies ~$0. Off is simpler and marginally better.

---

## 4. Ideas tested and REJECTED (don't chase these again)

- **Better signal / higher IC.** The lead-lag edge is capped at **IC ≈ 0.079**. A fully-fitted
  5-signal combination lands right at 0.079 OOS and cannot beat it — confirms the ceiling three ways
  (`edge_probe.py`). There is no better model to find; the edge is maxed.
- **Longer half-life.** Lowers mean (§3). Regime bet only.
- **Index next-day reversion.** Real historically (corr ~−0.10) but **decaying** — only −0.02 on the
  most recent window. Betting the forward book on a fading edge; passed. (`edge_probe.py`)
- **Turnover / no-trade band (hysteresis).** Commission is ~8% of PnL, but suppressing near-zero
  flips costs about as much forgone PnL as it saves — except a NARROW sweet spot at band≈0.05 that
  gives QUAL(b.20) a small win (+13 floor, −5% turnover) but HURTS SAFE(b.30). Fragile,
  config-specific. Optional bonus for the final book only, not a foundation. (`turnover_test.py`)
- **Exact dollar-neutral (balanced) sizing.** NOT a clean win. Decomposition shows it's not a
  variance-leak fix — it overrides the lowest-conviction (near-median) names, which helps the
  lead-lag-heavy QUAL book *specifically in lead-lag-weak regimes* (raises its floor) but hurts SAFE.
  It's a lead-lag-weak *regime hedge* expressed through sizing, equivalent to nudging BLEND up.
  (`review_probe.py`, `why_balanced.py`)

---

## 5. Code review verdict

- **No correctness bugs. No look-ahead.** Engine matches official grader to the cent.
- **One real inefficiency:** `sign(wz)` leaves the stock book with an unhedged residual index-beta
  of ±$40-60k/day (net long/short imbalance), and the ALGO fade consumes the whole $100k cap so
  nothing hedges it. Effect on score is small and only marginally/regime-dependently fixable
  (balanced sizing, §4). Worth knowing; not worth rewriting the production book.
- Minor/negligible: `.astype(int)` truncation (~½ share/name under-deploy); ridge refit-from-scratch
  each day (fine live).

---

## 6. Scripts (this session)

`fwd_window.py` short-window distribution · `qualifier_sweep.py` blend×HL by mean on long windows ·
`contra_sweep.py` ALGO notional · `hl_sweep.py` half-life · `turnover_test.py` hysteresis ·
`edge_probe.py` IC ceiling + index predictability · `review_probe.py` / `why_balanced.py` sizing review.
Run any with `/root/.venv/bin/python <script>` (needs numpy+pandas; the repo default python lacks them).
