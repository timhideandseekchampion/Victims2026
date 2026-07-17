# SAFE & SWING — how the strategy works

A plain-English + technical explainer for the two shipped books, `SAFE.py` and `SWING.py`.
Both are self-contained (`getMyPosition(prcSoFar)`, numpy only) and score against the standard
Algothon grader (`eval.py`): **Score = mean(dailyPnL) · SR²/(SR²+1)**, SR = √250·mean/std, with
instrument 0 (ALGO) getting a 10× position limit ($100k) and 5× lower commission (0.2bp).

---

## 1. What we're trading (the market)

- **51 instruments:** `ALGO` (instrument 0) + 50 named stocks. Daily prices.
- **`ALGO` is the equal-weight index** of the 50 stocks (corr ≈ 0.98). It gets a special
  **$100k position limit** and **0.2bp commission** — a deliberate hint to trade it hard.
- **The data-generating process** (reverse-engineered with high confidence): one common factor
  (the index, ~20% of each name's variance, all betas ≈ 1), returns approximately Gaussian, **no
  fat tails, no volatility clustering.** On top of that noise sit the only two tradeable edges:
  1. a **directed peer lead-lag** — today's cross-section of all 51 returns linearly predicts each
     name's *next-day* return;
  2. **cross-sectional mean-reversion** — names that get rich/cheap vs the universe revert over
     ~5–10 days (the index itself also mean-reverts, though that edge has faded recently).
- **Every stock's drift is zero and they are statistically identical.** There is *no* way to know
  in advance which names will win — so the book is **market-neutral and trades all 50 for breadth.**

---

## 2. The edge — three signals

Both books are built from the same three orthogonal signals. They differ only in *how much* they
lean on each (§4).

### Signal 1 — Peer lead-lag forecast (the core, ~85% of the book)
A **forgetting-weighted ridge regression** predicts each of the 50 stocks' next-day return from
**today's full 51-name return cross-section**:

```
returns r_t (51-vector)  →  ridge B (51×50)  →  r̂_{t+1} for the 50 stocks
```

- Fit on all history, exponentially down-weighting older days (`HALF_LIFE`), with a small L2
  penalty (`RIDGE_A = 0.1`) to stabilise the 2,550-coefficient matrix.
- The forecast is **demeaned** across the 50 names → market-neutral by construction.
- This is the dominant edge: cross-sectional **IC ≈ 0.079** (t ≈ 8) — small per-name, but applied
  across 50 names every day it is a strong, statistically robust signal.

### Signal 2 — Cross-sectional reversion (the diversifier, blended in)
`-zscore` of each name's trailing **10-day return**, demeaned (buy relative losers, sell relative
winners). Weaker on its own (IC ≈ 0.02) but **partly independent** of the lead-lag, so blending a
little in **steadies the combined signal across regimes** (see `BLEND`, §4).

### Signal 3 — ALGO index fade (the index leg)
Fade the index's recent **30-day move** (z-scored over a 60-day window), sized up to the special
**$100k cap**. Historically a real reversion edge; on recent data it has weakened to roughly neutral,
so it now mostly fills the ALGO capacity rather than adding much alpha — kept as cheap optionality
in case the index-reversion regime returns.

---

## 3. How a position is formed each day

For each trading day, `getMyPosition(prcSoFar)` does:

1. Compute the **lead-lag forecast** (Signal 1) and z-score it.
2. **Blend** in the reversion signal (Signal 2): `wz = (1−BLEND)·leadlag + BLEND·reversion`.
3. **Idio leg (50 stocks):** `position = sign(wz) × ($10,000 / price)` — i.e. **every name is held
   at its full $10k limit**, long if the signal is positive, short if negative. No conviction gate.
4. **ALGO leg:** the index fade (Signal 3), pinned toward its $100k cap.
5. **Clip** to integer shares within the dollar limits.

**Why sign-sizing / full deployment?** The Score is capital-capped and behaves like PnL at these
Sharpe levels, so deploying the full **~$600k legal gross** ($500k across the 50 stocks + $100k on
ALGO) maximises it. Down-weighting low-conviction names (vol-scaling, inverse-variance, OU-speed
sizing) all *lowered* the Score in testing — because it deploys less capital. **Breadth = Sharpe:**
trading all 50 names diversifies the signal; dropping any (even historical "losers") hurts, because
the names are statistically identical and losers don't persist.

---

## 4. SAFE vs SWING — the only differences

Both use `RIDGE_A = 0.1` and `HEDGE = False`. They differ in two knobs:

| knob | **SAFE** (`SAFE.py`) | **SWING** (`SWING.py`) | effect |
| :-- | :-- | :-- | :-- |
| memory | **HL-ensemble** (avg of half-lives 250/500/1000/2000) | single **HL = 1000** | ensemble lowers estimation variance → steadier |
| `BLEND` (reversion weight) | **0.30** | **0.15** | more reversion = steadier floor; less = leans on lead-lag = higher upside |

That's it — same code, two settings. Everything else (signals, sign-sizing, ALGO fade, limits) is
identical.

**Why these exact values** (both are flat-plateau choices, not fragile peaks):
- **SAFE `BLEND=0.30`** maximises the *worst-window* score (the floor) — right for reliably clearing
  a qualifying bar.
- **SWING `BLEND=0.15`** maximises the *average* score across windows — right for swinging at the
  podium.
- **`HEDGE=False`:** a residual beta-hedge was tested and found near-inert (the demean already makes
  the book ~beta-neutral, and the ALGO fade pins the cap leaving no room for a hedge) and marginally
  negative on recent data — so it's off in both.

### Performance profile (verified)

| metric | **SAFE** | **SWING** |
| :-- | --: | --: |
| Score on the last graded window (500–750), via `eval.py` | **612.98** | 574.19 |
| Sharpe (that window) | 5.68 | 5.09 |
| Mean across all 250-day windows | ~640 | **~718** |
| Floor (worst window) | **~495–513** | ~426–574 |
| Behaviour | higher floor, steadier, best on cold-start | higher mean, higher variance, bigger upside tail |

**SWING is the book that scored 1542 on days 750–800 (fresh, unseen data) → 4th place** — the same
~0.08-IC edge landing on a favourable short window.

---

## 5. Why it's built this way (design decisions, all evidence-backed)

- **IC is capped at ~0.079** — proven three ways: an oracle in-sample fit reaches 0.40 but collapses
  to 0.079 causally (the gap is irreducible estimation error on a 2,550-coefficient matrix);
  the process is **VAR(1)** (adding lag-2/3 only adds noise); and for Gaussian data nonlinear methods
  are provably equivalent to linear. So no exotic model beats the ridge.
- **The lead-lag matrix is stationary** (a Kalman/TVP fit prefers zero drift), so adaptive/
  online methods add nothing — the regime "swings" in Sharpe are realized-return variance, not
  structural change.
- **Reversion is cross-sectional, not basket/residual** — so the classic stat-arb recipes
  (Avellaneda-Lee PCA-OU, Box-Tiao/cointegration baskets) are dead here; simple `-zscore` wins.
- **Ensembling helps only across memory length** (SAFE's HL-ensemble); bagging and predictor-set
  diversification don't, because the members are ~98% correlated.
- **The leaderboard is a variance tail, not an IC ladder** — everyone with the lead-lag insight gets
  ~0.08 IC; ranks are decided by the window draw + full deployment + variance, not by a better signal.

---

## 6. How to use them (submission policy)

- **Every qualifier → `SAFE.py`.** Goal is to clear a low bar (top-10) reliably; SAFE's high floor
  and steadiness do that, and it can still spike on a good window.
- **The final → `SAFE.py` by default; switch to `SWING.py` only if you need points to reach the
  podium (1st–3rd).** Variance is a tool for *catching up* — use SWING when a steady score wouldn't
  reach a prize, but stay on SAFE if you're already near the top (variance can drop you as easily as
  lift you).

---

## 7. Parameter quick-reference

| parameter | SAFE | SWING | meaning |
| :-- | :-- | :-- | :-- |
| `HALF_LIVES` / `HALF_LIFE` | (250,500,1000,2000) | 1000 | forgetting memory of the ridge |
| `RIDGE_A` | 0.1 | 0.1 | L2 penalty stabilising the lead-lag matrix |
| `BLEND` | 0.30 | 0.15 | weight on cross-sectional reversion vs lead-lag |
| `REV_W` | 10 | 10 | reversion lookback (days) |
| `CONTRA_DOL` | 1,000,000 | 1,000,000 | ALGO fade notional (pins the $100k cap) |
| `CONTRA_K` / `CONTRA_WZ` | 30 / 60 | 30 / 60 | ALGO move lookback / z-score window |
| `HEDGE` | False | False | residual beta-hedge (off — near-inert here) |
| idio sizing | sign × $10k | sign × $10k | full-deploy every name, market-neutral |

*Reproduce any figure with the scripts in this folder (`eval.py`, `robustness.py`, `sensitivity.py`,
`ic_hunt.py`, etc.); full research trail in `FINDINGS.md`.*
