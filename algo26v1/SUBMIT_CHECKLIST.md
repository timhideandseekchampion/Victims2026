# SUBMISSION CHECKLIST — Algothon 2026

## Competition structure (confirmed by organizer)
- Dev data now: days **1–500** (`prices.txt`).
- Live board NOW: scored on **501–750** (we're ~41st @502) — **decides nothing**.
- Live board soon: scored on **751–1000** (practice).
- **General-round FINAL: develop on 1–1000, scored on 1001–1500. TOP 10 ADVANCE.**
- Finals: develop on 1–1500, scored on **1501–2000**.
- Real data for 501–750 arrives **~1 week** (then up to 1000). Every scoring window = 250 days.
- => The current board is low-stakes; the real optimization happens when data lands (use `validate_oos.py`).

## The current standing entry
On the live board keep **`Arbitrage_Victims_v4.zip`** (v2 + longer half-life, ≥ v2 everywhere:
last-250 763 vs 762, full 593 vs 582, higher Sharpe; leakage-clean, byte-verified). `Arbitrage_Victims_v2.zip`
remains the proven fallback. Do NOT burn scarce submissions optimizing the 501–750 board.

- Artifact verified 2026-07-14: zip integrity OK, contents byte-identical to the repo file,
  fresh-import scores exactly **761.76** (last-250) / **582.08** (full 60–500) locally,
  and **502.52** on the public leaderboard (matches book-only expectation → behaving as modeled).

## Optional test candidate: `Arbitrage_Victims_v3.zip`
v3 = v2 with the per-asset intercept dropped (imposes the known drift=0; one fewer parameter,
not a fitted signal). Better forecast IC (0.0589→0.0637, both halves) but **score-neutral**: full
589.3 / Sh 5.68 vs v2's 582.1, but last-250 **739.1 vs v2's 761.8** (worse on recent). A lateral,
principled move — NOT expected to raise the public score.
- Fine to submit tomorrow as a test (spare submissions, v2 is the guaranteed fallback).
- Keep it ONLY if its public score is clearly above ~502 beyond noise (unlikely — the change is
  below the board's ±15–50 resolution). Otherwise the **last-day submission is v2**.
- Do NOT ship a partial-intercept blend (k in (0,1)) — that looked best on our sample but is a
  warm-up-window artifact / overfitting.

## Protocol
1. Submit with **hours of buffer**, not minutes (platforms fail at deadlines). Set an alarm.
2. Keep **≥1 spare submission** in hand until v2 is confirmed as the standing entry.
3. After uploading, confirm on the platform that v2 is the ACTIVE/standing submission.
4. If the platform requires a specific filename inside the zip, re-zip the same .py under
   the required name — do not edit the code.

## NEVER submit
- ~~probe_long / probe_short~~ (deleted from repo — hardcoded directional probes, lose money)
- `Arbitrage_Victims.py` / `.zip` (superseded: same strategy, unhardened cache)
- `Arbitrage_Victims_lean.py` / `.zip` (valid but strictly-dominated choice: drops the small
  positive-EV reversion; only reconsider if you explicitly decide to minimize variance)
- Anything from `ols_*.py`, `pairs_overlay.py`, `adaptive_estimator.py`, `testers/` (research code)

## Why we are holding v2 (short version)
Three adversarial hunts (~30 hypotheses) found no change that improves expected PnL on unseen
days; the organizers set all idiosyncratic drifts to exactly 0 (directional bets are coin flips
by design); and the public 500–650 band is largely realization noise (final-score SE ≈ ±110).
v2 maximizes the two things that persist across the final re-draw: expected PnL and no wasted
variance. Full record: `FINDINGS.md` + `oracle_log.md`.
