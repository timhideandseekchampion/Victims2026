# Oracle log — public-leaderboard submissions (validation set)

The public leaderboard is a **fixed hidden test set** = out-of-sample **validation** data
(only the fresh final is the prize).

> **2026-07-14 BUDGET UPDATE — probes CANCELLED.** Only **2–3 submissions remain** and the
> **last submission carries over to the final**. Rules from here:
> 1. **No probes** (a probe left standing = catastrophic final entry; and probe results
>    wouldn't change what we ship anyway).
> 2. v2 (502) is the standing submission → we are already "shipped" for the final.
> 3. A new submission happens ONLY for a **locally-proven improvement over v2**
>    (walk-forward, both-halves, paired significance — the full standard).
> 4. Keep ≥1 submission in reserve at all times.

- Only submit configs with an **a-priori hypothesis** (never a blind grid).
- A change ships for the final only if it improves **both** the local 500-day sample **and** the public score.
- Every row below is one "look" — the count is our **overfitting budget**. Keep it small; prefer fewer params.

## Local reference scores (our 500-day `prices.txt`, via eval.py mechanics)

| strategy | last-250 mean / Sharpe / Score | full 60-500 mean / Sharpe / Score |
|---|---|---|
| probe_long (pure long)  | −421.9 / −1.12 / −421.9 | −106.9 / −0.28 / −106.9 |
| probe_short (pure short) |  419.7 /  1.11 /  231.6 |  104.9 /  0.28 /    7.6 |
| **v2 (ship candidate)**  |  777.0 /  7.07 /  761.8 |  601.4 /  5.49 /  582.1 |
| lean (reversion off)     |  639.7 /  6.78 /  626.1 |  519.7 /  5.48 /  503.0 |

Note: local `probe_long` is deeply negative → beta is NOT free on our realization (ALGO −17.5%).

## Public submissions (fill in the observed public score)

| # | file submitted | hypothesis being tested | LOCAL last-250 Score | PUBLIC score | notes |
|---|---|---|---|---|---|
| 1 | Arbitrage_Victims_v2.py | baseline (already submitted) | 761.8 | **502** | ≈ our full-window book-only (~503) → reversion added ~0 on hidden set |
| 2 | probe_long.py  | how much beta PnL in hidden window? | −421.9 | _TBD_ | large + ⇒ window trends UP (gap=beta); ~0/− ⇒ gap=edge |
| 3 | probe_short.py | confirm beta sign/magnitude | 231.6 | _TBD_ | should mirror #2 if gap is pure beta |
| 4 | lean (reversion off) | does dropping the reversion cost anything on hidden set? | 626.1 | _TBD_ | if ≈ 502, reversion is dead weight → ship lean for lower final variance |

## FINAL PROTOCOL (2026-07-14, user decision)

**Probes permanently cancelled by user** ("hardcoding directional elements is wrong") — probe files
DELETED from the repo so they can never be uploaded by mistake. One probe would have sufficed
informationally (long/short are invertible through the Score formula's win/lose asymmetry), but its
answer could not have changed the ship decision: we are market-neutral regardless (drift-tilt
analysis: no tilt clears break-even; idio drifts set to exactly 0 by the organizers).

**Endgame:** hold `Arbitrage_Victims_v2.zip` (artifact verified: byte-identical contents, fresh-import
scores exactly 761.76 / 582.08). Re-submit it on the LAST DAY with hours of buffer as the standing
final entry. See `SUBMIT_CHECKLIST.md`. Test submissions of strategy variants remain ruled out:
every candidate is either below the board's ±15–50 resolution or already decisively measured locally.

## Hunt #2 outcome (2026-07-14, local, zero submissions spent)

All five uncovered angles tested and **DEAD** (index-residual −70$/d; GLS/SUR ≤0; seasonality
null across ~257 corrected tests; commission reclaim <$1/d recoverable; drift tilt = coin flip
— organizers set all idio drifts to exactly 0). **Verdict: HOLD — v2 stands, no submission spent.**
Fresh-window Score noise is ±~110/day SE → the public 500–650 band is largely luck-band;
expected-PnL-per-unit-variance (v2's design) is the whole game.

## Decision rule (Phase 1) — CANCELLED, kept for reference

- **probe_long large POSITIVE** (hundreds): hidden window trends up → the ~300 gap to the top is
  largely **beta** on this specific draw → leaders regress on the fresh final → **hold the robust book, we win by generalizing.**
- **probe_long ~0 / NEGATIVE**: beta is not the story → the gap is **real edge** → run Phase 3
  (deep edge hunt) and validate candidates here.

## Interpreting the v2=502 vs lean anchor (#1 vs #4)

If lean (#4) ≈ 502, the reversion contributes nothing on unseen data and only adds variance →
prefer **lean** for the one-shot fresh final. If lean < 502 by a clear margin, the reversion is
pulling weight out-of-sample and is worth keeping small.
