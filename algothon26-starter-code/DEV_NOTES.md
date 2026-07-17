# Dev notes — Algothon 2026 workbench

Local tooling built around the official `eval.py`. None of this is submitted
except your algorithm file — see the single-file rule below.

## Files

| File | Role | Submitted? |
| :--- | :--- | :--- |
| `strategy.py` | The scaffold you develop in: toolkit + baselines + your alpha | via copy |
| `teamName.py` | A copy of `strategy.py` — the file `eval.py` imports & you submit | yes |
| `backtester.py` | Faithful `eval.py` replica + risk metrics, walk-forward, attribution, plots | no |
| `bench.py` | Compares the baselines (your yardstick) | no |
| `research.py` | Signal lab: IC, IC-decay, signal correlation, net-of-fees backtest, experiments | no |
| `analyze.py` | Reverse-engineer the generator (factor/PCA, ADF, cointegration, OU half-life) | no |
| `SIGNALS.md` | Signal research findings + experiment verdicts | no |
| `DGP.md` | How the data was likely generated & how to exploit it | no |
| `dashboard.py` + `dashboard_template.html` | Offline HTML workbench (Instrument / Portfolio / **Signals** / **Explore** tabs, with auto-Notes) | no |
| `eval.py`, `prices.txt` | Official grader + data (unchanged) | no |

**Single-file rule.** A submission is ONE algorithm file (+ optional
`requirements.txt`). So every helper, baseline and bit of plumbing lives inside
`strategy.py`. When ready: `cp strategy.py teamName.py` (rename to your team name
at submission time).

## The build loop

1. **Edit** `alpha()` in `strategy.py` §4 (or compose the §1/§2 helpers).
2. **Score it:** `.venv/bin/python backtester.py --strategy strategy --stats --walk-forward 5`
3. **Beat the baseline?** `.venv/bin/python bench.py --walk-forward`
4. **Eyeball it:**
   `.venv/bin/python backtester.py --strategy strategy --export-positions pos.csv`
   `.venv/bin/python dashboard.py --positions pos.csv` → open `dashboard.html`, pick "Loaded positions".
5. **Sync for the official grader / submission:** `cp strategy.py teamName.py && .venv/bin/python eval.py`

(`.venv` matches the grading sandbox: numpy 2.5.1, pandas 3.0.3. Create with
`python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`.)

## Toolkit reference (`strategy.py`)

**§1 Indicators** (each: prices `(nInst, t)` → length-`nInst` vector, latest day):
`returns(p, w)`, `momentum(p, w)`, `zscore(p, w)`, `rank(scores)` (cross-sectional,
→ ~[-1,1]), `realised_vol(p, w)` (annualised).

**§2 Sizing** (score vector → legal integer shares; handles $-limits, ÷price,
int rounding, clipping): `neutralize(scores)` (≈$-neutral), `size_fraction_of_limit`,
`size_inverse_vol` (risk-parity flavour), `to_shares`, `dollar_limits`.

**§4 Knobs:** `STRATEGY` (`"two_leg"` = the shipped book, `"single"` = the `alpha()`
path). Two-leg knobs: `IDIO_WINDOW`/`ALGO_WINDOW` (reversion lookbacks),
`IDIO_SCALE`/`ALGO_SCALE` (aggression; lower = bigger), `ALGO_FRAC` (fraction of ALGO's
$100k limit), `IDIO_SIZING`. Single-path knobs: `ACTIVE`, `SIZING`, `SCALE`.

**Current submission (`strategy.py` §3b `two_leg`):** ALGO index reversion ($100k) +
50-name cross-sectional reversion, both near the dollar limits — official Score **304**.
See SIGNALS.md ⭐ BREAKTHROUGH for the why (sizing is the dominant lever; the index leg is
a real edge). Inspect/tweak from the CLI without editing the file:
`backtester.py --two-leg [--idio-w --algo-w --idio-scale --algo-scale --algo-frac --idio-sizing]`
(composes with `--stats --walk-forward K --montecarlo --export-positions`).

## Baseline yardsticks (last 250 days, fraction sizing)

Un-tuned round defaults — the numbers to beat, **not** an optimised strategy.

| Baseline | Score | Sharpe | Sortino | note |
| :--- | ---: | ---: | ---: | :--- |
| `reversion` (−zscore, w=20) | **57.2** | 1.28 | 1.89 | this universe mean-reverts |
| `xs_rank` (fade 5-day winners) | 5.7 | 0.62 | 0.91 | mild |
| `momentum` (60-day) | −13.5 | −1.49 | −2.04 | trend-following loses here |
| `flat` (no trades) | 0.0 | — | — | sanity floor |

Reproduce anytime with `python bench.py`. (Starter momentum code scored ~0.10.)

## Watch-outs

- **Don't chase in-sample Score.** A short-window reversion hit ~169 on the last
  250 days but had a −137 walk-forward fold — overfit. Judge with
  `--walk-forward` / `bench.py --walk-forward`, not a single window.
- **Stay stateless.** `getMyPosition` recomputes from `prcSoFar` each call; no
  module globals that accumulate (an earlier version leaked state across folds).
- **Turnover costs.** The dashboard's portfolio panel is gross of fees; the
  backtester and `eval.py` charge commission (1bp; 0.2bp on instrument 0). A
  high-turnover edge can shrink a lot once fees bite — check the backtester Score,
  not just the dashboard.
