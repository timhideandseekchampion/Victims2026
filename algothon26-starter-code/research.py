#!/usr/bin/env python
"""Signal research lab for Algothon 2026 (dev-only).

Two questions, two reports:
  * Does a signal PREDICT?      -> IC report (Information Coefficient)
  * Can you MONETISE it net of fees? -> backtest report (Score/Sharpe/turnover)
Plus a correlation report to find complementary (diversifying) signals.

    python research.py                 # IC + correlation + net backtest tables
    python research.py --walk-forward  # add out-of-sample folds to the backtest table
    python research.py --start 60      # first day used for IC (needs warm-up)

IC = the cross-sectional correlation, on each day, between a signal's score and
the forward return over h days, averaged over days. Small but positive & stable
IC = a real edge. Turnover control / sizing don't change IC (they change the
score->trade mapping) — judge those in the backtest report.

Reuses backtester.run_backtest / make_grading_params and strategy's signals.
"""
import argparse

import numpy as np
import pandas as pd

import backtester as bt
import strategy as st

HORIZONS = (1, 5, 10)


def load():
    df = pd.read_csv("prices.txt", sep=r"\s+")
    return df.values.T, list(df.columns)


def signal_matrix(prc, signal_fn, start):
    """Score vector as-of the end of each day t, for t in [start, T-1]."""
    ts = list(range(start, prc.shape[1]))
    S = np.array([signal_fn(prc[:, :t + 1]) for t in ts])   # (ndays, N)
    return np.array(ts), S


def _rowcorr(A, B):
    """Per-row Pearson correlation of two (rows, N) matrices (NaN-safe)."""
    A = A - np.nanmean(A, axis=1, keepdims=True)
    B = B - np.nanmean(B, axis=1, keepdims=True)
    num = np.nansum(A * B, axis=1)
    den = np.sqrt(np.nansum(A * A, axis=1) * np.nansum(B * B, axis=1))
    return num / np.where(den < 1e-12, np.nan, den)


def ic_at(prc, ts, S, h):
    """Mean IC and IC-IR (mean/std of the daily IC series) at horizon h."""
    T = prc.shape[1]
    keep = ts + h < T
    tsv, Sv = ts[keep], S[keep]
    F = np.array([prc[:, t + h] / prc[:, t] - 1.0 for t in tsv])
    daily = _rowcorr(Sv, F)
    daily = daily[np.isfinite(daily)]
    if daily.size == 0:
        return 0.0, 0.0
    return float(daily.mean()), float(daily.mean() / daily.std()) if daily.std() > 1e-12 else 0.0


def ic_report(prc, start):
    print("=== IC report (cross-sectional corr of signal vs forward return) ===")
    print(f"{'signal':<12}" + "".join(f"IC{h:<6}" for h in HORIZONS) +
          f"{'IR@5':>7}  decay(IC h=1,2,5,10,20)")
    rows = []
    for name, fn in st.SIGNALS.items():
        ts, S = signal_matrix(prc, fn, start)
        ics = {h: ic_at(prc, ts, S, h)[0] for h in HORIZONS}
        ir5 = ic_at(prc, ts, S, 5)[1]
        decay = [ic_at(prc, ts, S, h)[0] for h in (1, 2, 5, 10, 20)]
        rows.append((name, ics, ir5, decay))
    rows.sort(key=lambda r: r[1][5], reverse=True)  # rank by 5-day IC
    for name, ics, ir5, decay in rows:
        print(f"{name:<12}" + "".join(f"{ics[h]:+.3f} " for h in HORIZONS) +
              f"{ir5:>+6.2f}  " + " ".join(f"{d:+.3f}" for d in decay))
    print("(positive IC = reversion signal works; ~0 or negative = no edge)\n")


def corr_report(prc, start):
    print("=== signal correlation (flattened score vectors) ===")
    names = list(st.SIGNALS)
    mats = {n: signal_matrix(prc, st.SIGNALS[n], start)[1].ravel() for n in names}
    print("           " + " ".join(f"{n[:7]:>7}" for n in names))
    for a in names:
        line = f"{a:<10} "
        for b in names:
            x, y = mats[a], mats[b]
            m = np.isfinite(x) & np.isfinite(y)
            c = np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 2 else np.nan
            line += f"{c:>7.2f}"
        print(line)
    print("(low/negative pairs are complementary — candidates to blend)\n")


def backtest_report(prc, names, walk_forward, days=None):
    comm, lim = bt.make_grading_params(prc.shape[0])
    nt = prc.shape[1]
    days = days if days else nt - 50            # default: all available minus 50-day warm-up
    blend = lambda p: (st.zrev(10)(p) + st.zrev(20)(p) + st.zrev(40)(p)) / 3.0

    # (label, config kwargs for make_get_position) — one turnover/sizing idea each
    configs = [
        ("rev20 baseline",      dict(signal_fn=st.zrev(20))),
        ("rev20 smooth5",       dict(signal_fn=st.zrev(20), smooth=5)),
        ("rev20 hold5",         dict(signal_fn=st.zrev(20), hold=5)),
        ("rev20 band0.3",       dict(signal_fn=st.zrev(20), band=0.3)),
        ("rev20 inverse_vol",   dict(signal_fn=st.zrev(20), sizing="inverse_vol")),
        ("rev30 baseline",      dict(signal_fn=st.zrev(30))),
        ("blend10/20/40",       dict(signal_fn=blend)),
        ("blend smooth5+invvol", dict(signal_fn=blend, smooth=5, sizing="inverse_vol")),
        # --- adaptive (EWMA) vs static, same sizing: does adaptivity win OOS? ---
        ("rev_ez20 (adaptive)",  dict(signal_fn=st.zrev_ewma(20))),
        ("rev_eblend (adaptive)", dict(signal_fn=st.alpha_rev_eblend)),
        # --- skeptical experiments (data argues against; measured anyway) ---
        ("blend + Kelly size",   dict(signal_fn=blend, sizing="kelly")),
        ("trend/revert mixed",   dict(signal_fn=st.alpha_trend_revert)),
    ]

    # combine (book-level) + overlays are position transforms, not plain configs:
    gp_blend = st.make_get_position(signal_fn=blend)
    gp_iv = st.make_get_position(signal_fn=blend, sizing="inverse_vol")
    gp_xs = st.make_get_position(signal_fn=st.alpha_xs_rank)
    extra = [
        ("combine blend+xs_rev5", st.combine_positions([(gp_blend, 0.7), (gp_xs, 0.3)])),
        ("blend + regime-gate", st.regime_gate(gp_blend)),
        ("blend+invvol (best ref)", gp_iv),
        ("blend+invvol + ev-gate0.6", st.make_get_position(signal_fn=st.cost_gate(blend, 0.6), sizing="inverse_vol")),
        ("blend+invvol + confidence", st.confidence_scale(gp_iv, blend)),
        ("blend+invvol + regime-scale", st.regime_scale(gp_iv)),
    ]

    # (label, get_position) for every config, plain + transform-based
    cases = [(label, st.make_get_position(**kw)) for label, kw in configs] + extra

    print(f"=== net-of-fees backtest report (last {days} days) ===")
    print(f"{'config':<24}{'Score':>8}{'Sharpe':>8}{'Sortino':>8}{'maxDD':>9}{'turn/day':>10}")
    results = []
    for label, gp in cases:
        r = bt.run_backtest(prc, gp, days, comm_rate=comm, dlr_pos_limit=lim, inst_names=names)
        results.append((label, gp, r))
        print(f"{label:<24}{r.score:>8.2f}{r.ann_sharpe:>8.2f}{r.sortino:>8.2f}"
              f"{r.max_drawdown:>9.0f}{r.avg_daily_turnover:>10.0f}")
    base = results[0][2].score
    best = max(results, key=lambda x: x[2].score)
    print(f"\nbaseline Score {base:.1f}; best net Score {best[2].score:.1f} -> {best[0]}")
    print(f"\n--- notes on best ({best[0]}) ---")
    for line in bt.insights(best[2]):
        print(f"  {line}")

    if walk_forward:
        folds = [nt - 4 * 50, nt - 3 * 50, nt - 2 * 50, nt - 50, nt]
        print(f"\n=== walk-forward (5 x 50-day OOS folds), Score per fold ===")
        print(f"{'config':<24}" + " ".join(f"f{d}".rjust(8) for d in folds) + f"{'mean':>8}{'min':>8}")
        for label, gp, _ in results:
            fs = np.array([bt.run_backtest(prc, gp, 50, last_day=d, comm_rate=comm,
                                           dlr_pos_limit=lim, inst_names=names).score for d in folds])
            print(f"{label:<24}" + " ".join(f"{x:>8.1f}" for x in fs) +
                  f"{fs.mean():>8.1f}{fs.min():>8.1f}")


def main():
    p = argparse.ArgumentParser(description="Signal research lab")
    p.add_argument("--start", type=int, default=60, help="first day for IC (warm-up before)")
    p.add_argument("--days", type=int, default=None, help="test days (default: all minus 50-day warm-up)")
    p.add_argument("--walk-forward", action="store_true")
    args = p.parse_args()

    prc, names = load()
    print(f"universe: {prc.shape[0]} instruments x {prc.shape[1]} days\n")
    ic_report(prc, args.start)
    corr_report(prc, args.start)
    backtest_report(prc, names, args.walk_forward, args.days)


if __name__ == "__main__":
    main()
