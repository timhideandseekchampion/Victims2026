#!/usr/bin/env python
"""Baseline benchmark — the yardstick for "did my idea beat the baseline?"

Runs every baseline alpha in strategy.BASELINES (plus a flat no-trade line, and
your current edited strategy.getMyPosition) through the faithful backtester over
the last 250 days, and prints a comparison table. Optionally add walk-forward.

    python bench.py                 # last-250 comparison table
    python bench.py --walk-forward  # also show per-fold Scores (robustness)
    python bench.py --sizing inverse_vol   # compare baselines under a sizing mode

Dev-only: not part of a submission. Reuses backtester.run_backtest, so numbers
match eval.py exactly.
"""
import argparse

import numpy as np
import pandas as pd

import backtester as bt
import strategy as st


def load():
    df = pd.read_csv("prices.txt", sep=r"\s+")
    prc = df.values.T
    names = list(df.columns)
    comm, lim = bt.make_grading_params(prc.shape[0])
    return prc, names, comm, lim


def run(prc, names, comm, lim, get_pos, days=250, last_day=None):
    return bt.run_backtest(prc, get_pos, num_test_days=days, last_day=last_day,
                           comm_rate=comm, dlr_pos_limit=lim, inst_names=names)


def main():
    p = argparse.ArgumentParser(description="Compare baseline alphas")
    p.add_argument("--days", type=int, default=None, help="test days (default: all minus 50-day warm-up)")
    p.add_argument("--sizing", default=None, help='override sizing: "fraction" or "inverse_vol"')
    p.add_argument("--walk-forward", action="store_true", help="also show 5 out-of-sample folds")
    args = p.parse_args()

    prc, names, comm, lim = load()
    nt = prc.shape[1]
    if args.days is None:
        args.days = nt - 50
    sizing = args.sizing or st.SIZING

    # candidates: flat, each baseline, and the current edited strategy
    rows = [("flat", lambda pr: np.zeros(pr.shape[0]))]
    for key in st.BASELINES:
        rows.append((f"baseline:{key}", st.make_get_position(active=key, sizing=sizing)))
    rows.append((f"strategy (§4, ACTIVE={st.ACTIVE})", st.getMyPosition))

    print(f"universe: {prc.shape[0]} instruments · scoring last {args.days} days · sizing={sizing}\n")
    print(f"{'candidate':<32} {'Score':>8} {'Sharpe':>7} {'Sortino':>8} {'Calmar':>7} "
          f"{'maxDD':>9} {'turn/day':>9}")
    print("-" * 84)
    for label, gp in rows:
        r = run(prc, names, comm, lim, gp, args.days)
        print(f"{label:<32} {r.score:>8.2f} {r.ann_sharpe:>7.2f} {r.sortino:>8.2f} "
              f"{r.calmar:>7.2f} {r.max_drawdown:>9.0f} {r.avg_daily_turnover:>9.0f}")

    if args.walk_forward:
        folds = [nt - 4 * 50, nt - 3 * 50, nt - 2 * 50, nt - 50, nt]
        print(f"\nwalk-forward (5 × 50-day out-of-sample folds), Score per fold:")
        print(f"{'candidate':<24} " + " ".join(f"f{d}".rjust(8) for d in folds) + f" {'mean':>8} {'min':>8}")
        print("-" * 84)
        for key in st.BASELINES:
            gp = st.make_get_position(active=key, sizing=sizing)
            scores = [run(prc, names, comm, lim, gp, 50, last_day=d).score for d in folds]
            s = np.array(scores)
            print(f"{'baseline:' + key:<24} " + " ".join(f"{x:>8.1f}" for x in s) +
                  f" {s.mean():>8.1f} {s.min():>8.1f}")


if __name__ == "__main__":
    main()
