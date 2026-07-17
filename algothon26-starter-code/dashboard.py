#!/usr/bin/env python
"""Algothon 2026 research workbench — generate an offline HTML dashboard.

Reads prices.txt (and, optionally, a per-instrument position matrix exported
by backtester.py) and emits a single self-contained `dashboard.html` you can
open by double-clicking — no server, works offline, lives in the repo.

The page renders, per instrument:
  * price line + Bollinger Bands + fast/slow moving averages
  * volatility regime and trend regime shading
  * entry/exit markers
  * a rolling-volatility panel
  * a strategy panel (equity curve + position state)

Entries/exits come from either a built-in illustrative signal (Bollinger
reversion, MA crossover, z-score reversion — so you can see the machinery
before you have a strategy) OR your own positions loaded via --positions.

Usage:
    python dashboard.py                      # build dashboard.html from prices.txt
    python dashboard.py --prices prices.txt --out dashboard.html
    python dashboard.py --positions pos.csv  # overlay a real strategy's positions

Regenerate whenever prices.txt changes or you produce new positions.
Export a position matrix from the backtester with:
    python backtester.py --export-positions pos.csv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd


def load_prices(fn):
    df = pd.read_csv(fn, sep=r"\s+", header=0, index_col=None)
    return df  # columns = instrument names, rows = days


def load_positions(fn):
    """Optional per-instrument position matrix (rows=days, cols=instruments).

    Accepts the CSV that `backtester.py --export-positions` writes: a `day`
    column plus one column per instrument. Returns (days, {name: [pos...]}).
    """
    df = pd.read_csv(fn)
    day_col = "day" if "day" in df.columns else df.columns[0]
    days = df[day_col].astype(int).tolist()
    pos = {c: df[c].astype(float).tolist() for c in df.columns if c != day_col}
    return days, pos


def build(prices_df, positions=None, runs=None, books=None):
    names = list(prices_df.columns)
    # per-instrument price arrays, rounded to 2dp (prices are 2dp anyway)
    prices = [[round(float(v), 2) for v in prices_df[c].tolist()] for c in names]
    payload = {
        "names": names,
        "prices": prices,
        "nDays": int(prices_df.shape[0]),
        "positions": None,
        "books": books,   # optional [{label, days, byName}, ...] for the multi-strategy picker
        "runs": runs,     # optional Compare-tab data from backtester --export-runs
    }
    if positions is not None:
        pos_days, pos_map = positions
        payload["positions"] = {
            "days": pos_days,
            "byName": pos_map,
        }
    elif books:
        # keep single-book "Loaded positions" working: default it to the first book
        b0 = books[0]
        payload["positions"] = {"days": b0["days"], "byName": b0["byName"]}
    return payload


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate the Algothon research dashboard")
    p.add_argument("--prices", default="./prices.txt")
    p.add_argument("--out", default="./dashboard.html")
    p.add_argument("--positions", help="optional per-instrument position CSV "
                                       "(from backtester.py --export-positions)")
    p.add_argument("--runs", help="optional runs JSON for the Compare tab "
                                  "(from backtester.py --export-runs)")
    p.add_argument("--books", help="optional JSON list of strategy books "
                                   "[{label,days,byName}, ...] (from export_books.py) "
                                   "— adds a per-strategy entries/exits picker")
    args = p.parse_args(argv)

    prices_df = load_prices(args.prices)
    positions = load_positions(args.positions) if args.positions else None
    runs = None
    if args.runs:
        with open(args.runs, "r", encoding="utf-8") as f:
            runs = json.load(f).get("runs")
    books = None
    if args.books:
        with open(args.books, "r", encoding="utf-8") as f:
            books = json.load(f)
    payload = build(prices_df, positions, runs, books)

    here = os.path.dirname(os.path.abspath(__file__))
    tpl_path = os.path.join(here, "dashboard_template.html")
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = f.read()

    data_json = json.dumps(payload, separators=(",", ":"))
    html = tpl.replace("/*__DATA__*/null", data_json)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    n_inst, n_days = prices_df.shape[1], prices_df.shape[0]
    extra = " with loaded positions" if positions else ""
    extra += f" + {len(runs)} compare-runs" if runs else ""
    print(f"Wrote {args.out}: {n_inst} instruments x {n_days} days{extra}")
    print(f"Open it in a browser:  file://{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
