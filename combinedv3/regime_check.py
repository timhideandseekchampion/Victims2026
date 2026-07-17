#!/usr/bin/env python
"""Submission-day helper: read a price file, report the current cross-sectional dispersion
regime, and suggest punt-vs-robust.

Rationale (see README / regime_test.py): the book earns ~$920/day when cross-sectional
dispersion is HIGH vs ~$530 when LOW. We can't exploit that intraday (capital-capped), but on
final day we CAN pick which build to submit: high-dispersion regime -> the edge will be strong,
submit the aggressive punt; low-dispersion -> quiet window, submit the robust book.

Usage:  python regime_check.py [prices.txt]
This is a soft LEAN (the dispersion->PnL correlation is only ~0.05-0.07), not a hard switch.
"""
import sys
import numpy as np
import pandas as pd

fn = sys.argv[1] if len(sys.argv) > 1 else "prices.txt"
prc = pd.read_csv(fn, sep=r"\s+", header=0).values.T          # (51, T) incl. ALGO row 0
nInst, T = prc.shape
lp = np.log(prc)
ret = lp[:, 1:] - lp[:, :-1]                                  # daily log returns

# daily cross-sectional dispersion = std across the 50 tradeable names each day
disp = ret[1:].std(0)                                         # (T-1,)
cur20 = disp[-20:].mean()                                     # current 20-day regime
pct = 100.0 * (disp < cur20).mean()                           # percentile of current vs full history

# map percentile -> lean, with an EV estimate anchored to the tercile PnLs (~$530 low, ~$920 high)
lo_pnl, hi_pnl = 530.0, 920.0
ev = lo_pnl + (hi_pnl - lo_pnl) * (pct / 100.0)               # rough per-day EV interpolation

if pct >= 60:
    lean = "PUNT  (aggressive: Arbitrage_Victims_combined_punt.py, DEMEAN=True)"
elif pct <= 40:
    lean = "ROBUST (Arbitrage_Victims_combined.py)"
else:
    lean = "ROBUST by default (middle regime; punt only if you want the variance)"

print(f"file: {fn}   ({nInst} instruments, {T} days)")
print(f"current 20d cross-sectional dispersion : {cur20:.5f}")
print(f"  percentile vs full history           : {pct:.0f}th")
print(f"  full-history range (min/median/max)  : {disp.min():.5f} / {np.median(disp):.5f} / {disp.max():.5f}")
print(f"  rough per-day EV at this regime      : ~${ev:.0f}/day  (tercile anchors $530 low .. $920 high)")
print(f"\nLEAN -> {lean}")
print("(soft signal: dispersion->PnL corr ~0.05-0.07; treat as a tilt, not a rule)")
