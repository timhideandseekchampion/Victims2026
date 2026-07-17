#!/usr/bin/env python
"""Export our real strategy position matrices as `books` for the workbench
dashboard (entries/exits per instrument). Writes books.json:
    [{label, days, byName:{INSTRUMENT:[shares...]}}, ...]
Positions are the ACTUAL held shares each day (after the eval $-cap clip),
so the ▲/▼ markers show the strategy's real long/short/flat flips.
"""
import json, sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/SIG2026/algo26v1")
import backtest_full as bt, Arbitrage_Victims as S, importlib
from pairs_overlay import PairsOverlay
importlib.reload(S)

prc = bt.prcAll
names = open("/home/SIG2026/algo26v1/prices.txt").readline().split()
START = 60                      # skip warm-up
dlr = bt.dlrPosLimit

def run_book(add_pairs=False, leg="all"):
    """leg: 'all' = full net positions; 'book' = 50-asset cross-sectional leg only (ALGO off);
    'algo' = ALGO overlay only (reversion+hedge on col 0, 50 assets off)."""
    ov = PairsOverlay(dollars_per_leg=8000); ov.RESELECT_EVERY = 50
    bt._reset(S)
    days = []; mat = []
    for t in range(START, 500):
        p = prc[:, :t]; cur = p[:, -1]
        base = S.getMyPosition(p).astype(float)
        if add_pairs: base = base + ov.positions(p)
        lim = (dlr / cur).astype(int)
        held = np.clip(base, -lim, lim).astype(int)
        if leg == "book": held[0] = 0                 # drop the ALGO overlay -> just the stat-arb book
        elif leg == "algo": held[1:] = 0              # keep only the ALGO index leg
        days.append(t)
        mat.append(held.copy())
    mat = np.array(mat)
    byName = {names[k]: [int(v) for v in mat[:, k]] for k in range(51)}
    return {"days": days, "byName": byName}

books = []
b = run_book(leg="all");  b["label"] = "1 · Shipped — full net (Score ~726)"; books.append(b)
b = run_book(leg="book"); b["label"] = "2 · Cross-sectional book only (50 names, ALGO off)"; books.append(b)
b = run_book(leg="algo"); b["label"] = "3 · ALGO leg only (reversion + hedge)"; books.append(b)
b = run_book(add_pairs=True); b["label"] = "4 · Shipped + pairs overlay (experimental)"; books.append(b)

json.dump(books, open("/home/SIG2026/algo26v1/testers/books.json", "w"), separators=(",", ":"))
# quick sanity: count entries/exits (sign flips) for a couple names on the shipped book
sh = books[0]["byName"]
for nm in ["ALGO", names[1], names[20]]:
    arr = np.sign(sh[nm]); flips = int((np.diff(arr) != 0).sum())
    print(f"  {nm:>6}: {flips} long/short/flat flips over {len(arr)} days")
print(f"wrote books.json: {len(books)} books, {len(books[0]['days'])} days each, 51 instruments")
