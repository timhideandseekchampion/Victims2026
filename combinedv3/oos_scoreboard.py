"""MOMENT OF TRUTH: score every strategy on the REAL now-revealed hidden window (days 500-750),
using only prior days as warm-up (exactly as the live grader did). No analysis of the new data —
just: what would each have scored? The originally-submitted book is the validation anchor
(should reproduce the ~502 seen live)."""
import importlib.util
import numpy as np
import pandas as pd

prc_all = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
print(f"panel: {nInst} instruments x {nt} days\n")


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu, 0.0
    sr = np.sqrt(250) * mu / sd
    return mu * sr**2 / (sr**2 + 1), sr


def load(path, reset_extra=None):
    spec = importlib.util.spec_from_file_location("m_" + path.split("/")[-1][:-3], path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run(m, start, end):
    if hasattr(m, "_cache"):
        m._cache = {k: (None if k != "last_fit_t" else -10) for k in m._cache}
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(start, end + 1):
        p = prc_all[:, :t]; cur = p[:, -1]
        npos = np.clip(m.getMyPosition(p), -(dlr / cur).astype(int), (dlr / cur).astype(int)).astype(int) if t < end else cp.copy()
        d = npos - cp; cash -= cur.dot(d) + cm; dv = cur * np.abs(d); cm = (dv * commRate).sum(); cp = npos.copy()
        pl = cash + cp.dot(cur) - val; val = cash + cp.dot(cur)
        if t > start: pll.append(pl)
    return score(np.array(pll))


strats = [
    ("ANCHOR: v1 submitted (was 502 live)", "/home/SIG2026/algo26v1/Arbitrage_Victims.py"),
    ("primary (fixed fade, HL500)", "Arbitrage_Victims_combined.py"),
    ("olsalgo (OLS-adaptive ALGO)", "Arbitrage_Victims_combined_olsalgo.py"),
    ("adaptive (gated sleeves)", "Arbitrage_Victims_combined_adaptive.py"),
    ("regime (punt+throttle)", "Arbitrage_Victims_combined_regime.py"),
    ("punt (as saved)", "Arbitrage_Victims_combined_punt.py"),
    ("punt_neutral", "Arbitrage_Victims_combined_punt_neutral.py"),
    ("revblend", "Arbitrage_Victims_combined_revblend.py"),
]

# windows: OLD in-sample last-250 (days 250-500) vs NEW hidden OOS (days 500-750)
old_start, old_end = 250, 500
oos_start, oos_end = nt - 250, nt

print(f"{'strategy':38} {'OLD 250-500':>14} {'NEW 500-750 (OOS)':>20}")
print(f"{'':38} {'Score / Sharpe':>14} {'Score / Sharpe':>20}")
for label, path in strats:
    try:
        m = load(path)
        so, sho = run(m, old_start, old_end)
        m2 = load(path)
        sn, shn = run(m2, oos_start, oos_end)
        print(f"{label:38} {so:7.0f}/{sho:4.1f}      {sn:8.0f}/{shn:4.1f}")
    except Exception as e:
        print(f"{label:38} ERROR: {e}")
