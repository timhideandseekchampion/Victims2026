"""signal_overlap.py — are the rotation challengers actually DIFFERENT edges, or all the same
cross-sectional stat-arb? Measure the average day-by-day correlation of the forecast VECTORS
(what actually drives positions) among all 7 signals over 500-750, plus each signal's mean IC."""
import numpy as np, pandas as pd
import SAFE_rotate as R

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
R._ensure_cache(prc)
names = ["champ"] + list(R.CHALLENGERS)
n = len(names)
T = range(500, 750)

M = np.zeros((n, n)); cnt = 0
for t in T:
    fs = {k: R._SIG[t][k] - R._SIG[t][k].mean() for k in names}
    cnt += 1
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            va, vb = fs[a], fs[b]
            d = np.sqrt((va @ va) * (vb @ vb))
            M[i, j] += (va @ vb / d) if d > 1e-12 else 0.0
M /= cnt

print("mean daily forecast-vector correlation (500-750):\n")
print("        " + "".join(f"{k:>7}" for k in names))
for i, k in enumerate(names):
    print(f"{k:>7} " + "".join(f"{M[i,j]:>7.2f}" for j in range(n)))

print("\nmean realized IC per signal (500-750):")
for k in names:
    ic = R._ic(k, 500, 750)
    print(f"  {k:<7} {ic.mean():+.4f}")

# crude clustering: which signals are >0.6 correlated with the champion's two components?
print("\ncorrelation with champion:")
for i, k in enumerate(names):
    if k == "champ": continue
    tag = "LEAD-LAG-like" if M[0, i] > 0.5 else ("OPPOSITE (momentum-side)" if M[0, i] < -0.3 else "partly independent")
    print(f"  {k:<7} corr={M[0,i]:+.2f}  -> {tag}")
