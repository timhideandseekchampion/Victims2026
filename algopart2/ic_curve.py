"""
ic_curve.py — the honest question: is IC still RISING with more data (data-starved -> finals will
help), or SATURATED (no more juice)? Fit the ridge on a FLAT training window of length L days and
measure OOS cross-sectional IC on a FIXED eval span, for increasing L. Also fit on ALL history to
each day (expanding) to show the best we currently get and extrapolate the 1/sqrt(T) trend.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]

def ridge_flat(t, L, a=0.1):
    """fit on the last L return-days before day t (flat weights), predict next-day idio return."""
    r = lp[:, :t]; r = r[:, 1:] - r[:, :-1]                 # (51, t-1)
    if r.shape[1] < L + 2: return None
    r = r[:, -(L + 1):]                                     # last L+1 returns
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    mx = X.mean(0); my = Y.mean(0); Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ Xc + a * np.eye(51), Xc.T @ Yc)
    f = my + (xin - mx) @ B; return f - f.mean()

def ic_over(evalspan, L):
    ics = []
    for t in range(evalspan[0], evalspan[1]):
        s = ridge_flat(t, L)
        if s is None: continue
        fwd = RET[1:, t - 1]                                # aligned: forecast targets RET[:,t-1]
        if s.std() > 1e-12 and fwd.std() > 1e-12: ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics)
    if len(ics) < 5: return np.nan, np.nan
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))

EVAL = (620, 749)                                            # fixed recent eval span (needs >=L history before)
print(f"IC learning curve — training-window length L vs OOS IC on days {EVAL[0]}-{EVAL[1]}:\n")
print(f"{'train L (days)':>16}{'IC':>9}{'t':>7}")
prev = None; curve = []
for L in (100, 150, 200, 300, 400, 500, 600):
    ic, t = ic_over(EVAL, L)
    curve.append((L, ic))
    arrow = "" if prev is None or np.isnan(ic) else (" ↑" if ic > prev + 0.001 else (" ↓" if ic < prev - 0.001 else " ="))
    print(f"{L:>16}{ic:>9.4f}{t:>7.2f}{arrow}")
    prev = ic

# is it still rising? fit IC ~ a - b/sqrt(L) and extrapolate to more data
xs = np.array([1 / np.sqrt(L) for L, ic in curve if ic == ic])
ys = np.array([ic for L, ic in curve if ic == ic])
A = np.vstack([np.ones_like(xs), xs]).T
coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
asymptote = coef[0]                                         # IC as L -> infinity
print(f"\nfit IC ≈ {coef[0]:.4f} - {(-coef[1]):.4f}/sqrt(L)   -> asymptotic IC (infinite data) ≈ {asymptote:.4f}")
for L in (750, 1000, 1500):
    print(f"   projected IC at L={L}: {coef[0] + coef[1]/np.sqrt(L):.4f}")
print("\nreading: if IC keeps rising L=400->600 and the asymptote is well above the current ~0.077,")
print("we're data-starved and the finals (more history) lift IC for free. If flat, we're saturated.")
