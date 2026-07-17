"""
boxtiao.py — Box-Tiao / d'Aspremont maximally-mean-reverting PORTFOLIO, tested causally.

Idea (Box-Tiao 1977; d'Aspremont 2011; Cuturi-d'Aspremont): instead of trading each name's
deviation, find the linear basket x whose value process is the LEAST predictable / most
mean-reverting, and trade its z-score. Predictability of x on a VAR(1) fit p_t = M p_{t-1}+e:
    nu(x) = (x' M Gamma M' x) / (x' Gamma x),   Gamma = cov(levels)
Minimising nu over x = the smallest generalized eigenvector of (M Gamma M', Gamma). The most
mean-reverting basket is that eigenvector. We build several top baskets, trade each as an OU
spread (fade the z-score), demean to stay market-neutral, size to $10k/name, and SCORE it
eval-faithfully OOS. Also test the multi-basket book vs the ridge book.

All causal: at day t we fit on prices[:, :t] only.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def boxtiao_baskets(P_levels, nb=5, lookback=250):
    """Return the nb most mean-reverting baskets (rows = weight vectors over the 50 idio names)."""
    X = P_levels[1:, -lookback:]                       # (50, L) idio price levels (log)
    X = X - X.mean(1, keepdims=True)
    p0 = X[:, :-1].T; p1 = X[:, 1:].T                  # (L-1, 50)
    # VAR(1): p1 ~ p0 M'  ->  M = (p1' p0)(p0'p0)^-1
    G = (p0.T @ p0) / p0.shape[0]
    M = np.linalg.solve(G + 1e-6 * np.eye(50), (p0.T @ p1) / p0.shape[0]).T
    A = M @ G @ M.T                                    # predictable part
    # generalized eigenproblem A x = lam G x ; smallest lam = most mean-reverting
    Gi = np.linalg.inv(G + 1e-6 * np.eye(50))
    ev, V = np.linalg.eig(Gi @ A)
    ev = ev.real; V = V.real
    order = np.argsort(ev)                              # ascending predictability
    return V[:, order[:nb]].T                          # (nb, 50)

def score(pll):
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu, 0.0
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1), sr

def backtest_boxtiao(Sd, Ed, nb=5, zwin=20, refit=10):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    baskets = None; hist = None
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 260:
            if baskets is None or (t % refit == 0):
                baskets = boxtiao_baskets(lp[:, :t], nb=nb)   # causal
            # each basket's value series over a z-window (causal, ends at t-1)
            sig_w = np.zeros(50)
            for b in baskets:
                val_series = b @ (lp[1:, t - zwin - 1:t - 1] - lp[1:, t - zwin - 1:t - 1].mean(1, keepdims=True))
                z = (val_series[-1] - val_series.mean()) / (val_series.std() + 1e-9)
                sig_w += -np.clip(z, -3, 3) * b            # fade: short the basket if rich
            if sig_w.std() > 1e-9:
                s = sig_w - sig_w.mean()
                s = s / (np.abs(s).max() + 1e-12)
                pos[1:] = s * (dlr[1:] / cur[1:])
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    return score(pll)

print("Box-Tiao maximally-mean-reverting basket book — SCORE by 250-day leg (causal, refit/10d):\n")
print(f"{'leg':<12}{'nb=3':>9}{'nb=5':>9}{'nb=10':>9}")
for S in range(250, 501, 50):
    row = []
    for nb in (3, 5, 10):
        sc, sr = backtest_boxtiao(S, S + 250, nb=nb)
        row.append(sc)
    print(f"{f'{S}-{S+250}':<12}{row[0]:9.0f}{row[1]:9.0f}{row[2]:9.0f}")

# also measure its cross-sectional IC to compare to ridge 0.079
def bt_ic(S, E, nb=5, zwin=20, refit=10):
    ics = []; baskets = None
    for t in range(max(S, 260), min(E, nt - 1)):
        if baskets is None or (t % refit == 0):
            baskets = boxtiao_baskets(lp[:, :t], nb=nb)
        sig_w = np.zeros(50)
        for b in baskets:
            vs = b @ (lp[1:, t - zwin:t] - lp[1:, t - zwin:t].mean(1, keepdims=True))
            z = (vs[-1] - vs.mean()) / (vs.std() + 1e-9)
            sig_w += -np.clip(z, -3, 3) * b
        s = sig_w - sig_w.mean(); fwd = lp[1:, t + 1] - lp[1:, t]; fwd = fwd - fwd.mean()
        if s.std() > 1e-9 and fwd.std() > 1e-9: ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics)
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)) + 1e-12)
ic, t = bt_ic(400, 749)
print(f"\nBox-Tiao basket cross-sectional IC (400-750): {ic:.4f} (t={t:.2f})   [ridge baseline 0.079]")
