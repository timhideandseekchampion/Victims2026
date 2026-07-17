"""
multilag.py — genuinely-untested IC/robustness ideas:
  (1) MULTI-LAG VAR: does adding lag-2/lag-3 returns as predictors add IC beyond lag-1?
      (if the DGP is VAR(1) -> no; if deeper lead-lag memory -> real new edge)
  (2) BAGGING the ridge: bootstrap-average the coefficient matrix -> variance reduction (floor)
  (3) PREDICTOR-SET ensemble: average {full, no-algo} forecasts
Measured by per-window IC + t (consistency) on 400-750. Baseline lead-lag IC ~0.077.
"""
import numpy as np, pandas as pd
from scipy import stats
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
RNG = np.random.default_rng(0)

def ewls_solve(X, Y, w, a):
    p = X.shape[1]
    XtWX = X.T @ (w[:, None] * X); XtWY = X.T @ (w[:, None] * Y)
    return np.linalg.solve(XtWX + a * np.eye(p), XtWY)

def multilag_fc(t, L=1, hl=1000, a=0.3):
    """predict next-day idio return from the last L lag-vectors of all 51 names (51*L predictors)."""
    r = lp[:, :t]; r = r[:, 1:] - r[:, :-1]              # (51, t-1)
    T = r.shape[1]
    if T < L + 50: return None
    # design: at time tau, predictor = [r[:,tau], r[:,tau-1],...,r[:,tau-L+1]] -> target r_idio[:,tau+1]
    rows = []; tars = []
    idx = np.arange(L - 1, T - 1)
    X = np.column_stack([r[:, idx - k].T for k in range(L)])   # (n, 51*L)
    Y = r[1:, idx + 1].T                                       # (n, 50)
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    B = ewls_solve(X - mx, Y - my, w, a)
    xin = np.concatenate([r[:, T - 1 - k] for k in range(L)])  # most recent L lag-vectors
    f = my + (xin - mx) @ B; return f - f.mean()

def bag_fc(t, hl=1000, a=0.3, nbag=15, frac=0.7):
    r = lp[:, :t]; r = r[:, 1:] - r[:, :-1]; T = r.shape[1]
    if T < 60: return None
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w0 = lam ** np.arange(n - 1, -1, -1)
    preds = []
    for _ in range(nbag):
        m = max(30, int(frac * n)); samp = RNG.choice(n, m, replace=True)
        Xs, Ys, ws = X[samp], Y[samp], w0[samp]; sw = ws.sum()
        mx = (ws[:, None] * Xs).sum(0) / sw; my = (ws[:, None] * Ys).sum(0) / sw
        B = ewls_solve(Xs - mx, Ys - my, ws, a)
        f = my + (xin - mx) @ B; preds.append((f - f.mean()) / (np.std(f - f.mean()) + 1e-12))
    return np.mean(preds, 0)

def predset_ens(t, hl=1000, a=0.3):
    fs = []
    for drop_algo in (False, True):
        r = lp[:, :t]; r = r[:, 1:] - r[:, :-1]
        X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
        if drop_algo: X = X[:, 1:]; xin = xin[1:]
        n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
        mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
        B = ewls_solve(X - mx, Y - my, w, a)
        f = my + (xin - mx) @ B; fs.append((f - f.mean()) / (np.std(f - f.mean()) + 1e-12))
    return np.mean(fs, 0)

def ic_windows(fc, legs):
    out = []
    for S, E in legs:
        ics = []
        for t in range(max(S, 120), min(E, nt - 1)):
            s = fc(t)                                     # forecast uses data thru day t-1, predicts RET[:,t-1]
            if s is None: continue
            fwd = RET[1:, t - 1]                          # aligned: the return the forecast targets
            if s.std() > 1e-12 and fwd.std() > 1e-12: ics.append(np.corrcoef(s, fwd)[0, 1])
        ics = np.array(ics)
        if len(ics) < 5: out.append((np.nan, np.nan)); continue
        out.append((ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))))
    return out

legs = [(250, 500), (350, 600), (500, 750), (400, 749)]
print(f"Per-window IC (t) on {[f'{a}-{b}' for a,b in legs]}; consistency = stable across legs.\n")
cands = [
    ("lag-1 ridge (baseline)", lambda t: multilag_fc(t, L=1)),
    ("lag-2 VAR", lambda t: multilag_fc(t, L=2)),
    ("lag-3 VAR", lambda t: multilag_fc(t, L=3)),
    ("bagged ridge (15x)", lambda t: bag_fc(t)),
    ("predictor-set ensemble", predset_ens),
]
print(f"{'method':<26}" + "".join(f"{f'{a}-{b}':>13}" for a, b in legs) + f"{'mean':>8}{'std':>7}")
for name, fc in cands:
    res = ic_windows(fc, legs)
    ics = [r[0] for r in res]
    cells = "".join(f"{ic:7.4f}({t:4.1f})" for ic, t in res)
    print(f"{name:<26}{cells}{np.mean(ics):8.4f}{np.std(ics[:3]):7.4f}")
print("\nverdict: if lag-2/lag-3 IC <= lag-1, the process is VAR(1) (no deeper lead-lag).")
print("bagging/ensemble help ROBUSTNESS (steadier across legs) even if mean IC is flat.")
