"""why_neg_ic.py — WHY does the ridge's common component anti-predict ALGO?
Hypothesis: (1) ALGO mean-reverts day-to-day (negative lag-1 autocorr).
            (2) the common component of the forecast is ~ today's market move re-expressed (momentum-ish).
            (1)+(2) => a level/momentum aggregate anti-predicts a mean-reverting index."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = [250, 500, 1000, 2000]

def common(t):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; out = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1,-1,-1); sw = w.sum()
        mx = (w[:,None]*X).sum(0)/sw; my = (w[:,None]*Y).sum(0)/sw
        Xc = X-mx; Yc = Y-my
        B = np.linalg.solve(Xc.T@(w[:,None]*Xc)+0.1*np.eye(nInst), Xc.T@(w[:,None]*Yc))
        out.append((my+(xin-mx)@B).mean())
    return float(np.mean(out))

for lbl, (S, E) in {"500-750": (500, 749), "400-500": (400, 499), "250-400": (250, 399)}.items():
    algo_ret = r_all[0, S:E]                 # ALGO daily return over window
    ac1 = np.corrcoef(algo_ret[:-1], algo_ret[1:])[0, 1]     # lag-1 autocorrelation of ALGO
    cc = np.array([common(t) for t in range(S, E)])
    today = np.array([r_all[0, t-1] for t in range(S, E)])   # today's ALGO move (input to forecast)
    fwd   = np.array([r_all[0, t]   for t in range(S, E)])   # tomorrow's ALGO move (what we predict)
    print(f"[{lbl}]  ALGO lag-1 autocorr = {ac1:+.3f}   "
          f"corr(common, today's ALGO move) = {np.corrcoef(cc, today)[0,1]:+.3f}   "
          f"corr(common, tomorrow) = {np.corrcoef(cc, fwd)[0,1]:+.3f}")
