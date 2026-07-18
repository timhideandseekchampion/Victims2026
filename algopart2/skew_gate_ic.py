"""skew_gate_ic.py — is the tilt's edge on ALGO concentrated on BIG-skew days?
For each decision day compute frac = mean(sign(wz)); measure IC(frac, next-day ALGO ret)
restricted to days with |frac| >= threshold, and how many days survive each gate."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = [250, 500, 1000, 2000]

def frac_of(t, blend=0.3):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]
    vs = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1,-1,-1); sw = w.sum()
        mx = (w[:,None]*X).sum(0)/sw; my = (w[:,None]*Y).sum(0)/sw
        Xc = X-mx; Yc = Y-my
        B = np.linalg.solve(Xc.T@(w[:,None]*Xc)+0.1*np.eye(nInst), Xc.T@(w[:,None]*Yc))
        f = my+(xin-mx)@B; d = f-f.mean(); vs.append(d/(d.std()+1e-12))
    v = np.mean(vs, 0)
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12)
    wz = (1-blend)*v + blend*rv
    return float(np.mean(np.sign(wz)))

for lbl, (S, E) in {"500-750": (500, 749), "400-500": (400, 499), "250-400": (250, 399)}.items():
    fr = np.array([frac_of(t) for t in range(S, E)])
    y  = np.array([float(r_all[0, t]) for t in range(S, E)])
    print(f"\n[{lbl}]  |frac| distribution: median {np.median(np.abs(fr)):.3f}  90th pct {np.quantile(np.abs(fr),0.9):.3f}")
    print(f"  {'gate |frac|>=':<16}{'n days':>8}{'IC(frac, next-day ALGO)':>26}{'IC*sign(frac) [aligned]':>26}")
    for g in (0.0, 0.06, 0.12, 0.18, 0.24):
        m = np.abs(fr) >= g
        if m.sum() < 8:
            print(f"  {g:<16.2f}{m.sum():>8}{'  (too few)':>26}"); continue
        ic = np.corrcoef(fr[m], y[m])[0,1]
        aligned = np.mean(np.sign(fr[m]) * y[m]) / (np.std(y[m])+1e-12)  # avg aligned move / vol
        print(f"  {g:<16.2f}{m.sum():>8}{ic:>26.3f}{aligned:>26.3f}")
