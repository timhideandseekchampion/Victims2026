"""ml_probe.py — would a nonlinear ML challenger help, and should it sit on the rotation bench?
ML here = Random Fourier Features + forgetting ridge (an RBF-kernel-ridge approximation; numpy only).
Theory: the DGP is Gaussian VAR(1) => E[Y|X] is LINEAR => ML cannot beat the linear ridge in
expectation, only overfit. So on real data ML's realized IC should be <= champion (it benches, 694
preserved). Its value is purely insurance for a NONLINEAR regime. Tests:
  (A) REAL 500-750: realized OOS IC of RFF-ML vs the linear ridge (expect ML <= linear -> benches)
  (B) SYNTHETIC NONLINEAR regime (next return = quadratic in a hidden projection, invisible to linear):
      does ML's IC beat linear's -> would the rotation correctly pick it up?
All causal (fit on past, predict next, forgetting-weighted). No look-ahead.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape

def lin_forecast(P, hl=1000, a=0.1):
    """single-HL linear forgetting ridge (the champion's core), causal."""
    lp = np.log(P); r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]
    lam = 0.5 ** (1/hl); w = lam ** np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
    B = np.linalg.solve(Xc.T@(w[:, None]*Xc) + a*np.eye(nInst), Xc.T@(w[:, None]*Yc))
    f = my + (xin-mx)@B; return f - f.mean()

def rff_forecast(P, D=256, gamma=6.0, hl=1000, a=1.0, seed=0):
    """Random-Fourier-Feature (RBF kernel) forgetting ridge — a nonlinear ML predictor, causal."""
    lp = np.log(P); r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n, p = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(0, gamma, (p, D)); bph = rng.uniform(0, 2*np.pi, D)
    Z = np.sqrt(2.0/D) * np.cos(X@W + bph)                       # n x D nonlinear features
    zin = np.sqrt(2.0/D) * np.cos(xin@W + bph)
    lam = 0.5 ** (1/hl); w = lam ** np.arange(n-1, -1, -1); sw = w.sum()
    mz = (w[:, None]*Z).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Zc = Z-mz; Yc = Y-my
    B = np.linalg.solve(Zc.T@(w[:, None]*Zc) + a*np.eye(D), Zc.T@(w[:, None]*Yc))
    f = my + (zin-mz)@B; return f - f.mean()

def corr(a, b):
    a = a-a.mean(); b = b-b.mean(); d = np.sqrt((a@a)*(b@b)); return float(a@b/d) if d > 1e-12 else 0.0
def tstat(x): x = np.asarray(x); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x))+1e-12)

def realized_ic(P, fc, S, E):
    lp = np.log(P); ics = []
    for t in range(S, E):
        f = fc(P[:, :t]); fwd = lp[1:, t] - lp[1:, t-1]; ics.append(corr(f, fwd))
    return np.array(ics)

print("(A) REAL 500-750: realized OOS IC (expect ML <= linear on Gaussian VAR(1) data)")
lic = realized_ic(prc, lin_forecast, 500, 750)
for g in (2.0, 6.0, 15.0):
    ric = realized_ic(prc, lambda P: rff_forecast(P, gamma=g), 500, 750)
    print(f"    linear IC {lic.mean():+.4f} (t={tstat(lic):.1f})   RFF-ML(gamma={g:>4}) IC {ric.mean():+.4f} (t={tstat(ric):.1f})"
          f"   ML-minus-linear {ric.mean()-lic.mean():+.4f}")

# (B) synthetic NONLINEAR regime: next idio return = quadratic in a hidden projection of the cross-section
def make_nonlin(D_days=400, seed=1, strength=0.9):
    rng = np.random.default_rng(seed); logp = np.log(prc[:, :nDays-D_days]).copy()
    vol = np.diff(np.log(prc[1:]), axis=1).std(); names = logp[1:].copy()
    v = rng.standard_normal(50); v /= np.linalg.norm(v)            # hidden projection direction
    for _ in range(D_days):
        r_prev = names[:, -1] - names[:, -2]                       # yesterday's idio returns
        proj = r_prev @ v                                          # scalar projection
        signal = strength * (proj**2 - (vol**2)) / (vol + 1e-9)    # QUADRATIC -> linear-invisible (symmetric)
        drift = signal * v                                         # push along v by the squared projection
        drift -= drift.mean(); noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :nDays-D_days] = prc[:, :nDays-D_days]; return full

full = make_nonlin()
S2 = nDays - 200
print(f"\n(B) SYNTHETIC NONLINEAR regime (quadratic dependence, invisible to a linear model):")
lic2 = realized_ic(full, lin_forecast, S2, full.shape[1]-1)
ric2 = realized_ic(full, lambda P: rff_forecast(P, gamma=6.0), S2, full.shape[1]-1)
print(f"    linear IC {lic2.mean():+.4f} (t={tstat(lic2):.1f})   RFF-ML IC {ric2.mean():+.4f} (t={tstat(ric2):.1f})"
      f"   ML-minus-linear {ric2.mean()-lic2.mean():+.4f}")
print("\nread: (A) ML-minus-linear <= 0 on real data -> ML benches (no cost, 694 preserved).")
print("      (B) ML-minus-linear >> 0 and significant -> the rotation WOULD switch to ML if the DGP goes nonlinear.")
