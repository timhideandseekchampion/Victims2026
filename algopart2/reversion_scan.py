"""reversion_scan.py — is there a STRONG single-name mean-reversion edge, and do we already have it?
(1) IC of pure reversion by horizon k: forecast = -zscore(trailing k-day return), all causal.
(2) Is the lead-lag ridge SECRETLY an own-name reversion model? Decompose its forecast into the
    OWN-name term (diagonal of B: name i's own return -> name i's next return) vs the PEER term
    (everything off-diagonal). If the diagonal is strongly negative and carries most of the IC,
    then 'lead-lag' is largely reversion in disguise and we already trade it hard.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
logp = np.log(prc)
ENS = [250, 500, 1000, 2000]

def corr(a, b):
    a = a - a.mean(); b = b - b.mean(); d = np.sqrt((a@a)*(b@b))
    return float(a@b/d) if d > 1e-12 else 0.0

def tstat(x): x = np.asarray(x); return x.mean()/(x.std(ddof=1)/np.sqrt(len(x)) + 1e-12)

# ---------- (1) reversion IC by horizon ----------
print("(1) pure reversion signal  -zscore(trailing k-day return)  -> next-day idio return")
print(f"    {'k (days)':<10}{'mean IC':>10}{'t-stat':>9}")
S, E = 100, nDays
for k in (1, 2, 3, 5, 7, 10, 15, 20, 30):
    ics = []
    for t in range(max(S, k+2), E):
        rev = -(logp[1:, t-1] - logp[1:, t-1-k])
        fwd = logp[1:, t] - logp[1:, t-1]
        ics.append(corr(rev, fwd))
    print(f"    {k:<10}{np.mean(ics):>+10.4f}{tstat(ics):>9.2f}")

# ---------- (2) lead-lag ridge: own-name (diagonal) vs peer (off-diagonal) ----------
def ewls(X, Y, hl, a=0.1):
    n, p = X.shape; lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw
    Xc, Yc = X-mx, Y-my; XtWX = Xc.T@(w[:, None]*Xc)
    B = np.linalg.solve(XtWX + (1e-8*np.trace(XtWX)/p + a)*np.eye(p), Xc.T@(w[:, None]*Yc))
    return B, mx, my

print("\n(2) is 'lead-lag' really own-name reversion? decompose the ensemble ridge forecast")
print(f"    {'window':<12}{'diag(B) avg':>12}{'IC full':>9}{'IC own':>9}{'IC peer':>9}")
for (a, b) in ((300, 500), (500, 750), (250, 750)):
    ic_full = []; ic_own = []; ic_peer = []; diags = []
    for t in range(a, b, 3):
        r = logp[:, :t]; r = r[:, 1:] - r[:, :-1]
        X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
        f_full = np.zeros(50); f_own = np.zeros(50)
        dsum = 0.0
        for hl in ENS:
            B, mx, my = ewls(X, Y, hl)
            f_full += (my + (xin - mx) @ B) / len(ENS)
            # own-name term: predictor for target j is instrument j+1 (0 = index)
            diag = np.array([B[j+1, j] for j in range(50)])
            dsum += diag.mean() / len(ENS)
            f_own += (my + (xin[1:] - mx[1:]) * diag) / len(ENS)
        fwd = logp[1:, t] - logp[1:, t-1]
        ic_full.append(corr(f_full - f_full.mean(), fwd))
        ic_own.append(corr(f_own - f_own.mean(), fwd))
        ic_peer.append(corr((f_full - f_own) - (f_full - f_own).mean(), fwd))
        diags.append(dsum)
    print(f"    {f'{a}-{b}':<12}{np.mean(diags):>+12.4f}{np.mean(ic_full):>+9.4f}{np.mean(ic_own):>+9.4f}{np.mean(ic_peer):>+9.4f}")

print("\n(diag(B) avg < 0  => the ridge shorts a name's own recent return = built-in own-name reversion)")
