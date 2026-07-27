"""blend_sweep.py — fast blend sweep for SAFE_lldollar: ridge computed ONCE per day (blend-independent),
then wz = (1-b)*leadlag_z + b*reversion recombined cheaply per blend. graded + rolling score."""
import sys, numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); ENS = [250, 500, 1000, 2000]; CONTRA_DOL = 1_000_000; GATE = 50_000

def daily(t):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]; cur = prc[:, t-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]; fs = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
        mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
        eps = 1e-8*np.trace(Xc.T@(w[:, None]*Xc))/X.shape[1]
        B = np.linalg.solve(Xc.T@(w[:, None]*Xc)+(eps+0.1)*np.eye(nInst), Xc.T@(w[:, None]*Yc))
        f = my+(xin-mx)@B; d = f-f.mean(); fs.append(d/(d.std()+1e-12))
    z = np.mean(fs, 0)                                            # leadlag ensemble z (blend-independent)
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12)
    cap = dlr[0]/cur[0]
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]; zz = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev_algo = -np.clip(zz, -3, 3)/3.0*CONTRA_DOL
    return z, rv, rev_algo, cap, cur

print("precomputing ridge once/day ...", flush=True)
SIG = {t: daily(t) for t in range(120, nDays+1)}

def pos(t, blend):
    z, rv, rev_algo, cap, cur = SIG[t]
    wz = (1-blend)*z + blend*rv
    ilim = (dlr[1:]/cur[1:]).astype(int); idio = np.clip(np.sign(wz)*(dlr[1:]/cur[1:]), -ilim, ilim).astype(int)
    net = float((idio*cur[1:]).sum())
    av = np.sign(net)*CONTRA_DOL if abs(net) >= GATE else rev_algo
    p = np.zeros(nInst); p[1:] = idio; p[0] = np.clip(av/cur[0], -cap, cap)
    lim = (dlr/cur).astype(int); return np.clip(p, -lim, lim).astype(int)

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def metrics(blend, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur = prc[:, t-1]; nP = pos(t, blend) if t < E else cp
        dP = nP-cp; cash -= cur.dot(dP)+comm; comm = np.sum(cur*np.abs(dP)*commRate); cp = nP
        pl = cash+cp.dot(cur)-value; value = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std(); return mu, sd, np.sqrt(250)*mu/sd, score(mu, sd)

ENDS = list(range(380, nDays+1, 10))
print(f"{'blend':<8}{'mean':>8}{'std':>9}{'Sharpe':>8}{'score(graded)':>15}{'score(rolling)':>16}", flush=True)
for b in (0.20, 0.25, 0.30, 0.35, 0.40):
    m = metrics(b, 501, 750); r = np.array([metrics(b, E-250, E) for E in ENDS])[:, 3].mean()
    mark = "  <-- 0.25 (current file)" if abs(b-0.25) < 1e-9 else ("  <-- 0.35 (asked)" if abs(b-0.35) < 1e-9 else "")
    print(f"{b:<8}{m[0]:>8.1f}{m[1]:>9.1f}{m[2]:>8.2f}{m[3]:>15.1f}{r:>16.1f}{mark}", flush=True)
