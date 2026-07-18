"""metrics_compare.py — did STD and SHARPE improve with the gated lead-lag ALGO leg?
Reports daily-PnL mean, std, annualised Sharpe (sqrt(250)*mu/std) and score for
baseline (reversion ALGO leg) vs gated lead-lag (|frac|>=0.12), per window + rolling."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); ENS = [250, 500, 1000, 2000]; FRAC_SCALE, CONTRA_DOL = 0.09, 1_000_000

def daily(t):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]; cur = prc[:, t-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]; fs = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
        mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
        eps = 1e-8*np.trace(Xc.T@(w[:, None]*Xc))/X.shape[1]
        B = np.linalg.solve(Xc.T@(w[:, None]*Xc)+(eps+0.1)*np.eye(nInst), Xc.T@(w[:, None]*Yc))
        f = my+(xin-mx)@B; d = f-f.mean(); fs.append(d/(d.std()+1e-12))
    wz = np.mean(fs, 0)
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12)
    wz = 0.7*wz + 0.3*rv
    idio = np.sign(wz)*(dlr[1:]/cur[1:]); cap = dlr[0]/cur[0]; notl = CONTRA_DOL/cur[0]
    frac = float(np.mean(np.sign(wz)))
    ll_av = np.clip(frac/FRAC_SCALE, -3, 3)/3.0*notl
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev_av = -np.clip(z, -3, 3)/3.0*notl
    return idio, frac, ll_av, rev_av, cap, cur

print("precomputing ..."); SIG = {t: daily(t) for t in range(120, nDays+1)}

def pos_for(t, gate):
    idio, frac, ll_av, rev_av, cap, cur = SIG[t]; p = np.zeros(nInst); p[1:] = idio
    av = ll_av if (gate is not None and abs(frac) >= gate) else rev_av
    p[0] = np.clip(av, -cap, cap); lim = (dlr/cur).astype(int)
    return np.clip(p, -lim, lim).astype(int)

def metrics(gate, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur = prc[:, t-1]; newPos = pos_for(t, gate) if t < E else cp
        dP = newPos-cp; cash -= cur.dot(dP)+comm; comm = np.sum(cur*np.abs(dP)*commRate); cp = newPos
        pl = cash+cp.dot(cur)-value; value = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250)*mu/sd; sc = mu*sr**2/(sr**2+1) if mu > 0 else mu
    return mu, sd, sr, sc

WINDOWS = {"500-750 (GRADED)": (501, 750), "400-650": (401, 650), "250-500": (251, 500)}
print(f"\n{'window':<18}{'':>4}{'mean':>8}{'std':>9}{'Sharpe':>8}{'score':>8}")
for wl, (S, E) in WINDOWS.items():
    b = metrics(None, S, E); g = metrics(0.12, S, E)
    print(f"{wl:<18}{'base':>4}{b[0]:>8.1f}{b[1]:>9.1f}{b[2]:>8.2f}{b[3]:>8.1f}")
    print(f"{'':<18}{'LL':>4}{g[0]:>8.1f}{g[1]:>9.1f}{g[2]:>8.2f}{g[3]:>8.1f}"
          f"   d: std {g[1]-b[1]:+.0f}  Sharpe {g[2]-b[2]:+.2f}  score {g[3]-b[3]:+.1f}")

ENDS = list(range(380, nDays+1, 10))
B = np.array([metrics(None, E-250, E) for E in ENDS]); G = np.array([metrics(0.12, E-250, E) for E in ENDS])
print(f"\nrolling {len(ENDS)} windows (mean across windows):")
print(f"  {'':>4}{'std':>9}{'Sharpe':>8}{'score':>8}")
print(f"  {'base':>4}{B[:,1].mean():>9.1f}{B[:,2].mean():>8.2f}{B[:,3].mean():>8.1f}")
print(f"  {'LL':>4}{G[:,1].mean():>9.1f}{G[:,2].mean():>8.2f}{G[:,3].mean():>8.1f}"
      f"   d: std {G[:,1].mean()-B[:,1].mean():+.0f}  Sharpe {G[:,2].mean()-B[:,2].mean():+.2f}")
