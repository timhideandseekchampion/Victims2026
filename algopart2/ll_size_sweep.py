"""ll_size_sweep.py — on a trigger (|frac|>=0.12), size the lead-lag ALGO leg to a TARGET dollar
(instead of always slamming the full $100k cap). Does a smaller ~$60k trigger keep the mean edge
while cutting the std the full-cap version added? Two modes:
  LL-only : trigger => +/-$LL_DOL lead-lag, rest of cap flat
  LL+rev  : trigger => +/-$LL_DOL lead-lag PLUS reversion filling the remaining ($100k-$LL_DOL) room
Reports mean / std / annualised Sharpe / score, graded leg + rolling avg."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); ENS = [250, 500, 1000, 2000]; CONTRA_DOL = 1_000_000; GATE = 0.12

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
    idio = np.sign(wz)*(dlr[1:]/cur[1:]); cap = dlr[0]/cur[0]
    frac = float(np.mean(np.sign(wz)))
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev_dol = -np.clip(z, -3, 3)/3.0*CONTRA_DOL                    # reversion target in $
    return idio, frac, rev_dol, cap, cur

print("precomputing ..."); SIG = {t: daily(t) for t in range(120, nDays+1)}

def pos_for(t, ll_dol, mode):
    idio, frac, rev_dol, cap, cur = SIG[t]; p = np.zeros(nInst); p[1:] = idio; c0 = cur[0]
    if ll_dol is None:                                            # baseline reversion
        dol = rev_dol
    elif abs(frac) >= GATE:                                       # trigger
        ll = np.sign(frac)*ll_dol
        if mode == "LLrev":
            room = 100_000 - ll_dol
            dol = ll + np.clip(rev_dol, -room, room)
        else:
            dol = ll
    else:
        dol = rev_dol                                            # gate off -> reversion default
    p[0] = np.clip(dol/c0, -cap, cap)
    lim = (dlr/cur).astype(int); return np.clip(p, -lim, lim).astype(int)

def metrics(ll_dol, mode, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur = prc[:, t-1]; newPos = pos_for(t, ll_dol, mode) if t < E else cp
        dP = newPos-cp; cash -= cur.dot(dP)+comm; comm = np.sum(cur*np.abs(dP)*commRate); cp = newPos
        pl = cash+cp.dot(cur)-value; value = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250)*mu/sd; sc = mu*sr**2/(sr**2+1) if mu > 0 else mu
    return mu, sd, sr, sc

ENDS = list(range(380, nDays+1, 10))
def roll(ll_dol, mode):
    M = np.array([metrics(ll_dol, mode, E-250, E) for E in ENDS]); return M.mean(0)

print(f"\n{'config':<26}{'mean':>8}{'std':>9}{'Sharpe':>8}{'score':>8}   [500-750 graded]")
gb = metrics(None, None, 501, 750)
print(f"{'baseline reversion':<26}{gb[0]:>8.1f}{gb[1]:>9.1f}{gb[2]:>8.2f}{gb[3]:>8.1f}")
for mode in ("LLonly", "LLrev"):
    for d in (40_000, 60_000, 80_000, 100_000):
        m = metrics(d, mode, 501, 750)
        print(f"{mode+' $'+str(d//1000)+'k':<26}{m[0]:>8.1f}{m[1]:>9.1f}{m[2]:>8.2f}{m[3]:>8.1f}")

print(f"\n{'config':<26}{'std':>9}{'Sharpe':>8}{'score':>8}   [rolling 38-window avg]")
rb = roll(None, None); print(f"{'baseline reversion':<26}{rb[1]:>9.1f}{rb[2]:>8.2f}{rb[3]:>8.1f}")
for mode in ("LLonly", "LLrev"):
    for d in (40_000, 60_000, 80_000, 100_000):
        r = roll(d, mode)
        print(f"{mode+' $'+str(d//1000)+'k':<26}{r[1]:>9.1f}{r[2]:>8.2f}{r[3]:>8.1f}")
