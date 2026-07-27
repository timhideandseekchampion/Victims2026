"""confirm_sizing.py — two questions:
(1) how much does STD improve if we MATCH the book (size ALGO = net$) vs full $100k?
(2) confirmation sizing: lead-lag puts $50k on; if the REVERSION leg points the SAME way, go to $100k;
    if reversion disagrees, stay at $50k. (Both signals agree -> high conviction -> full size.)
Reports mean/std/Sharpe/score on graded leg + rolling avg, and how often the two legs agree."""
import numpy as np, pandas as pd
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
    wz = np.mean(fs, 0)
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12); wz = 0.7*wz+0.3*rv
    idio = np.sign(wz)*(dlr[1:]/cur[1:]); il = (dlr[1:]/cur[1:]).astype(int); idio = np.clip(idio, -il, il).astype(int)
    net = float((idio*cur[1:]).sum()); cap = dlr[0]/cur[0]
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev = -np.clip(z, -3, 3)/3.0*CONTRA_DOL                       # reversion target ($), sign = its direction
    return idio, net, rev, cap, cur

print("precomputing ..."); SIG = {t: daily(t) for t in range(120, nDays+1)}

def pos_for(t, mode):
    idio, net, rev, cap, cur = SIG[t]; p = np.zeros(nInst); p[1:] = idio; c0 = cur[0]
    on = abs(net) >= GATE; lld = np.sign(net); agree = on and (np.sign(rev) == lld)
    if not on:                    dol = rev                       # gate off -> reversion default
    elif mode == "full":          dol = lld*CONTRA_DOL            # current: always full $100k
    elif mode == "match":         dol = net                      # size = book skew
    elif mode == "confirm":       dol = lld*(100_000 if agree else 50_000)   # $50k, ->$100k if rev agrees
    elif mode == "confirm_match": dol = (lld*100_000 if agree else net)      # matched, ->$100k if rev agrees
    p[0] = np.clip(dol/c0, -cap, cap); lim = (dlr/cur).astype(int)
    return np.clip(p, -lim, lim).astype(int)

def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def metrics(mode, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur = prc[:, t-1]; nP = pos_for(t, mode) if t < E else cp
        dP = nP-cp; cash -= cur.dot(dP)+comm; comm = np.sum(cur*np.abs(dP)*commRate); cp = nP
        pl = cash+cp.dot(cur)-value; value = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std(); return mu, sd, np.sqrt(250)*mu/sd, score(mu, sd)

# how often do the two legs agree on a trigger day?
on=agree=0
for t in range(131, nDays+1):
    _, net, rev, _, _ = SIG[t]
    if abs(net) >= GATE:
        on += 1; agree += (np.sign(rev) == np.sign(net))
print(f"\non {on} trigger days: reversion AGREES with lead-lag {agree} ({100*agree/on:.0f}%), "
      f"disagrees {on-agree} ({100*(on-agree)/on:.0f}%)\n")

ENDS = list(range(380, nDays+1, 10))
print(f"{'sizing':<26}{'mean':>8}{'std':>9}{'Sharpe':>8}{'score':>8}   [rolling score]")
for mode, lbl in [("full","full $100k (current)"),("match","match net$"),
                  ("confirm","$50k, ->$100k if rev agrees"),("confirm_match","match, ->$100k if rev agrees")]:
    m = metrics(mode, 501, 750); r = np.array([metrics(mode, E-250, E) for E in ENDS]).mean(0)
    print(f"{lbl:<26}{m[0]:>8.1f}{m[1]:>9.1f}{m[2]:>8.2f}{m[3]:>8.1f}      {r[3]:>8.1f}")
