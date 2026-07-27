"""feast_dial.py — the feast/famine dial. The $100k ALGO leg can either AMPLIFY the book's
directional tilt (current net-$ gate) or HEDGE it (neutralize the idio book's net index exposure).
Show score AND the downside tail for each choice, so the trade-off is explicit:
  amplify (binary) = max score, max feast AND famine   |   hedge = market-neutral, cuts both tails
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)
HLS=[250,500,1000,2000]; BLEND=0.30; REV_W=10; CONTRA_DOL=1_000_000; CONTRA_K=30; CONTRA_WZ=60; ALGO_LL=50_000

_wc={}
def wz_at(t):
    if t in _wc: return _wc[t]
    lp=logp[:,:t]; r=lp[:,1:]-lp[:,:-1]; X=r[:,:-1].T; Y=r[1:,1:].T; xin=r[:,-1]; n=X.shape[0]; fs=[]
    for hl in HLS:
        lam=0.5**(1/hl); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
        mx=(w[:,None]*X).sum(0)/sw; my=(w[:,None]*Y).sum(0)/sw; Xc=X-mx; Yc=Y-my
        B=np.linalg.solve(Xc.T@(w[:,None]*Xc)+0.1*np.eye(nInst), Xc.T@(w[:,None]*Yc))
        f=my+(xin-mx)@B; f=f-f.mean(); fs.append(f/(f.std()+1e-12))
    a=np.mean(fs,0); rr=logp[1:,t-1]-logp[1:,t-1-REV_W]; rr=rr-rr.mean(); rv=-rr/(rr.std()+1e-12)
    wz=(1-BLEND)*a+BLEND*rv; _wc[t]=wz; return wz

def positions(t, mode):
    cur=prc[:,t-1]; pos=np.zeros(nInst); wz=wz_at(t); pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
    ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
    net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]
    lpA=logp[0,:t]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
    z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
    fade=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    trend=float(np.clip(+np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))  # follow (TSMOM), = -fade
    if mode=="amplify":  av=float(np.sign(net)*cap) if abs(net)>=ALGO_LL else fade
    elif mode=="hedge":  av=float(np.clip(-net/cur[0], -cap, cap))          # neutralize idio tilt
    elif mode=="half":   av=float(np.clip(-0.5*net/cur[0], -cap, cap))      # partial hedge
    elif mode=="off":    av=0.0
    elif mode=="trend":  av=trend                                           # index time-series momentum
    else:                av=fade
    pos[0]=av; lim=(dlr/cur).astype(int); return np.clip(pos,-lim,lim).astype(int)

def run(mode,S,E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S,E+1):
        cur=prc[:,t-1]; newPos=positions(t,mode) if t<E else cp
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    return np.array(pll)

def sc(p):
    mu,sd=p.mean(),p.std(); return mu*(np.sqrt(250)*mu/sd)**2/((np.sqrt(250)*mu/sd)**2+1) if mu>0 else mu

print("ALGO leg dial, 500-750  (worst10 = avg of 10 worst days = the 'famine')")
print(f"{'mode':<9}{'SCORE':>7}{'mean':>7}{'std':>7}{'Sharpe':>7}{'worstDay':>10}{'worst10':>9}{'best10':>9}")
for m in ("amplify","half","hedge","off","fade","trend"):
    p=run(m,500,750); o=np.sort(p)
    tag="  <-- shipped" if m=="amplify" else ("  <-- index TSMOM (gate default=fade)" if m=="trend" else "")
    print(f"{m:<9}{sc(p):>7.0f}{p.mean():>7.0f}{p.std():>7.0f}{np.sqrt(250)*p.mean()/p.std():>7.2f}"
          f"{p.min():>10.0f}{o[:10].mean():>9.0f}{o[-10:].mean():>9.0f}{tag}")

print("\nrolling 250d windows (mean SCORE / floor / worst-day-avg):")
ends=[e for e in range(400,nDays+1,25) if e-250>=96]
print(f"{'mode':<9}{'meanSCORE':>11}{'floor':>8}{'avg worstDay':>14}")
for m in ("amplify","half","hedge","off"):
    scs=[]; wds=[]
    for e in ends:
        p=run(m,e-250,e); scs.append(sc(p)); wds.append(p.min())
    print(f"{m:<9}{np.mean(scs):>11.0f}{np.min(scs):>8.0f}{np.mean(wds):>14.0f}")
