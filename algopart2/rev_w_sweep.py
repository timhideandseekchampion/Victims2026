"""rev_w_sweep.py — does REV_W=7 beat 10 by SCORE (not just IC)? Test the full book
(ensemble lead-lag + reversion blend + net-$ ALGO gate) at several REV_W on the graded leg
AND rolling 250d windows, so we don't get fooled by a single window."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)
ENS=[250,500,1000,2000]; BLEND=0.30; CONTRA_DOL=1_000_000; CONTRA_K=30; CONTRA_WZ=60; ALGO_LL=50_000

_rc={}
def ridge_z(t, hl, a=0.1):
    key=(t,hl)
    if key in _rc: return _rc[key]
    lp=logp[:,:t]; r=lp[:,1:]-lp[:,:-1]; X=r[:,:-1].T; Y=r[1:,1:].T; xin=r[:,-1]
    n=X.shape[0]; lam=0.5**(1/hl); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
    mx=(w[:,None]*X).sum(0)/sw; my=(w[:,None]*Y).sum(0)/sw; Xc=X-mx; Yc=Y-my
    B=np.linalg.solve(Xc.T@(w[:,None]*Xc)+a*np.eye(nInst), Xc.T@(w[:,None]*Yc))
    f=my+(xin-mx)@B; f=f-f.mean(); v=f/(f.std()+1e-12); _rc[key]=v; return v

def run(rev_w, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur=prc[:,t-1]; pos=np.zeros(nInst)
        if t<E and t>=96:
            a=np.mean([ridge_z(t,hl) for hl in ENS],0)
            rr=logp[1:,t-1]-logp[1:,t-1-rev_w]; rr=rr-rr.mean(); rv=-rr/(rr.std()+1e-12)
            wz=(1-BLEND)*a+BLEND*rv; pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
            ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
            net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]
            if abs(net)>=ALGO_LL: av=float(np.sign(net)*cap)
            else:
                lpA=logp[0,:t]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
                z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
                av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
            pos[0]=av; lim=(dlr/cur).astype(int); pos=np.clip(pos,-lim,lim).astype(int)
        else: pos=cp.copy()
        dP=pos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=pos.copy()
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    pll=np.array(pll); mu,sd=pll.mean(),pll.std()
    sc=mu*(np.sqrt(250)*mu/sd)**2/((np.sqrt(250)*mu/sd)**2+1) if mu>0 else mu
    return mu,sd,np.sqrt(250)*mu/sd,sc

print("graded leg 500-750:")
print(f"  {'REV_W':<7}{'mean$':>8}{'std$':>8}{'Sharpe':>8}{'SCORE':>8}")
for rw in (5,7,10,15):
    mu,sd,sr,sc=run(rw,500,750)
    star="  <-- current" if rw==10 else ("  <-- proposed" if rw==7 else "")
    print(f"  {rw:<7}{mu:>8.0f}{sd:>8.0f}{sr:>8.2f}{sc:>8.0f}{star}")

print("\nrolling 250d windows (mean SCORE / floor / #windows where REV_W beats 10):")
ends=list(range(400,nDays+1,10))
base={e:run(10,e-250,e)[3] for e in ends if e-250>=96}
print(f"  {'REV_W':<7}{'meanSCORE':>11}{'floor':>8}{'wins_vs_10':>12}")
for rw in (5,7,10,15):
    ss={e:run(rw,e-250,e)[3] for e in ends if e-250>=96}
    arr=np.array(list(ss.values()))
    wins=sum(1 for e in ss if ss[e]>base[e]) if rw!=10 else 0
    print(f"  {rw:<7}{arr.mean():>11.0f}{arr.min():>8.0f}{(str(wins)+'/'+str(len(ss))) if rw!=10 else '--':>12}")
