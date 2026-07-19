"""factor_exposure.py — in a COMMON drawdown (the shared index/factor moves, everyone loses),
who loses least = who carries the least net factor exposure. Measure each ALGO-leg choice's
daily NET FACTOR EXPOSURE ($) = idio net$ (names have beta~1 to index) + ALGO position*price0
(ALGO *is* the index). Bigger |exposure| = you swing more with the common factor = you lose more
when it moves against you. Also: book PnL std on the biggest index-move days (the common shocks).
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
    if mode=="amplify": av=float(np.sign(net)*cap) if abs(net)>=ALGO_LL else fade
    elif mode=="half":  av=float(np.clip(-0.5*net/cur[0],-cap,cap))
    elif mode=="hedge": av=float(np.clip(-net/cur[0],-cap,cap))
    else:               av=0.0
    pos[0]=av; lim=(dlr/cur).astype(int); return np.clip(pos,-lim,lim).astype(int), net

S,E=500,750
idxret=np.array([logp[0,t-1]-logp[0,t-2] for t in range(S+1,E+1)])   # index move on each scored day
print(f"index daily move std over {S}-{E}: {idxret.std()*100:.2f}%\n")
print(f"{'mode':<9}{'|netfactor$|':>13}{'std netfac$':>12}{'PnL std':>9}{'PnL std|bigIdx':>15}{'worst10':>9}")
big = np.argsort(-np.abs(idxret))[:40]                                # the 40 biggest index-move days
for mode in ("amplify","half","hedge","off"):
    cp=np.zeros(nInst); cash=0.0; value=0.0; comm=0.0; pnl=[]; fexp=[]
    for t in range(S,E+1):
        cur=prc[:,t-1]
        if t>S:
            mv=cur-prc[:,t-2]; pnl.append(float(cp@mv))
            fexp.append(float(cp[1:]@cur[1:] + cp[0]*cur[0]))         # net $ exposure to the factor
        if t<E:
            newPos,net=positions(t,mode)
            dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
    pnl=np.array(pnl); fexp=np.array(fexp[:-1]) if len(fexp)>len(pnl) else np.array(fexp)
    fexp=np.array([float((positions(t,mode)[0][1:]@prc[1:,t-1]) + positions(t,mode)[0][0]*prc[0,t-1]) for t in range(S,E)])
    o=np.sort(pnl)
    print(f"{mode:<9}{np.abs(fexp).mean():>13.0f}{fexp.std():>12.0f}{pnl.std():>9.0f}"
          f"{pnl[big[big<len(pnl)]].std():>15.0f}{o[:10].mean():>9.0f}")
print("\n|netfactor$| = avg absolute net exposure to the common index factor (bigger = lose more in a common drawdown)")
