import numpy as np, pandas as pd
from strat_engine import Engine, cfg
from statsmodels.tsa.stattools import coint

prc_all = pd.read_csv("../prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape
commRate=np.full(nInst,1e-4); commRate[0]=2e-5
dlrLim=np.full(nInst,10_000); dlrLim[0]=100_000

def ridge_fit(X,Y,hl=2000,alpha=0.1):
    n,p=X.shape; lam=0.5**(1.0/hl); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
    mx=(w[:,None]*X).sum(0)/sw; my=(w[:,None]*Y).sum(0)/sw; Xc,Yc=X-mx,Y-my
    XtWX=Xc.T@(w[:,None]*Xc); XtWY=Xc.T@(w[:,None]*Yc); eps=1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+alpha)*np.eye(p),XtWY),mx,my

def base_pos(prc,cache):
    ni,t=prc.shape; pos=np.zeros(ni); lp=np.log(prc); ret=lp[:,1:]-lp[:,:-1]
    if cache["fit_t"]!=t: cache["model"]=ridge_fit(ret[:,:-1].T,ret[1:,1:].T); cache["fit_t"]=t
    B,mx,my=cache["model"]; pred=my+(ret[:,-1]-mx)@B; w=pred-pred.mean()
    s=np.sign(w)*(10000/prc[1:,-1]); pos[1:]=np.where(np.abs(w)>=0.2*(np.std(w)+1e-12),s,0.0)
    cap=100000/prc[0,-1]; rev=0.0
    if t>92:
        lpA=np.log(prc[0]); mv=lpA[30:]-lpA[:-30]; z=(mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
        rev=float(np.clip(-np.clip(z,-3,3)*200000/prc[0,-1],-cap,cap))
    return pos,lp,ret,cap,rev

def finish(pos,ret,prc,cap,rev):
    rA=ret[0]; rAc=rA-rA.mean(); den=rAc@rAc+1e-12
    betas=((ret[1:]-ret[1:].mean(1,keepdims=True))@rAc)/den
    net=(pos[1:]*prc[1:,-1])@betas; room=max(cap-abs(rev),0.0)
    pos[0]=rev+float(np.clip(-net/prc[0,-1],-room,room)); return pos.astype(int)

def make(edge=None,dollars=0,eng=None):
    cache={"fit_t":None,"model":None}
    def gp(prc):
        if prc.shape[1]<60: return np.zeros(prc.shape[0],dtype=int)
        pos,lp,ret,cap,rev=base_pos(prc,cache)
        if edge=="xs": pos+=eng._xs(prc,prc[:,-1])
        elif edge=="lead": pos+=eng._lead(lp,prc[:,-1])
        elif edge=="mf": pos+=eng._multifactor(lp,prc[:,-1])
        elif edge=="pairs": pos+=eng._pairs(prc,lp,prc[:,-1])
        return finish(pos,ret,prc,cap,rev)
    return gp

def score(mu,sd):
    if mu<=0 or sd<1e-10: return mu
    sr=np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def run(gp,start,end):
    cash=0;cp=np.zeros(nInst);val=0;cm=0;pll=[]
    for t in range(start,end+1):
        p=prc_all[:,:t];c=p[:,-1]
        npos=np.clip(gp(p),-(dlrLim/c).astype(int),(dlrLim/c).astype(int)).astype(int) if t<end else cp.copy()
        d=npos-cp;cash-=c.dot(d)+cm;dv=c*np.abs(d);cm=(dv*commRate).sum();cp=npos.copy()
        pl=cash+cp.dot(c)-val;val=cash+cp.dot(c)
        if t>start: pll.append(pl)
    pll=np.array(pll);return score(pll.mean(),pll.std())

print("=== v2 causal edges layered on ridge ===")
print(f"{'ridge alone':28} S@250 {run(make(),nt-250,nt):7.1f}  H2 {run(make(),280,500):7.1f}")
for e,d in [("xs",9000),("lead",5000),("mf",3000)]:
    eng=Engine(cfg(**{f"w_{e}":1.0,f"{e}_dollars":d}))
    print(f"{'+ '+e:28} S@250 {run(make(e,d,eng),nt-250,nt):7.1f}  H2 {run(make(e,d,eng),280,500):7.1f}")

print("\n=== clean OOS: pairs picked on days 0-250, traded 251-500 ===")
win=np.log(prc_all[:,:250]); C=np.corrcoef(np.diff(win,axis=1)); oos=[]
for i in range(nInst):
    for j in range(i+1,nInst):
        if abs(C[i,j])>0.4:
            try:
                if coint(win[i],win[j])[1]<0.02: oos.append([i,j])
            except: pass
eng=Engine(cfg(pair_lb=90,pair_entry=1.0,pair_exit=0.3,pair_dollars=10000,w_pairs=1.0,fixed_pairs=oos[:24]))
a=run(make(),280,500); b=run(make("pairs",0,eng),280,500)
print(f"{len(oos)} pairs picked on 0-250 | ridge alone(251-500) {a:.1f}  +OOS pairs {b:.1f}  delta {b-a:+.1f}")
print("DONE")
