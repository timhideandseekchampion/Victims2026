"""consistency.py — the user's real question: our score 'goes wild' (1700 -> 905) while low-std
teams hold rank. Quantify SCORE CONSISTENCY, not just daily std.

For each ALGO-leg variant (idio 50-name book identical in all), across rolling windows:
  - std of the windowed SCORE  = how much the leaderboard result swings with the data draw
  - floor (worst window)       = downside on a bad draw
  - standalone Sharpe of each leg on 500-750 (is the ALGO bet pulling its weight per unit risk?)

Correct leg PnL: position held INTO day t (established at t-1) earns move P_t - P_{t-1}.
"""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

HALF_LIVES=(250,500,1000,2000); RIDGE_A=0.1; BLEND=0.3; REV_W=10
CONTRA_DOL=1_000_000; CONTRA_K=30; CONTRA_WZ=60; WARMUP=96; ALGO_LL_DOLLAR=50_000

def _ewls_ridge(X,Y,hl,a):
    n,p=X.shape; lam=0.5**(1.0/hl); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
    mx=(w[:,None]*X).sum(0)/sw; my=(w[:,None]*Y).sum(0)/sw; Xc,Yc=X-mx,Y-my
    XtWX=Xc.T@(w[:,None]*Xc); XtWY=Xc.T@(w[:,None]*Yc); eps=1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p),XtWY),mx,my

_cache={}
def getpos(t, mode):
    """position for day t (uses prc[:, :t]); caches the expensive idio/forecast part per t."""
    if t not in _cache:
        prcSoFar=prc[:,:t]; cur=prcSoFar[:,-1]
        lp=np.log(prcSoFar); r=lp[:,1:]-lp[:,:-1]; fs=[]
        for hl in HALF_LIVES:
            B,mx,my=_ewls_ridge(r[:,:-1].T,r[1:,1:].T,hl,RIDGE_A)
            pred=my+(r[:,-1]-mx)@B; fi=pred-pred.mean(); fs.append(fi/(fi.std()+1e-12))
        wz=np.mean(fs,0)
        rr=lp[1:,-1]-lp[1:,-1-REV_W]; rr=rr-rr.mean(); rv=-rr/(rr.std()+1e-12)
        wz=(1-BLEND)*wz+BLEND*rv
        idio_lim=(dlr[1:]/cur[1:]).astype(int)
        idio_int=np.clip(np.sign(wz)*(dlr[1:]/cur[1:]),-idio_lim,idio_lim).astype(int)
        net_dol=float((idio_int*cur[1:]).sum())
        lpA=lp[0]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
        z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
        fade=-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0])
        _cache[t]=(idio_int,net_dol,fade,cur)
    idio_int,net_dol,fade,cur=_cache[t]
    pos=np.zeros(nInst); pos[1:]=idio_int; cap=dlr[0]/cur[0]
    if mode=="off": av=0.0
    elif mode=="fade": av=float(np.clip(fade,-cap,cap))
    elif mode=="prop":
        av=float(np.clip(net_dol/cur[0],-cap,cap)) if abs(net_dol)>=ALGO_LL_DOLLAR else float(np.clip(fade,-cap,cap))
    else:  # binary (shipped SAFE_lldollar)
        av=float(np.sign(net_dol)*cap) if abs(net_dol)>=ALGO_LL_DOLLAR else float(np.clip(fade,-cap,cap))
    pos[0]=av
    lim=(dlr/cur).astype(int); return np.clip(pos,-lim,lim).astype(int)

def run(mode,S,E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; tot=[]; algo=[]; idio=[]
    for t in range(S,E+1):
        cur=prc[:,t-1]
        if t>S:
            mv=cur-prc[:,t-2]; algo.append(cp[0]*mv[0]); idio.append((cp[1:]*mv[1:]).sum())
        newPos=getpos(t,mode) if t<E else cp
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: tot.append(pl)
    return np.array(tot),np.array(algo),np.array(idio)

def sc(mu,sd):
    if mu<=0 or sd<1e-10: return mu
    sr=np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def shrp(a): return np.sqrt(250)*a.mean()/a.std()

print("precomputing forecasts ..."); [getpos(t,"off") for t in range(WARMUP+1,nDays+1)]

# ---- standalone leg Sharpe on 500-750 (is the ALGO bet worth its variance?) ----
print("\n=== 500-750: does each leg earn its risk? (standalone Sharpe) ===")
print(f"{'mode':<8}{'TOTmu':>8}{'TOTstd':>8}{'TOTsr':>7}{'score':>7} | {'ALGOsr':>7}{'IDIOsr':>7}")
for mode in ["binary","prop","fade","off"]:
    tot,algo,idio=run(mode,500,750)
    a_sr=shrp(algo) if algo.std()>0 else 0.0
    star=" <-- shipped" if mode=="binary" else ""
    print(f"{mode:<8}{tot.mean():>8.0f}{tot.std():>8.0f}{shrp(tot):>7.2f}{sc(tot.mean(),tot.std()):>7.0f} | "
          f"{a_sr:>7.2f}{shrp(idio):>7.2f}{star}")

# ---- SCORE CONSISTENCY across rolling windows (the 'go wild' metric) ----
for L in (100, 250):
    ends=list(range(WARMUP+1+L, nDays+1, 10))
    print(f"\n=== {L}-day rolling windows (n={len(ends)}): score distribution ===")
    print(f"{'mode':<8}{'mean':>8}{'std':>8}{'floor':>8}{'max':>8}{'CV%':>7}   (CV=std/mean, lower=steadier)")
    for mode in ["binary","prop","fade","off"]:
        s=np.array([sc(*(lambda x:(x.mean(),x.std()))(run(mode,e-L,e)[0])) for e in ends])
        star=" <-- shipped" if mode=="binary" else ""
        print(f"{mode:<8}{s.mean():>8.0f}{s.std():>8.0f}{s.min():>8.0f}{s.max():>8.0f}{100*s.std()/s.mean():>7.1f}{star}")
