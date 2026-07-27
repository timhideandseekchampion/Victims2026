"""var_decomp.py — where does the stdev come from?

Runs the EXACT eval engine but splits each day's PnL into
  (a) ALGO leg  (instrument 0, the $100k index bet)
  (b) IDIO book (the 50 market-neutral stock lines)
so we can see which leg drives the total stdev the user is worried about (2046 live).

Compares ALGO-leg variants (idio leg identical in all):
  binary   = current SAFE_lldollar: |net$|>=50k -> FULL +/-$100k (the shipped book)
  prop     = size ALGO proportional to net$ (softer, the '+0.15 Sharpe / -8 score' option)
  fade     = base SAFE reversion fade (CONTRA_DOL=1M), no lead-lag gate  == SAFE.py
  off      = no ALGO position at all (pure idio book)
"""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)

HALF_LIVES=(250,500,1000,2000); RIDGE_A=0.1; BLEND=0.3; REV_W=10
CONTRA_DOL=1_000_000; CONTRA_K=30; CONTRA_WZ=60; WARMUP=96
ALGO_LL_DOLLAR=50_000

def _ewls_ridge(X,Y,hl,a):
    n,p=X.shape; lam=0.5**(1.0/hl); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
    mx=(w[:,None]*X).sum(0)/sw; my=(w[:,None]*Y).sum(0)/sw; Xc,Yc=X-mx,Y-my
    XtWX=Xc.T@(w[:,None]*Xc); XtWY=Xc.T@(w[:,None]*Yc)
    eps=1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p),XtWY),mx,my

def getpos(prcSoFar, mode):
    prcSoFar=np.asarray(prcSoFar,float); nInst,t=prcSoFar.shape
    cur=prcSoFar[:,-1]; pos=np.zeros(nInst)
    if t<WARMUP: return pos.astype(int)
    lp=np.log(prcSoFar); r=lp[:,1:]-lp[:,:-1]
    fs=[]
    for hl in HALF_LIVES:
        B,mx,my=_ewls_ridge(r[:,:-1].T,r[1:,1:].T,hl,RIDGE_A)
        pred=my+(r[:,-1]-mx)@B; fi=pred-pred.mean(); fs.append(fi/(fi.std()+1e-12))
    wz=np.mean(fs,0)
    if BLEND>0:
        rr=lp[1:,-1]-lp[1:,-1-REV_W]; rr=rr-rr.mean(); rv=-rr/(rr.std()+1e-12)
        wz=(1-BLEND)*wz+BLEND*rv
    pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
    idio_lim=(dlr[1:]/cur[1:]).astype(int)
    idio_int=np.clip(pos[1:],-idio_lim,idio_lim).astype(int)
    net_dol=float((idio_int*cur[1:]).sum())
    cap=dlr[0]/cur[0]
    if mode=="off":
        av=0.0
    elif mode=="fade":
        lpA=lp[0]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
        z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
        av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    elif mode=="prop":
        # size ALGO proportional to the book skew, capped at $100k
        av=float(np.clip(net_dol/cur[0],-cap,cap)) if abs(net_dol)>=ALGO_LL_DOLLAR else 0.0
        if abs(net_dol)<ALGO_LL_DOLLAR:
            lpA=lp[0]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
            z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
            av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    else:  # binary (shipped)
        if ALGO_LL_DOLLAR>0 and abs(net_dol)>=ALGO_LL_DOLLAR:
            av=float(np.sign(net_dol)*cap)
        else:
            lpA=lp[0]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
            z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
            av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    pos[0]=av
    lim=(dlr/cur).astype(int)
    return np.clip(pos,-lim,lim).astype(int)

def run(mode, startDay, endDay):
    """eval engine; returns per-day total/algo/idio PnL arrays over (startDay, endDay]."""
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0
    tot=[]; algo=[]; idio=[]
    prevPrices=None
    for t in range(startDay, endDay+1):
        cur=prc[:,t-1]
        newPos=getpos(prc[:,:t],mode) if t<endDay else cp
        dP=newPos-cp
        cash-=cur.dot(dP)+comm
        dvol=cur*np.abs(dP); comm=np.sum(dvol*commRate)
        # per-leg mark-to-market PnL from yesterday's holdings over today's price move
        if prevPrices is not None and t>startDay:
            dpx=cur-prevPrices
            algo.append(cp_prev[0]*dpx[0])
            idio.append((cp_prev[1:]*dpx[1:]).sum())
        cp_prev=cp.copy(); prevPrices=cur.copy()
        cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>startDay: tot.append(pl)
    return np.array(tot), np.array(algo), np.array(idio)

def score(mu,sd):
    if mu<=0 or sd<1e-10: return mu
    sr=np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)

for (S,E,lbl) in [(500,750,"500-750 (250d, the graded leg we can see)"),
                  (650,750,"650-750 (100d, most recent)")]:
    print(f"\n================  {lbl}  ================")
    print(f"{'mode':<8}{'TOTmean':>9}{'TOTstd':>9}{'Shrp':>6}{'score':>7} | "
          f"{'ALGOmean':>9}{'ALGOstd':>9} | {'IDIOmean':>9}{'IDIOstd':>9}")
    for mode in ["binary","prop","fade","off"]:
        tot,algo,idio=run(mode,S,E)
        mu,sd=tot.mean(),tot.std(); sh=np.sqrt(250)*mu/sd
        am,asd=algo.mean(),algo.std(); im,isd=idio.mean(),idio.std()
        star=" <-- shipped" if mode=="binary" else ""
        print(f"{mode:<8}{mu:>9.1f}{sd:>9.1f}{sh:>6.2f}{score(mu,sd):>7.0f} | "
              f"{am:>9.1f}{asd:>9.1f} | {im:>9.1f}{isd:>9.1f}{star}")
