#!/usr/bin/env python
"""Strategy X-ray data: precompute per-day forecasts (w), ALGO betas, and the
price matrix over the eval window so the dashboard can (a) attribute PnL,
(b) score signal quality, (c) chart risk/exposure, and (d) run a FULLY
client-side interactive backtester (sliders recombine these cheaply; the
expensive EWLS fit is baked in here).

Also replicates the JS backtest loop in Python and asserts it reproduces the
shipped Score (~726) — so the dashboard's numbers are trustworthy.
"""
import json, numpy as np, warnings; warnings.filterwarnings("ignore")
import backtest_full as bt, adaptive_estimator as ae
prc = bt.prcAll; nInst = 51
names = open("prices.txt").readline().split()
CFG = {"scheme":"ewls","half_life":2000,"alpha":0.1}
WINDOW = 250; nt = prc.shape[1]; start = nt - WINDOW   # decisions for t in [start, nt)

# per-decision-day forecasts + betas
W=[]; BETAS=[]; cache={"m":None,"last":-10}
for t in range(start, nt):
    ret = np.log(prc[:, :t]); ret = ret[:,1:]-ret[:,:-1]        # (51, t-1)
    if cache["m"] is None or t-cache["last"]>=1:
        cache["m"]=ae.fit_rows(ret[:,:-1].T, ret[1:,1:].T, CFG); cache["last"]=t
    B,mx,my=cache["m"]; pred=my+(ret[:,-1]-mx)@B; w=pred-pred.mean()
    rA=ret[0]; rAc=rA-rA.mean(); den=rAc@rAc+1e-12
    betas=((ret[1:]-ret[1:].mean(1,keepdims=True))@rAc)/den
    W.append(w); BETAS.append(betas)
W=np.array(W); BETAS=np.array(BETAS)                            # (250, 50)
PRICES=prc[:, start-1:nt]                                       # (51, 251) indices start-1..nt-1

# ---- Python replica of the client-side backtest (must match backtest_full) ----
def backtest(conv_z=0.2, contra=200_000, K=30, WZ=60, hedge="beta", limit=10_000, algo_cap=100_000):
    cash=0; curPos=np.zeros(nInst); comm=0; value=0; pll=[]
    lpA_full=np.log(prc[0])                                     # ALGO log price (full, for reversion lookback)
    for t in range(start, nt+1):
        cur = PRICES[:, t-1-(start-1)]                          # = prc[:,t-1]
        if t<nt:
            i=t-start; w=W[i]; betas=BETAS[i]
            pos=np.zeros(nInst)
            keep=np.abs(w)>=conv_z*(np.std(w)+1e-12)
            pos[1:]=np.where(keep, np.sign(w)*(limit/cur[1:]), 0.0)
            cap0=algo_cap/cur[0]; rev=0.0
            if contra>0 and t>K+WZ+2:
                mv=lpA_full[:t]; mv=mv[K:]-mv[:-K]
                z=(mv[-1]-mv[-WZ:].mean())/(mv[-WZ:].std()+1e-12)
                rev=-float(np.clip(z,-3,3))*contra/cur[0]
            rev=float(np.clip(rev,-cap0,cap0))
            h=0.0
            if hedge=="beta": h=-((pos[1:]*cur[1:])@betas)/cur[0]
            elif hedge=="dollar": h=-((pos[1:]*cur[1:]).sum())/cur[0]
            room=max(cap0-abs(rev),0.0); pos[0]=rev+float(np.clip(h,-room,room))
            lim=(bt.dlrPosLimit/cur).astype(int); newPos=np.clip(pos,-lim,lim).astype(int)
        else: newPos=np.array(curPos)
        d=newPos-curPos; cash-=cur.dot(d)+comm; dv=cur*np.abs(d); comm=np.sum(dv*bt.commRate)
        curPos=np.array(newPos); pv=curPos.dot(cur); today=cash+pv-value; value=cash+pv
        if t>start: pll.append(today)
    a=np.array(pll); mu,sd=a.mean(),a.std()
    return bt.score(mu,sd), np.sqrt(250)*mu/sd, sd, a
sc,sh,sd,pll=backtest()
ref_m,ref_sd,ref_sh,_=bt.calcPL(prc,250,strat=__import__("Arbitrage_Victims_v4"))
print(f"replica Score={sc:.2f} Sharpe={sh:.2f}  |  shipped eval Score={bt.score(ref_m,ref_sd):.2f} Sharpe={ref_sh:.2f}")
assert abs(sc-bt.score(ref_m,ref_sd))<1.0, "replica does not match shipped!"

# ---- signal quality (param-independent): daily IC + hit-rate by conviction bucket ----
ic=[]; realized=[]
for i,t in enumerate(range(start,nt)):
    r = np.log(prc[1:,t]) - np.log(prc[1:,t-1])          # realized next-day return of the 50 assets
    realized.append(r)
    w = W[i][ : ] if W[i].shape[0]==50 else W[i]          # w is length 50 (demeaned forecast on tradeables? -> W has 50)
    # W rows are length 50 (forecast for the 50 tradeable assets)
    c = np.corrcoef(w, r)[0,1] if np.std(w)>0 and np.std(r)>0 else 0.0
    ic.append(round(float(c),4))
realized=np.array(realized)                               # (250,50)
# conviction buckets: |w|/std(w) per asset-day -> hit = sign(w)==sign(realized)
zabs=[]; hit=[]
for i in range(len(W)):
    s=np.std(W[i])+1e-12
    zabs.append(np.abs(W[i])/s); hit.append((np.sign(W[i])==np.sign(realized[i])).astype(float))
zabs=np.concatenate(zabs); hit=np.concatenate(hit)
edges=[0,0.2,0.5,1.0,1.5,10]; buckets=[]
for a,b in zip(edges[:-1],edges[1:]):
    m=(zabs>=a)&(zabs<b)
    buckets.append({"lo":a,"hi":(b if b<10 else None),"n":int(m.sum()),"hit":round(float(hit[m].mean()*100),1) if m.sum() else 0})

out={
  "names":names, "start":start, "nt":nt, "window":WINDOW,
  "ic":ic, "meanIC":round(float(np.mean(ic)),4), "icTstat":round(float(np.mean(ic)/(np.std(ic)/np.sqrt(len(ic)))),2),
  "convBuckets":buckets, "convZ_line":0.2,
  "prices":[[round(float(v),3) for v in PRICES[k]] for k in range(nInst)],
  "w":[[round(float(v),6) for v in row] for row in W],
  "betas":[[round(float(v),4) for v in row] for row in BETAS],
  "algoLogPrice":[round(float(v),6) for v in np.log(prc[0])],   # full 500d, for reversion lookback at any K/WZ
  "commRate":[float(x) for x in bt.commRate], "dlrLimit":[int(x) for x in bt.dlrPosLimit],
  "shipped":{"conv_z":0.2,"contra":200000,"K":30,"WZ":60,"hedge":"beta","score":round(sc,1),"sharpe":round(sh,2)},
}
json.dump(out, open("xray_data.json","w"), separators=(",",":"))
import os; print(f"wrote xray_data.json ({os.path.getsize('xray_data.json')//1024} KB)")
PY = None