"""Instrument the ALGO (index) leg: decompose the daily ALGO position into its
beta-HEDGE part and its contrarian-REVERSION part, and attribute PnL to each.
Also record the 50-asset book PnL for comparison. -> algo_data.json"""
import numpy as np, json, pandas as pd, warnings; warnings.filterwarnings("ignore")
import backtest_full as bt, adaptive_estimator as ae
prc=bt.prcAll; nInst=prc.shape[0]; dlrPosLimit=bt.dlrPosLimit; commRate=bt.commRate
CFG={"scheme":"ewls","half_life":250,"alpha":0.1,"conv_z":0.2}; K=30; Wz=60; CD=200000; csd=lambda v:v-v.mean()
NTD=250; startDay=500-NTD

cache={"m":None,"last":-10}; curPos=np.zeros(nInst)
rec=[]   # per test day
for t in range(startDay,500):
    curP=prc[:,:t][:,-1]
    lp=np.log(prc[:,:t]); ret=lp[:,1:]-lp[:,:-1]
    if cache["m"] is None or t-cache["last"]>=1: cache["m"]=ae.fit_rows(ret[:,:-1].T,ret[1:,1:].T,CFG); cache["last"]=t
    B,mx,my=cache["m"]; w=csd(my+(ret[:,-1]-mx)@B)
    keep=np.abs(w)>=CFG["conv_z"]*(np.std(w)+1e-12)
    pos50=np.where(keep,np.sign(w)*(10000/curP[1:]),0.0)
    rA=ret[0];rAc=rA-rA.mean();den=rAc@rAc+1e-12; b=((ret[1:]-ret[1:].mean(1,keepdims=True))@rAc)/den
    hedge_sh = -((pos50*curP[1:])@b)/curP[0]                    # beta-hedge shares
    lpA=lp[0]; mv=lpA[K:]-lpA[:-K]; z=(mv[-1]-mv[-Wz:].mean())/(mv[-Wz:].std()+1e-12); z=float(np.clip(z,-3,3))
    lim=int(dlrPosLimit[0]/curP[0])                            # ALGO cap in shares
    contra_c = float(np.clip(-z*CD/curP[0], -lim, lim))        # reversion FIRST claim on the cap
    room = max(lim-abs(contra_c), 0.0)
    hedge_c = float(np.clip(hedge_sh, -room, room))            # hedge fills only leftover room (applied LAST)
    net = contra_c + hedge_c
    # next-day ALGO price change for PnL
    dP = prc[0,t]-prc[0,t-1] if t<500 else 0.0
    rec.append(dict(day=t-startDay, price=float(prc[0,t-1]), z=z, move=float(mv[-1]*100),
        hedge=float(hedge_c*curP[0]), contra=float(contra_c*curP[0]), net=float(net*curP[0]),
        hedge_pnl=float(hedge_c*dP), contra_pnl=float(contra_c*dP), algo_pnl=float(net*dP),
        book_pnl=float((curPos[1:]*(curP[1:]-prc[1:,t-2])).sum()) if t>startDay else 0.0))
    curPos=np.zeros(nInst); curPos[1:]=pos50.astype(int); curPos[0]=net

R=rec
def cum(k): return list(np.cumsum([r[k] for r in R]).round(0))
def sh(k): a=np.array([r[k] for r in R]); return round(float(np.sqrt(250)*a.mean()/a.std()),2) if a.std()>0 else 0.0
hedge_pnl=np.array([r["hedge_pnl"] for r in R]); contra_pnl=np.array([r["contra_pnl"] for r in R])
algo_pnl=np.array([r["algo_pnl"] for r in R]); book_pnl=np.array([r["book_pnl"] for r in R])
net=np.array([r["net"] for r in R])
out={"meta":{
    "days":len(R),
    "algo_total":round(float(algo_pnl.sum()),0),"hedge_total":round(float(hedge_pnl.sum()),0),"contra_total":round(float(contra_pnl.sum()),0),
    "book_total":round(float(book_pnl.sum()),0),
    "algo_sharpe":sh("algo_pnl"),"hedge_sharpe":sh("hedge_pnl"),"contra_sharpe":sh("contra_pnl"),
    "pct_long":round(float(100*(net>0).mean()),0),"pct_short":round(float(100*(net<0).mean()),0),
    "corr_algo_book":round(float(np.corrcoef(algo_pnl,book_pnl)[0,1]),2),
    "avg_hedge":round(float(np.mean([abs(r["hedge"]) for r in R])),0),"avg_contra":round(float(np.mean([abs(r["contra"]) for r in R])),0),
    "contra_hit":round(float(100*np.mean(np.sign(contra_pnl[contra_pnl!=0])>0)) if (contra_pnl!=0).any() else 0,0)},
  "price":[round(r["price"],2) for r in R],"z":[round(r["z"],2) for r in R],"move":[round(r["move"],1) for r in R],
  "hedge":[round(r["hedge"],0) for r in R],"contra":[round(r["contra"],0) for r in R],"net":[round(r["net"],0) for r in R],
  "cum_hedge":[float(x) for x in cum("hedge_pnl")],"cum_contra":[float(x) for x in cum("contra_pnl")],
  "cum_algo":[float(x) for x in cum("algo_pnl")],"cum_book":[float(x) for x in cum("book_pnl")]}
open("algo_data.json","w").write(json.dumps(out))
m=out["meta"]
print(f"ALGO leg over {m['days']}d:  total PnL ${m['algo_total']:.0f}  (hedge ${m['hedge_total']:.0f} + reversion ${m['contra_total']:.0f})")
print(f"  Sharpe:  ALGO leg {m['algo_sharpe']}  |  hedge {m['hedge_sharpe']}  |  reversion {m['contra_sharpe']}")
print(f"  ALGO position: {m['pct_long']}% long / {m['pct_short']}% short  | avg hedge ${m['avg_hedge']:.0f}, avg reversion ${m['avg_contra']:.0f}")
print(f"  corr(ALGO leg PnL, 50-book PnL) = {m['corr_algo_book']}   | 50-book total ${m['book_total']:.0f}")
