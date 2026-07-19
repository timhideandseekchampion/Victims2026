"""turnover_band.py — no-trade band on the ACTUAL shipped book (SAFE_lldollar: ensemble lead-lag
+ reversion, net-$ ALGO gate). Hysteresis: keep a name's position unless |signal| clears the band,
so near-zero coin-flip churn is held instead of paid for. Reports score, turnover, commission
(split idio vs ALGO), on the graded leg + rolling windows. band=0 must reproduce 694."""
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

def run(band, S, E, algo_buf=0.0):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]; dv_id=0.0; dv_al=0.0; cm_id=0.0; cm_al=0.0
    for t in range(S, E+1):
        cur=prc[:,t-1]; pos=np.zeros(nInst)
        if t<E and t>=96:
            wz=wz_at(t); desired=np.sign(wz)*(dlr[1:]/cur[1:])
            if band>0:
                prev=cp[1:]; flip=np.sign(wz)!=np.sign(prev)
                hold=flip & (np.abs(wz)<band) & (prev!=0); desired=np.where(hold,prev,desired)
            pos[1:]=desired
            ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
            net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]
            # ALGO net-$ gate, with optional hysteresis buffer (stay in until net$ falls below LL-buf)
            prev0=cp[0]; trig = ALGO_LL - (algo_buf if (prev0!=0) else 0.0)
            if abs(net)>=trig: av=float(np.sign(net)*cap)
            else:
                lpA=logp[0,:t]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
                z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
                av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
            pos[0]=av; lim=(dlr/cur).astype(int); pos=np.clip(pos,-lim,lim).astype(int)
        else: pos=cp.copy()
        dp=pos-cp; cash-=cur.dot(dp)+comm
        dvv=cur*np.abs(dp); dv_id+=dvv[1:].sum(); dv_al+=dvv[0]
        cm_id+=dvv[1:].sum()*1e-4; cm_al+=dvv[0]*2e-5
        comm=np.sum(dvv*commRate); cp=pos.copy()
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    pll=np.array(pll); mu,sd=pll.mean(),pll.std()
    sc=mu*(np.sqrt(250)*mu/sd)**2/((np.sqrt(250)*mu/sd)**2+1) if mu>0 else mu
    gross=pll.sum()+cm_id+cm_al
    return dict(mu=mu,sd=sd,sr=np.sqrt(250)*mu/sd,sc=sc,dv_id=dv_id,dv_al=dv_al,cm_id=cm_id,cm_al=cm_al,
                comm_pct=100*(cm_id+cm_al)/gross if gross>0 else 0)

print("GRADED LEG 500-750 — idio no-trade band on SAFE_lldollar:")
print(f"  {'band':<6}{'SCORE':>7}{'Sharpe':>7}{'idio_comm':>11}{'algo_comm':>11}{'comm%PnL':>9}{'idio_turnover':>14}")
for b in (0.0,0.05,0.10,0.15,0.20,0.30,0.50):
    m=run(b,500,750); tag="  <-- current" if b==0 else ""
    print(f"  {b:<6}{m['sc']:>7.0f}{m['sr']:>7.2f}{m['cm_id']:>11.0f}{m['cm_al']:>11.0f}{m['comm_pct']:>9.1f}{m['dv_id']:>14.0f}{tag}")

print("\nROLLING 250d windows (mean SCORE / floor / #wins vs band=0):")
ends=list(range(400,nDays+1,10))
base={e:run(0.0,e-250,e)['sc'] for e in ends if e-250>=96}
print(f"  {'band':<6}{'meanSCORE':>11}{'floor':>8}{'wins':>8}")
for b in (0.0,0.05,0.10,0.15,0.20,0.30):
    ss={e:run(b,e-250,e)['sc'] for e in ends if e-250>=96}; arr=np.array(list(ss.values()))
    wins='--' if b==0 else f"{sum(1 for e in ss if ss[e]>base[e])}/{len(ss)}"
    print(f"  {b:<6}{arr.mean():>11.0f}{arr.min():>8.0f}{wins:>8}")

print("\n+ ALGO-gate hysteresis buffer (band=0.10 idio, vary algo_buf $):")
for buf in (0,5000,10000,15000):
    m=run(0.10,500,750,algo_buf=buf)
    print(f"  algo_buf={buf:>6}: SCORE={m['sc']:.0f}  algo_comm={m['cm_al']:.0f}  algo_turnover={m['dv_al']:.0f}")
