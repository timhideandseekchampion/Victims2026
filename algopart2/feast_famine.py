"""feast_famine.py — WHERE do the huge up-days and huge down-days come from?
Split SAFE_lldollar's daily PnL into ALGO leg (the $100k net-$ gate bet) vs IDIO book (50 names),
tag each day by whether the gate was ON (|net$|>=50k -> full $100k directional), and look at:
  - gate-ON vs gate-OFF day distributions
  - the 10 best and 10 worst days: which leg drove them, was the gate on
  - corr(ALGO pnl, IDIO pnl): does the gate AMPLIFY the book's directional bet (concentration)
    or diversify it? If positively correlated on gate-on days, the gate is a leverage on conviction
    -> feast AND famine are the same bet.
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

def positions(t):
    cur=prc[:,t-1]; pos=np.zeros(nInst); wz=wz_at(t); pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
    ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
    net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]; gate=abs(net)>=ALGO_LL
    if gate: av=float(np.sign(net)*cap)
    else:
        lpA=logp[0,:t]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
        z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
        av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    pos[0]=av; lim=(dlr/cur).astype(int); return np.clip(pos,-lim,lim).astype(int), gate

S,E=500,750
rows=[]  # (day, total, algo, idio, gate_on)
cp,gate_prev=np.zeros(nInst),False
for t in range(S,E+1):
    cur=prc[:,t-1]
    if t>S:
        mv=cur-prc[:,t-2]; a=cp[0]*mv[0]; i=(cp[1:]*mv[1:]).sum(); rows.append((t,a+i,a,i,gate_prev))
    if t<E:
        cp,gate_prev=positions(t)
R=np.array([(r[1],r[2],r[3]) for r in rows]); gate=np.array([r[4] for r in rows])
tot,alg,idi=R[:,0],R[:,1],R[:,2]

def st(x): return f"mean={x.mean():7.0f} std={x.std():7.0f} min={x.min():8.0f} max={x.max():8.0f}"
print(f"days scored: {len(tot)}   gate ON: {gate.sum()} ({100*gate.mean():.0f}%)\n")
print(f"TOTAL   {st(tot)}")
print(f"ALGO    {st(alg)}   skew={pd.Series(alg).skew():.2f}")
print(f"IDIO    {st(idi)}   skew={pd.Series(idi).skew():.2f}")
print(f"\ncorr(ALGO, IDIO) all days: {np.corrcoef(alg,idi)[0,1]:+.2f}   "
      f"on gate-ON days: {np.corrcoef(alg[gate],idi[gate])[0,1]:+.2f}")
print(f"\ngate ON  days: total {st(tot[gate])}")
print(f"gate OFF days: total {st(tot[~gate])}")

order=np.argsort(tot)
print(f"\n10 WORST days (total | ALGO | IDIO | gateON):")
for k in order[:10]:
    print(f"   {tot[k]:8.0f} | {alg[k]:8.0f} | {idi[k]:8.0f} | {'ON' if gate[k] else 'off'}")
print(f"10 BEST days:")
for k in order[::-1][:10]:
    print(f"   {tot[k]:8.0f} | {alg[k]:8.0f} | {idi[k]:8.0f} | {'ON' if gate[k] else 'off'}")

# how much of total variance does the ALGO leg add on gate-on days?
print(f"\nALGO leg share of the 10 worst days' losses: "
      f"{100*alg[order[:10]].sum()/tot[order[:10]].sum():.0f}%")
print(f"ALGO leg share of the 10 best days' gains:  "
      f"{100*alg[order[::-1][:10]].sum()/tot[order[::-1][:10]].sum():.0f}%")
