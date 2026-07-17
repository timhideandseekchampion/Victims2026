"""Lift the score ceiling by raising RETURN-PER-DOLLAR of the reversion signal.

Levers tested (all market-neutral, scored by exact eval.py logic):
  A. binary pair sizing            (baseline)
  B. z-proportional sizing         (deploy ~ |z|, capped at limit)
  C. Kalman dynamic hedge + z-prop (cleaner spread)
  D. mean-variance optimal reversion portfolio (Sigma^-1 alpha on pair spreads)
Key metric: mean$ per day AND return-per-$gross (mean / avg gross exposure).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, section, RESULTS)

P, df, tickers = prices_array()
N, T = P.shape
IDX = {t: i for i, t in enumerate(tickers)}
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
GROSSCAP = dlrPosLimit.sum()
cd = pd.read_csv(f"{RESULTS}/coint_all_pairs.csv").sort_values("coint_p")
PAIRS = [(IDX[a], IDX[b]) for a, b in zip(cd[cd.coint_p < 0.02].a, cd[cd.coint_p < 0.02].b)]
LB = 90


def bt(get_pos, start=T-250, end=T):
    cash=0.0; cp=np.zeros(N); tv=0.0; val=0.0; cm=0.0; pll=[]; gr=[]
    state = {}
    for t in range(start, end+1):
        h=P[:, :t]; cur=h[:, -1]
        if t < end:
            lim=(dlrPosLimit/cur).astype(int); pos=np.clip(get_pos(h, state), -lim, lim).astype(int)
        else: pos=np.array(cp)
        d=pos-cp; cash-=cur.dot(d)+cm; dv=cur*np.abs(d); cm=np.sum(dv*commRate); tv+=dv.sum()
        cp=np.array(pos); pv=cp.dot(cur); gr.append(np.abs(cp*cur).sum())
        pll.append(cash+pv-val) if t>start else None; val=cash+pv
    pll=np.array(pll); mu,sd=pll.mean(),pll.std(); sr=np.sqrt(250)*mu/sd if sd>0 else 0
    sc=mu*(sr**2/(sr**2+1)) if mu>0 and sd>1e-10 else mu
    rpd = mu/ (np.mean(gr)/1e6) if np.mean(gr)>0 else 0   # $PnL per $M gross
    return sr, sc, mu, np.mean(gr)/1e3, rpd


def pairs_binary(h, st, entry=1.0, exit=0.3, dollars=10000):
    n,t=h.shape; pos=np.zeros(n)
    if t<LB+2: return pos
    for i,j in PAIRS:
        beta=np.polyfit(h[j,-LB:], h[i,-LB:],1)[0]; sp=h[i,:]-beta*h[j,:]
        w=sp[-LB:]; z=(sp[-1]-w.mean())/(w.std()+1e-9); k=st.get((i,j),0)
        if k==0 and abs(z)>entry: k=-int(np.sign(z))
        elif k!=0 and abs(z)<exit: k=0
        st[(i,j)]=k
        if k: pos[i]+=k*dollars/h[i,-1]; pos[j]+=-k*beta*dollars/h[j,-1]
    return pos

def pairs_zprop(h, st, zmax=2.0, dollars=12000):
    n,t=h.shape; pos=np.zeros(n)
    if t<LB+2: return pos
    for i,j in PAIRS:
        beta=np.polyfit(h[j,-LB:], h[i,-LB:],1)[0]; sp=h[i,:]-beta*h[j,:]
        w=sp[-LB:]; z=(sp[-1]-w.mean())/(w.std()+1e-9)
        size=-np.clip(z/zmax,-1,1)                     # proportional to z, capped
        pos[i]+=size*dollars/h[i,-1]; pos[j]+=-size*beta*dollars/h[j,-1]
    return pos

def pairs_kalman_zprop(h, st, zmax=2.0, dollars=12000, q=1e-4):
    n,t=h.shape; pos=np.zeros(n)
    if t<40: return pos
    for i,j in PAIRS:
        key=(i,j)
        beta,Pv,k0 = st.get(key,(1.0,1.0,1))
        x=np.log(h[j,:]); y=np.log(h[i,:])
        # incremental update from last processed index k0
        res=[]
        for k in range(max(1,k0), t):
            e=y[k]-beta*x[k]; Pv+=q; K=Pv*x[k]/(x[k]*x[k]*Pv+1.0); beta+=K*e; Pv*=(1-K*x[k]); res.append(e)
        # keep a rolling residual buffer in state
        buf = st.get((key,'buf'), [])
        buf = (buf+res)[-LB:]; st[(key,'buf')]=buf; st[key]=(beta,Pv,t)
        if len(buf)<10: continue
        arr=np.array(buf); z=(arr[-1]-arr.mean())/(arr.std()+1e-9)
        size=-np.clip(z/zmax,-1,1)
        # beta in log-space ~ ratio; approximate share hedge with price beta
        pb=np.polyfit(h[j,-LB:],h[i,-LB:],1)[0]
        pos[i]+=size*dollars/h[i,-1]; pos[j]+=-size*pb*dollars/h[j,-1]
    return pos

def mv_reversion(h, st, dollars_scale=2.5, lb=60, k=3):
    """mean-variance optimal reversion: alpha=-residual s-score, w=Sigma^-1 alpha."""
    n,t=h.shape
    if t<lb+2: return np.zeros(n)
    R=np.diff(np.log(h[:,-lb:]),axis=1).T           # (lb-1) x N
    Rc=R-R.mean(0)
    U,S,Vt=np.linalg.svd(Rc,full_matrices=False)
    comp=Vt[:k]
    resid = Rc - (Rc@comp.T)@comp                   # residual returns
    cumres = resid.cumsum(0)                        # residual "price"
    s = (cumres[-1]-cumres.mean(0))/(cumres.std(0)+1e-9)   # s-score per name
    alpha = -s
    Sig = np.cov(resid.T)+1e-4*np.eye(n)
    w = np.linalg.solve(Sig, alpha)
    w -= w.mean()                                   # market neutral
    w = w/ (np.abs(w).sum()+1e-12)
    return (w*dollars_scale*GROSSCAP/h[:,-1]).astype(float)


section("31. RETURN-PER-DOLLAR LEVERS (Sharpe / Score / mean$ / gross$k / PnL-per-$M)")
tests = [
 ("A binary pairs",        lambda h,st: pairs_binary(h,st,1.0,0.3,10000)),
 ("A binary e1.25",        lambda h,st: pairs_binary(h,st,1.25,0.3,10000)),
 ("B z-proportional",      lambda h,st: pairs_zprop(h,st,2.0,12000)),
 ("B z-prop zmax1.5",      lambda h,st: pairs_zprop(h,st,1.5,12000)),
 ("C Kalman z-prop",       lambda h,st: pairs_kalman_zprop(h,st,2.0,12000)),
 ("D MV reversion",        lambda h,st: mv_reversion(h,st,2.5,60,3)),
 ("D MV reversion k5",     lambda h,st: mv_reversion(h,st,2.5,60,5)),
]
print(f"{'lever':<22}{'Sharpe':>7}{'Score':>7}{'mean$':>7}{'gross$k':>9}{'PnL/$M':>8}")
for name,fn in tests:
    sr,sc,mu,gr,rpd=bt(fn)
    print(f"{name:<22}{sr:>7.2f}{sc:>7.0f}{mu:>7.0f}{gr:>9.0f}{rpd:>8.1f}")
