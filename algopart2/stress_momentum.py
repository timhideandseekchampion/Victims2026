"""stress_momentum.py — survival stress test across HOSTILE regimes injected after day 750.
The real question isn't 'momentum' specifically — it's: what regime change actually KILLS an
adaptive lead-lag+reversion book, and does any detector/breaker help?

Regimes (appended to the real 750-day panel, calibrated to real idio vol, index kept ~flat):
  momentum  winners keep winning (cross-sectional trend)        <- the user's fear
  flip      alternates momentum/reversion every `period` days   <- whipsaw the adaptation
  noise     no cross-sectional predictability at all            <- the edge simply dies

Books:
  lldollar  shipped                       rotate  IC-gated rotation (SAFE_rotate)
  breaker   lldollar + drawdown circuit-breaker (go flat on sustained loss, regime-agnostic)
  momentum  a pure momentum book (the 'right' book for a trend)
HYPOTHETICAL synthetic scenarios — not predictions of the real 750+ data.
"""
import numpy as np, pandas as pd
import SAFE_lldollar as LL
import SAFE_rotate as ROT

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def make_ext(kind, T_ext=150, mom=0.6, period=25, seed=1):
    rng = np.random.default_rng(seed)
    logp = np.log(prc).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy(); K = 5
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        if kind == "noise":
            drift = np.zeros(50)
        elif kind == "flip":
            sgn = 1.0 if (step // period) % 2 == 0 else -1.0     # momentum block then reversion block
            drift = sgn * mom * (tc / (tc.std() + 1e-9)) * vol
        else:                                                     # momentum
            drift = mom * (tc / (tc.std() + 1e-9)) * vol
        drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :nDays] = prc
    return full

def eval_book(full, getpos, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]; algo=[]; idio=[]
    for t in range(S, E+1):
        cur=full[:,t-1]
        if t>S:
            mv=cur-full[:,t-2]; algo.append(cp[0]*mv[0]); idio.append((cp[1:]*mv[1:]).sum())
        newPos = getpos(full[:,:t]) if t<E else cp
        lim=(dlr/cur).astype(int); newPos=np.clip(newPos,-lim,lim).astype(int)
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    return np.array(pll), np.array(algo), np.array(idio)

def momentum_pos(P):
    P=np.asarray(P,float); ni,t=P.shape; cur=P[:,-1]; pos=np.zeros(ni)
    if t<LL.WARMUP: return pos.astype(int)
    lp=np.log(P); trail=lp[1:,-1]-lp[1:,-1-LL.REV_W]; wz=trail-trail.mean()
    pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
    ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
    net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]
    pos[0]=float(np.sign(net)*cap) if abs(net)>=LL.ALGO_LL_DOLLAR else 0.0
    return np.clip(pos,-(dlr/cur).astype(int),(dlr/cur).astype(int)).astype(int)

_pos_hist={}
def make_breaker(Wbd=15, loss_thresh=-40_000.0):
    def breaker_pos(P):
        P=np.asarray(P,float); ni,t=P.shape
        base=LL.getMyPosition(P)
        if t>=LL.WARMUP+Wbd+1:
            pnl=sum(float(_pos_hist[s-1] @ (P[:,s-1]-P[:,s-2])) for s in range(t-Wbd,t) if s-1 in _pos_hist)
            if pnl < loss_thresh:
                _pos_hist[t]=np.zeros(ni); return np.zeros(ni,int)
        _pos_hist[t]=base.copy(); return base
    return breaker_pos

def worst_roll(pll, w=15):
    c=np.concatenate([[0],pll.cumsum()]); return min(c[i+w]-c[i] for i in range(len(pll)-w+1)) if len(pll)>=w else pll.sum()

S,E=nDays,nDays+150
books=[("lldollar",LL.getMyPosition),("rotate",ROT.getMyPosition),("breaker",make_breaker()),("momentum",momentum_pos)]
for kind in ("momentum","flip","noise"):
    full=make_ext(kind)
    # each injected regime is a DIFFERENT price series past day 750 -> rebuild SAFE_rotate's
    # forecast/IC/trend caches (they are keyed by column count, not by series).
    ROT._SIG.clear(); ROT._RET.clear(); ROT._ICD.clear(); ROT._AZ.clear(); ROT._XC.clear()
    r=np.diff(np.log(full[1:]),axis=1)[:,nDays:]
    ac=np.mean([np.corrcoef(r[:,tt-1],r[:,tt])[0,1] for tt in range(1,r.shape[1])])
    print(f"\n===== {kind.upper()}  (vol x{r.std()/np.diff(np.log(prc[1:]),axis=1).std():.1f}, lag-1 autocorr {ac:+.2f}) =====")
    print(f"{'book':<10}{'totalPnL':>11}{'maxDD':>10}{'worst15d':>10}{'IDIOleg':>10}   per-50d")
    for nm,fn in books:
        _pos_hist.clear(); tot,algo,idio=eval_book(full,fn,S,E)
        cum=tot.cumsum(); dd=(cum-np.maximum.accumulate(cum)).min()
        segs="  ".join(f"{cum[min(i+50,len(cum))-1]-(cum[i-1] if i>0 else 0):>7.0f}" for i in range(0,len(tot),50))
        print(f"{nm:<10}{cum[-1]:>11.0f}{dd:>10.0f}{worst_roll(tot):>10.0f}{idio.sum():>10.0f}   {segs}")
