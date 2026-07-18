"""why_balanced.py — decompose sign vs balanced sizing into mu / std, and test whether the
sign-imbalance is INFORMATIVE (correlated with next-day market move) or just a noise leak,
separately for b0.20 and b0.30. Score = mu * SR^2/(SR^2+1), so watch mu vs std."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = [250, 500, 1000, 2000]
_rc = {}
def ridge_z(t, hl, a=0.1):
    key = (t, hl)
    if key in _rc: return _rc[key]
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5**(1/hl); w = lam**np.arange(n-1,-1,-1); sw = w.sum()
    mx = (w[:,None]*X).sum(0)/sw; my = (w[:,None]*Y).sum(0)/sw
    Xc = X-mx; Yc = Y-my
    B = np.linalg.solve(Xc.T@(w[:,None]*Xc)+a*np.eye(nInst), Xc.T@(w[:,None]*Yc))
    f = my+(xin-mx)@B; f = f-f.mean(); v = f/(f.std()+1e-12); _rc[key]=v; return v
def revz(t, w):
    rr = logp[1:, t-1]-logp[1:, t-1-w]; rr = rr-rr.mean(); return -rr/(rr.std()+1e-12)
def wzsig(t, blend):
    a = np.mean([ridge_z(t, hl) for hl in ENS], 0)
    return (1-blend)*a + blend*revz(t, 10)

def run(blend, Sd, Ed, sizing):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(Sd, Ed+1):
        cur=prc[:,t-1]; pos=np.zeros(nInst)
        if t<Ed and t>=96:
            wz=wzsig(t,blend)
            if sizing=="sign": pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
            else: pos[1:]=np.where(wz>=np.median(wz),1.0,-1.0)*(dlr[1:]/cur[1:])
            cap=dlr[0]/cur[0]; lpA=logp[0,:t]; mv=lpA[30:]-lpA[:-30]
            z=(mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
            pos[0]=float(np.clip(-np.clip(z,-3,3)/3.0*(1_000_000/cur[0]),-cap,cap))
            lim=(dlr/cur).astype(int); pos=np.clip(pos,-lim,lim).astype(int)
        else: pos=cp.copy()
        dp=pos-cp; cash-=cur.dot(dp)+comm; comm=np.sum(cur*np.abs(dp)*commRate); cp=pos.copy()
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>Sd: pll.append(pl)
    pll=np.array(pll); mu,sd=pll.mean(),pll.std(); sr=np.sqrt(250)*mu/sd
    return mu, sd, mu*sr**2/(sr**2+1)

print("500-750 leg: mu (mean PnL) / std / score")
for blend in (0.20, 0.30):
    mu_s,sd_s,sc_s = run(blend,500,750,"sign")
    mu_b,sd_b,sc_b = run(blend,500,750,"bal")
    print(f"  b{blend}:  sign  mu={mu_s:6.1f} std={sd_s:7.1f} score={sc_s:6.1f}")
    print(f"         bal   mu={mu_b:6.1f} std={sd_b:7.1f} score={sc_b:6.1f}   "
          f"(dmu={mu_b-mu_s:+.1f}, dstd={sd_b-sd_s:+.1f})")

print("\nimbalance diagnostics over 500-750:")
for blend in (0.20, 0.30):
    nlong=[]; net_beta=[]; imb=[]; fwd_mkt=[]
    for t in range(500, 749):                            # need r_all[0,t] (next-day index ret) to exist
        cur=prc[:,t-1]; wz=wzsig(t,blend); s=np.sign(wz)
        nlong.append(int((s>0).sum()))
        r=r_all[:,:t-1]; rA=r[0]-r[0].mean(); beta=((r[1:]-r[1:].mean(1,keepdims=True))@rA)/(rA@rA+1e-12)
        stk=s*((dlr[1:]/cur[1:]).astype(int))
        net_beta.append(float((stk*cur[1:])@beta))
        imb.append(float(s.sum()))                       # + => net long tilt
        fwd_mkt.append(float(r_all[0, t]))               # next-day index return
    nlong=np.array(nlong); net_beta=np.array(net_beta); imb=np.array(imb); fwd_mkt=np.array(fwd_mkt)
    c=np.corrcoef(imb, fwd_mkt)[0,1]                      # is the tilt informative about next-day market?
    print(f"  b{blend}: #long mean {nlong.mean():4.1f}/50  std {nlong.std():4.1f}   "
          f"net-beta$ std {net_beta.std():7,.0f}   corr(tilt, next-day index ret)={c:+.3f}")
