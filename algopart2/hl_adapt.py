"""hl_adapt.py — the half-life trade-off: shorter HL = faster regime adaptation, but does it cost
score in the stable regime? Test each HL config on BOTH:
  (A) normal 500-750  (the COST: does short HL lower the graded score?)
  (B) injected momentum & flip regimes after 750 (the BENEFIT: does short HL adapt faster?)
Idio = sign(ensemble lead-lag blended with reversion); same net-$ ALGO gate as SAFE_lldollar.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
BLEND=0.30; REV_W=10; CONTRA_DOL=1_000_000; CONTRA_K=30; CONTRA_WZ=60; ALGO_LL=50_000

def make_ext(kind, T_ext=150, mom=0.6, period=25, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc).copy()
    vol = np.diff(logp[1:], axis=1).std(); names = logp[1:, :].copy(); K = 5
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        if kind == "noise": drift = np.zeros(50)
        elif kind == "flip":
            sgn = 1.0 if (step // period) % 2 == 0 else -1.0
            drift = sgn * mom * (tc/(tc.std()+1e-9)) * vol
        else: drift = mom * (tc/(tc.std()+1e-9)) * vol
        drift -= drift.mean(); noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1]+drift+noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :nDays] = prc
    return full

_cache = {}
def ridge_z(P, tag, t, hl, a=0.1):
    key = (tag, t, hl)
    if key in _cache: return _cache[key]
    lp = np.log(P[:, :t]); r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
    B = np.linalg.solve(Xc.T@(w[:, None]*Xc) + a*np.eye(nInst), Xc.T@(w[:, None]*Yc))
    f = my + (xin-mx)@B; f = f-f.mean(); v = f/(f.std()+1e-12); _cache[key] = v; return v

def run(P, tag, hls, S, E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S, E+1):
        cur=P[:,t-1]; pos=np.zeros(nInst)
        if t<E and t>=96:
            a = np.mean([ridge_z(P, tag, t, hl) for hl in hls], 0)
            lp=np.log(P[:,:t]); rr=lp[1:,-1]-lp[1:,-1-REV_W]; rr=rr-rr.mean(); rv=-rr/(rr.std()+1e-12)
            wz=(1-BLEND)*a+BLEND*rv; pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
            ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
            net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]
            if abs(net)>=ALGO_LL: av=float(np.sign(net)*cap)
            else:
                lpA=lp[0]; mv=lpA[CONTRA_K:]-lpA[:-CONTRA_K]
                z=(mv[-1]-mv[-CONTRA_WZ:].mean())/(mv[-CONTRA_WZ:].std()+1e-12)
                av=float(np.clip(-np.clip(z,-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
            pos[0]=av; lim=(dlr/cur).astype(int); pos=np.clip(pos,-lim,lim).astype(int)
        else: pos=cp.copy()
        dP=pos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=pos.copy()
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    return np.array(pll)

def score(pll):
    mu,sd=pll.mean(),pll.std()
    if mu<=0 or sd<1e-10: return mu
    sr=np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)

configs = {"ens(current)":[250,500,1000,2000], "shorter":[125,250,500,1000],
           "fast":[60,125,250], "single250":[250], "single60":[60]}

print("(A) NORMAL 500-750  — the COST of a shorter memory in the stable regime")
print(f"    {'config':<16}{'mean$':>8}{'std$':>8}{'Sharpe':>8}{'SCORE':>8}")
for name, hls in configs.items():
    pll = run(prc, "real", hls, 500, 750); mu,sd=pll.mean(),pll.std()
    print(f"    {name:<16}{mu:>8.0f}{sd:>8.0f}{np.sqrt(250)*mu/sd:>8.2f}{score(pll):>8.0f}")

for kind in ("momentum","flip"):
    full = make_ext(kind)
    print(f"\n(B) {kind.upper()} regime 750-900 — the BENEFIT of faster adaptation")
    print(f"    {'config':<16}{'totalPnL':>10}{'first50d':>10}{'maxDD':>9}")
    for name, hls in configs.items():
        pll = run(full, f"{kind}", hls, nDays, nDays+150)
        cum=pll.cumsum(); dd=(cum-np.maximum.accumulate(cum)).min()
        print(f"    {name:<16}{cum[-1]:>10.0f}{cum[min(50,len(cum))-1]:>10.0f}{dd:>9.0f}")
