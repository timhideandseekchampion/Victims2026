"""stability.py — where does our PnL std come from, and what lowers it WITHOUT killing mean?
Decompose std (idio leg vs ALGO leg), test per-name vol dispersion (room for risk-parity),
and compare stability levers on the 500-750 leg: mean / std / Sharpe / score."""
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

def stats(pll):
    pll=np.array(pll); mu,sd=pll.mean(),pll.std()
    if mu<=0 or sd<1e-10: return mu,sd,0.0,mu
    sr=np.sqrt(250)*mu/sd; return mu,sd,sr,mu*sr**2/(sr**2+1)

def run(blend, Sd, Ed, mode="base", gross=1.0, voltarget=None):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(Sd, Ed+1):
        cur=prc[:,t-1]; pos=np.zeros(nInst)
        if t<Ed and t>=96:
            wz=wzsig(t,blend); s=np.sign(wz)
            if mode in ("base","noalgo","idio_scaled"):
                sh = s*(dlr[1:]/cur[1:])
                if mode=="idio_scaled": sh *= gross
                pos[1:]=sh
            elif mode=="invvol":
                # inverse-vol weights, normalized so the LARGEST name sits at $10k (rest scaled down)
                vol = r_all[1:, max(0,t-1-60):t-1].std(1) + 1e-9
                wt = (1.0/vol); wt = wt/wt.max()          # in (0,1], top name =1
                pos[1:] = s*wt*(dlr[1:]/cur[1:])
            if mode!="noalgo":
                cap=dlr[0]/cur[0]; lpA=logp[0,:t]; mv=lpA[30:]-lpA[:-30]
                z=(mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
                pos[0]=float(np.clip(-np.clip(z,-3,3)/3.0*(1_000_000/cur[0]),-cap,cap))
            lim=(dlr/cur).astype(int); pos=np.clip(pos,-lim,lim).astype(int)
        else: pos=cp.copy()
        dp=pos-cp; cash-=cur.dot(dp)+comm; comm=np.sum(cur*np.abs(dp)*commRate); cp=pos.copy()
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>Sd: pll.append(pl)
    return stats(pll)

# per-name vol dispersion (are the 50 names really identical vol?)
vols = r_all[1:, -250:].std(1)
print(f"per-name daily-vol dispersion (last 250d): min {vols.min():.4f}  median {np.median(vols):.4f}  "
      f"max {vols.max():.4f}  max/min ratio {vols.max()/vols.min():.2f}")

print("\n500-750 leg — mean / std / Sharpe / score:")
def show(lbl, res): print(f"  {lbl:<28} mu={res[0]:6.1f}  std={res[1]:7.1f}  SR={res[2]:.2f}  score={res[3]:6.1f}")
for blend,name in ((0.30,"SAFE b.30"),(0.20,"QUAL b.20")):
    print(f" [{name}]")
    show("baseline (full)",      run(blend,500,750,"base"))
    show("idio only (no ALGO)",  run(blend,500,750,"noalgo"))
    show("inverse-vol idio",     run(blend,500,750,"invvol"))
    for g in (0.85, 0.70):
        show(f"gross x{g}",       run(blend,500,750,"idio_scaled",gross=g))
