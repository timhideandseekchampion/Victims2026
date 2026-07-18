"""edge_probe.py — is there any UNCAPTURED edge? Two cheap probes, no scipy.
(A) fitted multi-signal combo IC OOS vs the ~0.079 baseline (does a fuller fit add anything?)
(B) is the INDEX's own next-day return predictable from today's cross-section? (we only fade it)"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc)

_rc = {}
def ridge(t, hl, a=0.1):
    key = (t, hl)
    if key in _rc: return _rc[key]
    L = lp[:, :t]; r = L[:, 1:] - L[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5**(1/hl); w = lam**np.arange(n-1,-1,-1); sw = w.sum()
    mx = (w[:,None]*X).sum(0)/sw; my = (w[:,None]*Y).sum(0)/sw
    Xc = X-mx; Yc = Y-my
    B = np.linalg.solve(Xc.T@(w[:,None]*Xc)+a*np.eye(nInst), Xc.T@(w[:,None]*Yc))
    f = my + (xin-mx)@B; f = f-f.mean(); v = f/(f.std()+1e-12); _rc[key]=v; return v
def revz(t, w):
    rr = lp[1:, t-1]-lp[1:, t-1-w]; rr = rr-rr.mean(); return -rr/(rr.std()+1e-12)

def sigmat(t):
    return np.column_stack([ridge(t,500), ridge(t,2000), revz(t,5), revz(t,10), revz(t,40)])
names = ["ridge500","ridge2000","revz5","revz10","revz40"]
def gather(S,E):
    Xs=[]; ys=[]
    for t in range(max(S,96), min(E,nt-1)):
        s=sigmat(t); fwd=lp[1:,t]-lp[1:,t-1]; fwd=fwd-fwd.mean(); Xs.append(s); ys.append(fwd)
    return np.vstack(Xs), np.concatenate(ys)
Xf,yf = gather(250,500)
wgt = np.linalg.solve(Xf.T@Xf+1e-6*np.eye(5), Xf.T@yf); wgt/=np.abs(wgt).sum()
print("(A) fitted combo weights (250-500):", {n:round(float(x),3) for n,x in zip(names,wgt)})
def combo_ic(S,E,w):
    ics=[]
    for t in range(max(S,96), min(E,nt-1)):
        s=sigmat(t)@w; fwd=lp[1:,t]-lp[1:,t-1]; fwd=fwd-fwd.mean()
        if s.std()>1e-12 and fwd.std()>1e-12: ics.append(np.corrcoef(s,fwd)[0,1])
    ics=np.array(ics); t_=ics.mean()/(ics.std(ddof=1)/np.sqrt(len(ics)))
    return ics.mean(), t_
for lbl,S,E in [("fit 250-500",250,500),("OOS 500-750",500,749),("all 400-750",400,749)]:
    ic,tt = combo_ic(S,E,wgt); print(f"    {lbl}: IC={ic:.4f} t={tt:.2f}")
ic5,t5 = combo_ic(500,749, np.array([0.7,0,0,0.3,0]))
print(f"    baseline (0.7*ridge500+0.3*revz10) OOS 500-749: IC={ic5:.4f} t={t5:.2f}")

print("\n(B) INDEX next-day predictability from today's cross-section (we currently only FADE it):")
# predict r0[t+1] from full return cross-section r[:,t] via forgetting ridge, measure OOS corr
def idx_probe(S,E,hl=1000,a=0.1):
    preds=[]; acts=[]
    for t in range(max(S,96), min(E,nt-1)):
        L=lp[:,:t]; r=L[:,1:]-L[:,:-1]
        X=r[:,:-1].T; y=r[0,1:]; xin=r[:,-1]
        n=X.shape[0]; lam=0.5**(1/hl); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
        mx=(w[:,None]*X).sum(0)/sw; my=(w*y).sum()/sw
        Xc=X-mx; yc=y-my
        B=np.linalg.solve(Xc.T@(w[:,None]*Xc)+a*np.eye(nInst), Xc.T@(w*yc))
        pred=my+(xin-mx)@B; act=lp[0,t]-lp[0,t-1]
        preds.append(pred); acts.append(act)
    preds=np.array(preds); acts=np.array(acts)
    return np.corrcoef(preds,acts)[0,1]
for lbl,S,E in [("500-750",500,749),("250-500",250,500),("400-750",400,749)]:
    c=idx_probe(S,E); print(f"    corr(pred_index_nextret, actual) {lbl}: {c:.4f}")
