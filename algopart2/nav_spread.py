"""nav_spread.py — the last untested industry signal family: INDEX ARB / ETF-NAV convergence.
ALGO is the equal-weight index of the 50 names at corr ~0.976 (not 1.0), so ALGO ~ NAV + tracking
noise. Classic trade: fade the spread  s_t = log(ALGO) - mean(log names)  when it is rich/cheap.

(1) STRUCTURE: is the spread stationary/mean-reverting? (lag-1 autocorr of s and of ds, half-life)
(2) IC: does -z(spread) predict ALGO's next-day return? vs the shipped 30d-move fade's IC.
    And does it predict the spread's own convergence (the cleaner arb statement)?
(3) TRADED: in the shipped book frame (idio champ book + net-$ gate intact), replace the FADE branch
    with the spread-fade on non-gated days. Score on 500-750 + rolling 250d vs the shipped baseline.
The bar is NOT "beat zero" — it is "beat the shipped fade" for the same $100k ALGO capacity.
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)
HLS=[250,500,1000,2000]; BLEND=0.30; REV_W=10
CONTRA_DOL=1_000_000; CONTRA_K=30; CONTRA_WZ=60; ALGO_LL=50_000

# ---------------- (1) spread structure ----------------
s = logp[0] - logp[1:].mean(0)                    # the ALGO-vs-NAV spread (log units)
ds = np.diff(s)
r0 = np.diff(logp[0]); rn = np.diff(logp[1:].mean(0))
print("(1) STRUCTURE")
print(f"    corr(ALGO ret, NAV ret) = {np.corrcoef(r0, rn)[0,1]:+.4f}")
sc_ = s - s.mean()
ar1 = float(np.corrcoef(sc_[:-1], sc_[1:])[0,1])
print(f"    spread lag-1 autocorr AR(1) = {ar1:+.4f}   (1.0 = random walk, <1 = mean-reverting)")
hl = np.log(0.5)/np.log(abs(ar1)) if 0 < abs(ar1) < 1 else float('inf')
print(f"    implied half-life = {hl:.1f} days    | d(spread) lag-1 autocorr = {np.corrcoef(ds[:-1], ds[1:])[0,1]:+.4f}")
print(f"    spread std = {s.std():.4f} log-units (~{100*s.std():.2f}% of ALGO price)")

# ---------------- (2) IC of the spread signal on ALGO ----------------
def zspread(t, W=60):
    ss = s[:t]; w = ss[-W:]
    return float((ss[-1] - w.mean()) / (w.std() + 1e-12))
def zfade(t):
    lpA = logp[0,:t]; mv = lpA[CONTRA_K:] - lpA[:-CONTRA_K]
    return float((mv[-1] - mv[-CONTRA_WZ:].mean()) / (mv[-CONTRA_WZ:].std() + 1e-12))
print("\n(2) IC on ALGO next-day return (500-750) and on next-day spread convergence")
for W in (20, 60, 120):
    za = np.array([zspread(t, W) for t in range(500, 750)])
    fwdA = np.array([logp[0,t] - logp[0,t-1] for t in range(500, 750)])      # ALGO next-day ret
    fwdS = np.array([s[t] - s[t-1] for t in range(500, 750)])                # spread change
    icA = float(np.corrcoef(-za, fwdA)[0,1]); icS = float(np.corrcoef(-za, fwdS)[0,1])
    n = len(za); tA = icA*np.sqrt((n-2)/(1-icA**2+1e-12)); tS = icS*np.sqrt((n-2)/(1-icS**2+1e-12))
    print(f"    W={W:>3}:  IC(-z, ALGO ret) = {icA:+.4f} (t={tA:+.2f})   IC(-z, d spread) = {icS:+.4f} (t={tS:+.2f})")
zf = np.array([zfade(t) for t in range(500, 750)])
fwdA = np.array([logp[0,t] - logp[0,t-1] for t in range(500, 750)])
icF = float(np.corrcoef(-zf, fwdA)[0,1]); tF = icF*np.sqrt((len(zf)-2)/(1-icF**2+1e-12))
print(f"    shipped fade: IC(-z30, ALGO ret) = {icF:+.4f} (t={tF:+.2f})   <- the bar to beat")

# ---------------- (3) traded, in the shipped book frame ----------------
_wc = {}
def wz_at(t):
    if t in _wc: return _wc[t]
    lp=logp[:,:t]; r=lp[:,1:]-lp[:,:-1]; X=r[:,:-1].T; Y=r[1:,1:].T; xin=r[:,-1]; n=X.shape[0]; fs=[]
    for hlv in HLS:
        lam=0.5**(1/hlv); w=lam**np.arange(n-1,-1,-1); sw=w.sum()
        mx=(w[:,None]*X).sum(0)/sw; my=(w[:,None]*Y).sum(0)/sw; Xc=X-mx; Yc=Y-my
        B=np.linalg.solve(Xc.T@(w[:,None]*Xc)+0.1*np.eye(nInst), Xc.T@(w[:,None]*Yc))
        f=my+(xin-mx)@B; f=f-f.mean(); fs.append(f/(f.std()+1e-12))
    a=np.mean(fs,0); rr=logp[1:,t-1]-logp[1:,t-1-REV_W]; rr=rr-rr.mean(); rv=-rr/(rr.std()+1e-12)
    wz=(1-BLEND)*a+BLEND*rv; _wc[t]=wz; return wz

def positions(t, algo_mode, W=60):
    cur=prc[:,t-1]; pos=np.zeros(nInst); wz=wz_at(t); pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
    ii=np.clip(pos[1:],-(dlr[1:]/cur[1:]).astype(int),(dlr[1:]/cur[1:]).astype(int)).astype(int)
    net=float((ii*cur[1:]).sum()); cap=dlr[0]/cur[0]
    if abs(net)>=ALGO_LL:
        av=float(np.sign(net)*cap)                              # net-$ gate intact in all modes
    elif algo_mode=="spread":
        av=float(np.clip(-np.clip(zspread(t,W),-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    elif algo_mode=="blend":                                    # avg of fade and spread signals
        z2=(-np.clip(zfade(t),-3,3)-np.clip(zspread(t,W),-3,3))/2.0
        av=float(np.clip(z2/3.0*(CONTRA_DOL/cur[0])*2,-cap,cap))
    else:
        av=float(np.clip(-np.clip(zfade(t),-3,3)/3.0*(CONTRA_DOL/cur[0]),-cap,cap))
    pos[0]=av; lim=(dlr/cur).astype(int); return np.clip(pos,-lim,lim).astype(int)

def run(mode,S,E):
    cash=0.0; cp=np.zeros(nInst); value=0.0; comm=0.0; pll=[]
    for t in range(S,E+1):
        cur=prc[:,t-1]; newPos=positions(t,mode) if t<E else cp
        dP=newPos-cp; cash-=cur.dot(dP)+comm; comm=np.sum(cur*np.abs(dP)*commRate); cp=newPos
        pl=cash+cp.dot(cur)-value; value=cash+cp.dot(cur)
        if t>S: pll.append(pl)
    p=np.array(pll); mu,sd=p.mean(),p.std()
    scv=mu*(np.sqrt(250)*mu/sd)**2/((np.sqrt(250)*mu/sd)**2+1) if mu>0 else mu
    return mu,sd,scv

print("\n(3) TRADED in the shipped frame (idio champ + net-$ gate; only the non-gated ALGO branch differs)")
print(f"    {'ALGO branch':<12}{'mean$':>8}{'std$':>8}{'SCORE':>8}   [500-750]")
for m in ("fade","spread","blend"):
    mu,sd,scv=run(m,500,750)
    tag="  <-- shipped baseline (694)" if m=="fade" else ""
    print(f"    {m:<12}{mu:>8.0f}{sd:>8.0f}{scv:>8.0f}{tag}")
print(f"\n    rolling 250d (step 25): mean SCORE / floor")
ends=[e for e in range(400,nDays+1,25) if e-250>=96]
for m in ("fade","spread","blend"):
    ss=[run(m,e-250,e)[2] for e in ends]
    print(f"    {m:<12} mean {np.mean(ss):>5.0f}   floor {np.min(ss):>5.0f}")
