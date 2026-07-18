"""trigger_stats.py — when does the |frac|>=0.12 gate actually fire, which way, how often does it
DISAGREE with reversion, and does the ALGO move go its way the next day?"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = [250, 500, 1000, 2000]; GATE = 0.12

def sig(t):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]; X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]
    fs = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1,-1,-1); sw = w.sum()
        mx = (w[:,None]*X).sum(0)/sw; my = (w[:,None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
        B = np.linalg.solve(Xc.T@(w[:,None]*Xc)+0.1*np.eye(nInst), Xc.T@(w[:,None]*Yc))
        f = my+(xin-mx)@B; d = f-f.mean(); fs.append(d/(d.std()+1e-12))
    wz = 0.7*np.mean(fs,0) + 0.3*(lambda rr:-(rr-rr.mean())/(rr.std()+1e-12))(logp[1:,t-1]-logp[1:,t-1-10])
    frac = float(np.mean(np.sign(wz)))
    lpA = logp[0,:t]; mv = lpA[30:]-lpA[:-30]; z=(mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev_dir = -np.sign(np.clip(z,-3,3))                       # direction reversion would take
    return frac, rev_dir

for lbl,(S,E) in {"500-750":(500,749),"400-500":(400,499),"250-400":(250,399)}.items():
    F=[]; RD=[]; Y=[]
    for t in range(S,E):
        fr,rd = sig(t); F.append(fr); RD.append(rd); Y.append(float(r_all[0,t]))
    F=np.array(F); RD=np.array(RD); Y=np.array(Y); N=len(F)
    on = np.abs(F)>=GATE; nlong_thresh = (F[on]>0)
    disagree = on & (np.sign(F)!=RD)                          # gate flips vs reversion
    # does ALGO move the gate's way next day? aligned return = sign(frac)*next-ret on trigger days
    aligned = np.sign(F[on])*Y[on]
    print(f"[{lbl}] N={N}  triggers={on.sum()} ({100*on.sum()/N:.0f}%)  "
          f"long={int(nlong_thresh.sum())} short={int((~nlong_thresh).sum())}  "
          f"flips-vs-reversion={disagree.sum()} ({100*disagree.sum()/max(1,on.sum()):.0f}% of triggers)")
    print(f"         next-day ALGO move in gate's direction: mean {aligned.mean():+.5f}  "
          f"hit-rate {100*np.mean(aligned>0):.0f}%   (vs all-day |move| {np.abs(Y).mean():.5f})")
