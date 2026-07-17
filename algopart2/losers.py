"""
losers.py — are ILVX/SMAH/FWWG/ELLT/ACIX CONSISTENT losers (real, actionable) or unlucky draws?
Tests: (1) per-window PnL for the 5 (consistent across time?); (2) rank-persistence of per-name PnL
across the two halves for ALL names (do losers stay losers?); (3) does DROPPING them help or hurt
the book (breadth vs. cutting dead weight)?
"""
import json, numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(prc.columns); P = prc.values.T
nInst, nt = P.shape
lp_all = np.log(P)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
D = json.load(open("positions_data.json"))
watch = ["ILVX", "SMAH", "FWWG", "ELLT", "ACIX"]
wi = [names.index(w) for w in watch]

# ---- (1) per-window PnL of the 5, from stored cumulative pnl ----
def win_pnl(pnl, a, b): return pnl[b] - pnl[a]
wins = [(130, 254), (254, 378), (378, 502), (502, 626), (626, 749)]
for strat in ("SAFE", "SWING"):
    pnl = D[strat]["pnl"]
    print(f"\n[1] {strat} per-window PnL ($) for the 5 watched names:")
    print(f"  {'name':<6}" + "".join(f"{a}-{b}".rjust(11) for a, b in wins) + "     total")
    for w, i in zip(watch, wi):
        cells = [win_pnl(pnl[i], a, b) for a, b in wins]
        print(f"  {w:<6}" + "".join(f"{c:11.0f}" for c in cells) + f"{sum(cells):10.0f}")
    # how many of the 5 windows each name loses
    for w, i in zip(watch, wi):
        negs = sum(win_pnl(pnl[i], a, b) < 0 for a, b in wins)
        print(f"    {w}: loses {negs}/5 windows")

# ---- (2) rank persistence: do per-name losers PERSIST across halves? ----
print("\n[2] Rank persistence (SAFE) — per-name PnL first half vs second half, all 50 idio names:")
pnl = np.array(D["SAFE"]["pnl"])
h1 = pnl[1:, 440] - pnl[1:, 130]        # first-half PnL per idio name
h2 = pnl[1:, 749] - pnl[1:, 440]        # second-half
from scipy.stats import spearmanr, pearsonr
rs, ps = spearmanr(h1, h2); rp, _ = pearsonr(h1, h2)
print(f"  corr(firstHalf, secondHalf) Spearman={rs:+.3f} (p={ps:.2f})  Pearson={rp:+.3f}")
w1 = set(np.argsort(h1)[:5]); w2 = set(np.argsort(h2)[:5])
print(f"  worst-5 names 1st half: {[names[i+1] for i in sorted(w1)]}")
print(f"  worst-5 names 2nd half: {[names[i+1] for i in sorted(w2)]}")
print(f"  overlap of worst-5 across halves: {len(w1&w2)}/5   (low overlap + ~0 corr => LUCK, not bad names)")

# ---- (3) does dropping the 5 help or hurt? (SAFE config: HL-ens, blend .3, hedge off, sign) ----
HLS = [250, 500, 1000, 2000]; _rc = {}
def ridge_z(t, hl, a=0.1):
    k = (t, hl)
    if k in _rc: return _rc[k]
    lp = lp_all[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = .5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw
    B = np.linalg.solve((X-mx).T@(w[:, None]*(X-mx))+a*np.eye(51), (X-mx).T@(w[:, None]*(Y-my)))
    f = my+(xin-mx)@B; f -= f.mean(); v = f/(f.std()+1e-12); _rc[k] = v; return v
def book(drop, Sd, Ed, blend=.3):
    cash=0.;cp=np.zeros(nInst);val=0.;cm=0.;pll=[]
    dropmask=np.ones(nInst);
    for i in drop: dropmask[i]=0.0
    for t in range(Sd, Ed+1):
        cur=P[:, t-1];pos=np.zeros(nInst)
        if t<Ed and t>=130:
            core=np.mean([ridge_z(t, hl) for hl in HLS], 0)
            rr=lp_all[1:, t-1]-lp_all[1:, t-11];rr-=rr.mean();wz=(1-blend)*core+blend*(-rr/(rr.std()+1e-12))
            pos[1:]=np.sign(wz)*(dlr[1:]/cur[1:])
            cap=dlr[0]/cur[0];lpA=lp_all[0,:t];mv=lpA[30:]-lpA[:-30]
            z=(mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
            pos[0]=float(np.clip(-np.clip(z,-3,3)/3.*(1e6/cur[0]),-cap,cap))
            pos*=dropmask
            lim=(dlr/cur).astype(int);pos=np.clip(pos,-lim,lim).astype(int)
        else: pos=cp.copy()
        d=pos-cp;cash-=cur.dot(d)+cm;cm=np.sum(cur*np.abs(d)*commRate);cp=pos.copy()
        pl=cash+cp.dot(cur)-val;val=cash+cp.dot(cur)
        if t>Sd: pll.append(pl)
    pll=np.array(pll);mu,sd=pll.mean(),pll.std()
    if mu<=0 or sd<1e-9: return mu
    sr=np.sqrt(250)*mu/sd;return mu*sr**2/(sr**2+1)
print("\n[3] DROP-the-5 test (SAFE config) — does excluding the losers help?")
legs=[(250,500),(350,600),(500,750)]
print(f"  {'window':<10}{'keep all':>10}{'drop 5':>10}{'diff':>8}")
for S,E in legs:
    k=book([],S,E);d=book(wi,S,E);print(f"  {f'{S}-{E}':<10}{k:10.0f}{d:10.0f}{d-k:8.0f}")
