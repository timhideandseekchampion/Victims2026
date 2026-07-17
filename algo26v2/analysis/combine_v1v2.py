"""Combine v1's peer-lead-lag ridge book with v2's cointegration-pairs overlay.

Hypothesis: pairs deploy otherwise-idle capital on the ~9 names/day the ridge's
conviction gate skips -> orthogonal add. Test honestly: rolling (leakage-safe) pair
selection AND fixed in-sample pairs (optimistic upper bound), on 250 & 440 + H1/H2.
"""
import json
import numpy as np
import pandas as pd
from strat_engine import Engine, cfg

prc_all = pd.read_csv("../prices.txt", sep=r"\s+", header=0, index_col=None).values.T
nInst, nt = prc_all.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlrLim = np.full(nInst, 10_000); dlrLim[0] = 100_000

# ---- v1 ridge book (HALF_LIFE=2000, ALPHA=0.1, CONV_Z=0.2, ALGO contra 200k, hedge) ----
def ridge_fit(X, Y, hl=2000, alpha=0.1):
    n, p = X.shape
    lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + alpha) * np.eye(p), XtWY), mx, my


def make_combined(pair_weight=0.0, pair_cfg=None, gate_only=False):
    cache = {"fit_t": None, "model": None}
    eng = Engine(pair_cfg) if pair_weight > 0 else None

    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 60:
            return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:] - lp[:, :-1]
        if cache["fit_t"] != t:
            cache["model"] = ridge_fit(ret[:, :-1].T, ret[1:, 1:].T); cache["fit_t"] = t
        B, mx, my = cache["model"]
        pred = my + (ret[:, -1] - mx) @ B
        w = pred - pred.mean()
        sized = np.sign(w) * (10_000 / prc[1:, -1])
        keep = np.abs(w) >= 0.2 * (np.std(w) + 1e-12)
        sized = np.where(keep, sized, 0.0)
        pos[1:] = sized
        # ALGO contrarian
        cap = 100_000 / prc[0, -1]; rev = 0.0
        if t > 92:
            lpA = np.log(prc[0]); mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            rev = -float(np.clip(z, -3, 3)) * 200_000 / prc[0, -1]
        rev = float(np.clip(rev, -cap, cap))
        # v2 pair overlay
        if eng is not None:
            ppos = eng._pairs(prc, lp, prc[:, -1])
            if gate_only:                      # only add pair legs on names the gate SKIPPED
                skipped = pos[1:] == 0
                mask = np.ones(ni, bool); mask[1:] = skipped
                ppos = ppos * mask
            pos += pair_weight * ppos
        # hedge into leftover ALGO room
        rA = ret[0]; rAc = rA - rA.mean(); den = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / den
        net = (pos[1:] * prc[1:, -1]) @ betas
        hedge = -net / prc[0, -1]
        room = max(cap - abs(rev), 0.0)
        pos[0] = rev + float(np.clip(hedge, -room, room))
        return pos.astype(int)
    return gp


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1)


def run(gp, start, end):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(start, end + 1):
        p = prc_all[:, :t]; c = p[:, -1]
        if t < end:
            lim = (dlrLim / c).astype(int); npos = np.clip(gp(p), -lim, lim).astype(int)
        else:
            npos = cp.copy()
        d = npos - cp; cash -= c.dot(d) + cm; dv = c * np.abs(d); cm = (dv * commRate).sum()
        cp = npos.copy(); pl = cash + cp.dot(c) - val; val = cash + cp.dot(c)
        if t > start: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    return score(mu, sd), (np.sqrt(250) * mu / sd if sd > 0 else 0)


fixed = json.load(open("results/best_combo.json"))["edge_cfgs"]["pairs"]["fixed_pairs"]
pc_roll = cfg(pmax=0.02, max_pairs=24, pair_lb=90, pair_entry=1.0, pair_exit=0.3,
              pair_dollars=10000, w_pairs=1.0)
pc_fixed = cfg(pair_lb=90, pair_entry=1.0, pair_exit=0.3, pair_dollars=10000,
               w_pairs=1.0, fixed_pairs=fixed[:24])

print(f"{'config':34} {'S@250':>7} {'S@440':>7} {'H1':>6} {'H2':>6}")
def show(name, gpf):
    s250, _ = run(gpf(), nt - 250, nt)
    s440, _ = run(gpf(), nt - 440, nt)
    h1, _ = run(gpf(), 60, 280)
    h2, _ = run(gpf(), 280, 500)
    print(f"{name:34} {s250:7.1f} {s440:7.1f} {h1:6.0f} {h2:6.0f}")

show("v1 ridge alone", lambda: make_combined(0.0))
show("+ pairs(roll,24) full", lambda: make_combined(1.0, pc_roll))
show("+ pairs(roll,24) gate-only", lambda: make_combined(1.0, pc_roll, gate_only=True))
show("+ pairs(fixed,24) full", lambda: make_combined(1.0, pc_fixed))
show("+ pairs(fixed,24) gate-only", lambda: make_combined(1.0, pc_fixed, gate_only=True))

print("\n=== other v2 edges (causal) layered on ridge ===")
from strat_engine import Engine as E2
def make_edge(edge, dollars, extra=None):
    cache={"fit_t":None,"model":None}
    kw={f"w_{edge}":1.0}; 
    if edge=="xs": kw["xs_dollars"]=dollars
    elif edge=="lead": kw["lead_dollars"]=dollars
    elif edge=="corr": kw["corr_dollars"]=dollars
    elif edge=="mf": kw["mf_dollars"]=dollars
    eng=E2(cfg(**kw))
    def gp(prc):
        ni,t=prc.shape; pos=np.zeros(ni)
        if t<60: return pos.astype(int)
        lp=np.log(prc); ret=lp[:,1:]-lp[:,:-1]
        if cache["fit_t"]!=t:
            cache["model"]=ridge_fit(ret[:,:-1].T,ret[1:,1:].T); cache["fit_t"]=t
        B,mx,my=cache["model"]; pred=my+(ret[:,-1]-mx)@B; w=pred-pred.mean()
        s=np.sign(w)*(10000/prc[1:,-1]); s=np.where(np.abs(w)>=0.2*(np.std(w)+1e-12),s,0.0); pos[1:]=s
        cap=100000/prc[0,-1]; rev=0.0
        if t>92:
            lpA=np.log(prc[0]); mv=lpA[30:]-lpA[:-30]
            z=(mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12); rev=-float(np.clip(z,-3,3))*200000/prc[0,-1]
        rev=float(np.clip(rev,-cap,cap))
        if edge=="xs": pos+=eng._xs(prc,prc[:,-1])
        elif edge=="lead": pos+=eng._lead(lp,prc[:,-1])
        elif edge=="corr": pos+=eng._corr(prc,lp,prc[:,-1])
        elif edge=="mf": pos+=eng._multifactor(lp,prc[:,-1])
        rA=ret[0]; rAc=rA-rA.mean(); den=rAc@rAc+1e-12
        betas=((ret[1:]-ret[1:].mean(1,keepdims=True))@rAc)/den
        net=(pos[1:]*prc[1:,-1])@betas; room=max(cap-abs(rev),0.0)
        pos[0]=rev+float(np.clip(-net/prc[0,-1],-room,room))
        return pos.astype(int)
    return gp
for edge,dol in [("xs",9000),("lead",5000),("corr",4000),("mf",3000)]:
    show(f"+ {edge} (causal)", lambda e=edge,d=dol: make_edge(e,d))

# clean OOS pair test: select pairs on days 0-250 ONLY, trade 251-500 on ridge
print("\n=== clean OOS: pairs selected on 0-250, traded 251-500 ===")
from statsmodels.tsa.stattools import coint
win=np.log(prc_all[:,:250]); rr=np.diff(win,axis=1); C=np.corrcoef(rr); oos=[]
for i in range(nInst):
    for j in range(i+1,nInst):
        if abs(C[i,j])>0.4:
            try:
                if coint(win[i],win[j])[1]<0.02: oos.append([i,j])
            except: pass
print(f"selected {len(oos)} pairs on days 0-250")
pc_oos=cfg(pair_lb=90,pair_entry=1.0,pair_exit=0.3,pair_dollars=10000,w_pairs=1.0,fixed_pairs=oos[:24])
r_alone,_=run(make_combined(0.0),280,500)
r_oos,_=run(make_combined(1.0,pc_oos),280,500)
print(f"ridge alone (251-500): {r_alone:.1f}   + OOS-selected pairs: {r_oos:.1f}   delta {r_oos-r_alone:+.1f}")
