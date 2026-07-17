"""HUNT: what is the MAX causal score extractable from the revealed window 500-750, and how?
Aggressively search forecast x sizing x ALGO-leg x reversion x hedge, fit to 500-750. Report the
best configs (and their 250-500 score to see if any generalizes). If even a hard in-sample fit
can't clear ~600, then 700-800 is NOT reachable by strategy on this window (a hard fact). If it
can, we've found the recipe and check whether it holds on the other window."""
import itertools
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000

_fc = {}                                                    # forecast cache: (hl, t) -> (w, ret, cur)
def ewls(X, Y, hl, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my

def forecast(prc, hl):
    t = prc.shape[1]; key = (hl, t)
    if key in _fc: return _fc[key]
    lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
    B, mx, my = ewls(ret[:, :-1].T, ret[1:, 1:].T, hl); pred = my+(ret[:, -1]-mx)@B
    out = (pred-pred.mean(), ret, prc[:, -1]); _fc[key] = out; return out

def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)

def backtest(cfg, S, E):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(S, E+1):
        p = prc[:, :t]; cur = p[:, -1]
        if t < E:
            pos = np.zeros(nInst)
            if t >= 95:
                w, ret, _ = forecast(p, cfg["hl"])
                wz = w/(np.std(w)+1e-12)
                # reversion blend
                if cfg["blend"] > 0:
                    r = ret[1:, -cfg["revw"]:].sum(1); r -= r.mean()
                    wz = (1-cfg["blend"])*wz + cfg["blend"]*(-r/(np.std(r)+1e-12))
                sh = np.sign(wz)*(10000/cur[1:])
                pos[1:] = np.where(np.abs(wz) >= cfg["conv"], sh, 0.0)
                cap = 100000/cur[0]; rev = 0.0
                lpA = np.log(p[0]); mv = lpA[30:]-lpA[:-30]
                z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
                mode = cfg["algo"]
                if mode == "fade":   rev = -np.clip(z, -3, 3)*cfg["adol"]/cur[0]
                elif mode == "follow": rev = np.sign(mv[-1])*min(abs(z), 3)*cfg["adol"]/cur[0]
                elif mode == "long":  rev = cfg["adol"]/cur[0]
                rev = float(np.clip(rev, -cap, cap))
                hs = 0.0
                if cfg["hedge"]:
                    rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
                    betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
                    hs = -((pos[1:]*cur[1:])@betas)/cur[0]
                room = max(cap-abs(rev), 0.0); pos[0] = rev+float(np.clip(hs, -room, room))
            npos = np.clip(pos, -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int)
        else:
            npos = cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    return score(np.array(pll))

grid = dict(
    hl=[500, 1000], conv=[0.15, 0.2, 0.25], blend=[0.0, 0.15, 0.3], revw=[10],
    algo=["fade", "follow", "long", "off"], adol=[200_000, 500_000, 1_000_000], hedge=[True, False],
)
keys = list(grid); results = []
S2, E2 = nt-250, nt
for combo in itertools.product(*[grid[k] for k in keys]):
    cfg = dict(zip(keys, combo))
    if cfg["algo"] == "off": cfg["adol"] = 0
    sc = backtest(cfg, S2, E2)
    results.append((sc, cfg))
# dedup identical (algo=off collapses adol)
seen = set(); uniq = []
for sc, cfg in results:
    k = tuple(sorted(cfg.items()))
    if k in seen: continue
    seen.add(k); uniq.append((sc, cfg))
uniq.sort(key=lambda x: -x[0])
print(f"searched {len(uniq)} configs on 500-750. TOP 12 (with 250-500 for generalization):\n")
print(f"{'S@500-750':>10} {'S@250-500':>10}  config")
for sc, cfg in uniq[:12]:
    so = backtest(cfg, 250, 500)
    c = {k: cfg[k] for k in ("hl", "conv", "blend", "algo", "adol", "hedge")}
    print(f"{sc:10.0f} {so:10.0f}  {c}")
print(f"\nMAX achievable on 500-750 (hard in-sample fit): {uniq[0][0]:.0f}")
