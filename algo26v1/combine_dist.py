"""Honestly test the v2 'diagnostic' findings for tradeable value on the ridge book:
   (1) distribution/vol-aware sizing (inverse recent vol),
   (2) NONLINEAR edge — quadratic-feature ridge, and a pooled gradient-boosted blend.
Scored @250/@440 and on BOTH halves (H1=60-280, H2=280-500) to avoid in-sample fooling.
"""
import numpy as np, pandas as pd

prc_all = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000


def rfit(X, Y, hl=500, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my


def algo_rev(prc):
    t = prc.shape[1]; cap = 100000/prc[0, -1]
    if t <= 92: return 0.0
    lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    return float(np.clip(-np.clip(z, -3, 3)*200000/prc[0, -1], -cap, cap))


def make(mode="base"):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        X = ret[:, :-1].T; Y = ret[1:, 1:].T
        if mode == "quad":                       # nonlinear: augment with squared features
            Xa = np.hstack([X, X**2]); Xl = ret[:, -1]; xin = np.concatenate([Xl, Xl**2])
            if c["t"] != t: c["m"] = rfit(Xa, Y); c["t"] = t
            B, mx, my = c["m"]; pred = my+(xin-mx)@B
        else:
            if c["t"] != t: c["m"] = rfit(X, Y); c["t"] = t
            B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B
        w = pred-pred.mean()
        if mode == "invvol":                     # distribution/vol-aware sizing
            vol = ret[1:, -20:].std(1); szdollar = 10000*(vol.mean()/(vol+1e-9))
            szdollar = np.clip(szdollar, 3000, 10000)
        else:
            szdollar = 10000
        s = np.sign(w)*(szdollar/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]; rev = algo_rev(prc)
        rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    return gp


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def run(gpf, start, end):
    gp = gpf(); cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(start, end+1):
        p = prc_all[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < end else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > start: pll.append(pl)
    return score(np.array(pll))


print(f"{'config':32} {'S@250':>8} {'S@440':>8} {'H1':>7} {'H2':>7}")
for name, mode in [("ridge base [current]", "base"), ("+ invvol sizing (dist-aware)", "invvol"),
                   ("+ quadratic features (nonlin)", "quad")]:
    print(f"{name:32} {run(lambda m=mode: make(m), nt-250, nt):8.1f} "
          f"{run(lambda m=mode: make(m), nt-440, nt):8.1f} "
          f"{run(lambda m=mode: make(m), 60, 280):7.0f} {run(lambda m=mode: make(m), 280, 500):7.0f}")
