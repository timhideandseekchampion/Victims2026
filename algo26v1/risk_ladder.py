"""Tournament risk ladder on the combinedv3 book: measure each 'more-risk' lever's effect
on SCORE and on VARIANCE / regime-dependence (H1 vs H2), so the upside-vs-gamble tradeoff
is explicit. Levers: drop hedge, raw directional tilt (no demean), lower conviction gate
(bigger book), pin the ALGO cap ($1M contra)."""
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


def make(hedge=True, demean=True, conv_z=0.2, contra=200_000, hl=500):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T, hl=hl); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B
        w = (pred-pred.mean()) if demean else pred          # raw tilt = net directional exposure
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= conv_z*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]; rev = 0.0
        if t > 92:
            lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
            rev = float(np.clip(-np.clip(z, -3, 3)*contra/prc[0, -1], -cap, cap))
        hedge_sh = 0.0
        if hedge:
            rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
            betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
            hedge_sh = -((pos[1:]*prc[1:, -1])@betas)/prc[0, -1]
        room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(hedge_sh, -room, room)); return pos.astype(int)
    return gp


def stats(gpf, start, end):
    gp = gpf(); cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []; gross = []
    for t in range(start, end+1):
        p = prc_all[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < end else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > start: pll.append(pl); gross.append(np.abs(cp*cur).sum())
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sc = mu*( (np.sqrt(250)*mu/sd)**2/((np.sqrt(250)*mu/sd)**2+1) ) if mu > 0 and sd > 1e-10 else mu
    sh = np.sqrt(250)*mu/sd if sd > 0 else 0
    return sc, mu, sd, sh, np.mean(gross)


print(f"{'config':38} {'Score':>7} {'mean$':>7} {'std$':>7} {'Sharpe':>6} {'gross$k':>7}  {'H1':>6} {'H2':>6}")
ladder = [
    ("base [primary: hedge,neutral]", dict()),
    ("no hedge", dict(hedge=False)),
    ("no hedge + raw tilt", dict(hedge=False, demean=False)),
    ("no hedge + raw tilt + conv0.1", dict(hedge=False, demean=False, conv_z=0.1)),
    ("MAXRISK (+ contra $1M)", dict(hedge=False, demean=False, conv_z=0.1, contra=1_000_000)),
    ("MAXRISK + HL2000", dict(hedge=False, demean=False, conv_z=0.1, contra=1_000_000, hl=2000)),
]
for name, kw in ladder:
    sc, mu, sd, sh, g = stats(lambda kw=kw: make(**kw), nt-250, nt)
    h1 = stats(lambda kw=kw: make(**kw), 60, 280)[0]; h2 = stats(lambda kw=kw: make(**kw), 280, 500)[0]
    print(f"{name:38} {sc:7.0f} {mu:7.0f} {sd:7.0f} {sh:6.2f} {g/1000:7.0f}  {h1:6.0f} {h2:6.0f}")

print("\n=== SMART-aggressive (keep the market-neutral edge, max deployment) ===")
for name, kw in [
    ("neutral + nohedge + conv0.1 + contra1M", dict(hedge=False, demean=True, conv_z=0.1, contra=1_000_000)),
    ("neutral + nohedge + conv0.15 + contra400k", dict(hedge=False, demean=True, conv_z=0.15, contra=400_000)),
    ("neutral + conv0.1 + contra400k (hedge on)", dict(hedge=True, demean=True, conv_z=0.1, contra=400_000)),
]:
    sc, mu, sd, sh, g = stats(lambda kw=kw: make(**kw), nt-250, nt)
    h1 = stats(lambda kw=kw: make(**kw), 60, 280)[0]; h2 = stats(lambda kw=kw: make(**kw), 280, 500)[0]
    print(f"{name:44} {sc:7.0f} {mu:7.0f} {sd:7.0f} {sh:6.2f} {g/1000:7.0f}  {h1:6.0f} {h2:6.0f}")
