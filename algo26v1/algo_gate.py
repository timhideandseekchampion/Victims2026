"""Is the ALGO-autocorr regime EXPLOITABLE? Gate the ALGO contra leg by recent ALGO return
autocorrelation: full strength when mean-reverting (ac<=thr), scaled/off when trending (ac>thr).
Score the FULL book @250/@440 + both halves vs always-on."""
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


def make(gate=None, trend_mult=0.0):
    """gate=None -> always on. gate=thr -> when ALGO ret-autocorr > thr (trending), scale contra by trend_mult."""
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]
        lpA = np.log(prc[0]); rA = lpA[1:]-lpA[:-1]; mv = lpA[30:]-lpA[:-30]
        z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
        dollars = 200_000
        if gate is not None:
            ac = np.corrcoef(rA[-40:-1], rA[-39:])[0, 1]
            if ac > gate: dollars *= trend_mult          # trending regime -> throttle the contra leg
        rev = float(np.clip(-np.clip(z, -3, 3)*dollars/prc[0, -1], -cap, cap))
        rA0 = ret[0]; rAc = rA0-rA0.mean(); den = rAc@rAc+1e-12
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


print(f"{'ALGO-leg rule':38} {'S@250':>8} {'S@440':>8} {'H1':>7} {'H2':>7}")
for name, kw in [
    ("always on (current)", dict()),
    ("OFF when trending (ac>0.05)", dict(gate=0.05, trend_mult=0.0)),
    ("OFF when trending (ac>0.0)", dict(gate=0.0, trend_mult=0.0)),
    ("HALF when trending (ac>0.05)", dict(gate=0.05, trend_mult=0.5)),
]:
    print(f"{name:38} {run(lambda kw=kw: make(**kw), nt-250, nt):8.1f} "
          f"{run(lambda kw=kw: make(**kw), nt-440, nt):8.1f} "
          f"{run(lambda kw=kw: make(**kw), 60, 280):7.0f} {run(lambda kw=kw: make(**kw), 280, 500):7.0f}")
