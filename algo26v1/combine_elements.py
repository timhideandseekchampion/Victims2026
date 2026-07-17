"""Test the GENUINE v1-vs-v2 differences as combinations on the ridge core.

v1 ALGO leg = fade 30-day index move (long horizon). v2 ALGO leg = fade 5-day (short).
These are different horizons -> a real thing to blend. Also test v2's active residual
reversion. Measured on known windows @250/@440.
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


def algo_rev(prc, k, wz, dollars):
    """Fade the k-day ALGO move, z-scored over wz days. Returns desired shares (unclipped)."""
    t = prc.shape[1]
    if t <= k+wz+2: return 0.0
    lpA = np.log(prc[0]); mv = lpA[k:]-lpA[:-k]
    z = (mv[-1]-mv[-wz:].mean())/(mv[-wz:].std()+1e-12)
    return -float(np.clip(z, -3, 3))*dollars/prc[0, -1]


def make(algo_mode="v1", corr_dollars=0.0):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        # v2 active residual-vs-ALGO reversion (per name), competes for the $10k caps
        if corr_dollars > 0 and t > 92:
            la = lp[0, -90:]
            for i in range(1, ni):
                beta = np.polyfit(la, lp[i, -90:], 1)[0]; resid = lp[i]-beta*lp[0]
                z = (resid[-1]-resid[-90:].mean())/(resid[-90:].std()+1e-9)
                if abs(z) > 1.0: pos[i] += -np.sign(z)*corr_dollars/prc[i, -1]
        cap = 100000/prc[0, -1]
        if algo_mode == "v1":      rev = algo_rev(prc, 30, 60, 200000)
        elif algo_mode == "v2":    rev = algo_rev(prc, 5, 60, 200000)
        elif algo_mode == "blend": rev = 0.5*algo_rev(prc, 30, 60, 200000)+0.5*algo_rev(prc, 5, 20, 200000)
        rev = float(np.clip(rev, -cap, cap))
        rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    return gp


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)
def run(gpf, nd):
    gp = gpf(); cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(nt-nd, nt+1):
        p = prc_all[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < nt else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > nt-nd: pll.append(pl)
    return score(np.array(pll))


print(f"{'config':40} {'S@250':>8} {'S@440':>8}")
for name, kw in [
    ("ridge + v1 ALGO(30d)  [current]", dict(algo_mode="v1")),
    ("ridge + v2 ALGO(5d)", dict(algo_mode="v2")),
    ("ridge + blend ALGO(30d+5d)", dict(algo_mode="blend")),
    ("ridge + v1 ALGO + v2 residual-rev $3k", dict(algo_mode="v1", corr_dollars=3000)),
    ("ridge + v1 ALGO + v2 residual-rev $6k", dict(algo_mode="v1", corr_dollars=6000)),
]:
    print(f"{name:40} {run(lambda kw=kw: make(**kw), 250):8.1f} {run(lambda kw=kw: make(**kw), 440):8.1f}")
