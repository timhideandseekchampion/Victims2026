"""Consistency check: don't trust the single last-250 score (763). Run ONE continuous
backtest, then re-score every rolling sub-window to see the DISTRIBUTION of scores a
fresh 250-day window might give. Compares the full book vs lean (idio-only, no ALGO
overlay) vs the reversion-blend variant.
"""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
comm = np.full(nInst, 1e-4); comm[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000


def ridge_fit(X, Y, hl=2000, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my


def make(algo_overlay=True, blend=0.0, rev_w=10):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 60: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = ridge_fit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        wz = w/(np.std(w)+1e-12)
        if blend > 0:
            r = ret[1:, -rev_w:].sum(1); r -= r.mean(); wz = (1-blend)*wz+blend*(-r/(np.std(r)+1e-12))
        s = np.sign(wz)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(wz) >= 0.2*(np.std(wz)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]; rev = 0.0
        if algo_overlay and t > 92:
            lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
            rev = float(np.clip(-np.clip(z, -3, 3)*200000/prc[0, -1], -cap, cap))
        rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    return gp


def daily_pnl(gp, start=100):
    """One continuous run from `start` to nt; return per-day PnL array (len nt-start)."""
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(start, nt+1):
        p = prc[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < nt else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*comm).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > start: pll.append(pl)
    return np.array(pll)


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)


def roll_scores(pll, W):
    return np.array([score(pll[s:s+W]) for s in range(0, len(pll)-W+1)])


configs = {
    "full book (ridge+ALGO)": make(True, 0.0),
    "lean (idio only, no ALGO)": make(False, 0.0),
    "revblend (ridge+rev+ALGO)": make(True, 0.2, 10),
}
print("Continuous run from day 100 -> 500 (400 scored days). Re-scoring rolling windows.\n")
for name, gp in configs.items():
    pll = daily_pnl(gp, start=100)
    print(f"### {name}   [full-run mean ${pll.mean():.0f}/day, ann.Sharpe {np.sqrt(250)*pll.mean()/pll.std():.2f}]")
    for W in (125, 250):
        rs = roll_scores(pll, W)
        pct = 100*np.mean(rs > 300)
        print(f"   W={W}: median {np.median(rs):6.0f}  mean {rs.mean():6.0f}  min {rs.min():6.0f}  "
              f"max {rs.max():6.0f}  p10 {np.percentile(rs,10):6.0f}  p25 {np.percentile(rs,25):6.0f}  "
              f">{300}:{pct:3.0f}%")
    print()
