"""ALGO leg where a rolling OLS decides fade-vs-follow, weighted by statistical strength.

Instead of hardcoding 'fade' (coefficient -1 on the move z-score), regress next-day ALGO
return on the move z-signal over a trailing window -> slope beta. beta<0 = reversion (fade),
beta>0 = momentum (follow). The book only tilts toward trend to the degree the DATA says
momentum dominates reversion. Optional t-stat shrinkage keeps the reversion prior unless the
trend evidence is significant. Compare fade / ols / ols+shrink on known windows."""
import numpy as np, pandas as pd

prc_all = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
K, WZ, DOLLARS = 30, 60, 200_000


def rfit(X, Y, hl=500, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my


def z_series(lpA):
    """z-score of the K-day move at each day (causal, same construction as the current leg)."""
    L = len(lpA); z = np.full(L, np.nan)
    mv = np.full(L, np.nan)
    mv[K:] = lpA[K:] - lpA[:-K]
    for d in range(K + WZ, L):
        seg = mv[d - WZ + 1:d + 1]
        z[d] = (mv[d] - np.nanmean(seg)) / (np.nanstd(seg) + 1e-12)
    return z


def algo_signal(lpA, mode, W=250):
    """Return the (signed, unit-ish) ALGO signal. fade = -z. ols = beta*z from rolling reg."""
    L = len(lpA); rA = np.diff(lpA)
    z = z_series(lpA)
    zt = z[-1]
    if not np.isfinite(zt):
        return 0.0
    if mode == "fade":
        return -float(np.clip(zt, -3, 3))
    # rolling OLS: predict next-day return rA[d] from z[d]; pairs d in valid range, trailing W
    ds = np.arange(K + WZ, L - 1)                       # z[d] known, rA[d] = next-day return observed
    ds = ds[-W:]
    x = z[ds]; y = rA[ds]
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 30:
        return -float(np.clip(zt, -3, 3))
    xm = x - x.mean(); beta = (xm @ (y - y.mean())) / ((xm @ xm) + 1e-18)
    resid = (y - y.mean()) - beta * xm
    se = np.sqrt((resid @ resid) / max(len(x) - 2, 1) / ((xm @ xm) + 1e-18))
    tstat = beta / (se + 1e-18)
    if mode == "ols":
        eff = beta                                      # data-driven coefficient (can be <0 or >0)
    elif mode == "ols_shrink":
        # keep reversion prior; only let beta move the signal insofar as it's significant
        shrink = min(abs(tstat) / 2.0, 1.0)             # 0..1 by significance
        eff = beta * shrink + (-abs(beta)) * (1 - shrink)  # blend toward the reversion (neg) prior
    # normalize eff*z to a ~unit signal: divide by rolling std of beta*z over the window
    sig_series = beta * x
    scale = np.std(sig_series) + 1e-12
    return float(np.clip(eff * zt / scale, -3, 3))


def make(mode="fade"):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]
        sig = algo_signal(np.log(prc[0]), mode)
        rev = float(np.clip(sig * DOLLARS / prc[0, -1], -cap, cap))
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


print(f"{'ALGO-leg mode':28} {'S@250':>8} {'S@440':>8} {'H1':>7} {'H2':>7}")
for name, mode in [("fade (current)", "fade"), ("ols (data-driven beta)", "ols"),
                   ("ols_shrink (reversion prior)", "ols_shrink")]:
    print(f"{name:28} {run(lambda m=mode: make(m), nt-250, nt):8.1f} "
          f"{run(lambda m=mode: make(m), nt-440, nt):8.1f} "
          f"{run(lambda m=mode: make(m), 60, 280):7.0f} {run(lambda m=mode: make(m), 280, 500):7.0f}")
