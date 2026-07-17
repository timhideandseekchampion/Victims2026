"""Can we extract a HIGHER-IC cross-sectional forecast than the baseline ridge? That's the only
durable path from ~505 toward 700+. Test principled variants (reduced-rank regression, VAR(2),
alpha/half-life) on the idio-only book, scored on BOTH windows (250-500 AND 500-750). A variant
must beat baseline on BOTH to count as a real edge, not a fit to the revealed window."""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)


def ewls(X, Y, hl, a):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    B = np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY)
    return B, mx, my


def make(mode="base", hl=500, a=0.1, rank=None, nlag=1):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if nlag == 1:
            X = ret[:, :-1].T; Y = ret[1:, 1:].T; xin = ret[:, -1]
        else:                                              # VAR(2): stack lag-1 and lag-2
            X = np.hstack([ret[:, 1:-1].T, ret[:, :-2].T]); Y = ret[1:, 2:].T
            xin = np.concatenate([ret[:, -1], ret[:, -2]])
        if c["t"] != t:
            B, mx, my = ewls(X, Y, hl, a)
            if rank is not None:                           # reduced-rank truncation of B
                U, s, Vt = np.linalg.svd(B, full_matrices=False)
                B = (U[:, :rank]*s[:rank]) @ Vt[:rank]
            c["m"] = (B, mx, my)
        B, mx, my = c["m"]; pred = my+(xin-mx)@B
        w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        # beta-hedge with ALGO (no contra leg — it was dead weight OOS)
        cap = 100000/prc[0, -1]; rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        pos[0] = float(np.clip(-((pos[1:]*prc[1:, -1])@betas)/prc[0, -1], -cap, cap))
        return pos.astype(int)
    return gp


def run(gp, S, E):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(S, E+1):
        p = prc[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < E else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    return score(np.array(pll))


print(f"{'forecast variant (idio-only book)':40} {'250-500':>9} {'500-750':>9}  {'both?':>6}")
variants = [
    ("baseline ridge HL500 a0.1", dict()),
    ("reduced-rank r=3", dict(rank=3)),
    ("reduced-rank r=5", dict(rank=5)),
    ("reduced-rank r=8", dict(rank=8)),
    ("reduced-rank r=15", dict(rank=15)),
    ("alpha=0.03", dict(a=0.03)),
    ("alpha=0.5", dict(a=0.5)),
    ("alpha=2.0", dict(a=2.0)),
    ("half-life=250", dict(hl=250)),
    ("half-life=1000", dict(hl=1000)),
    ("VAR(2)", dict(nlag=2)),
    ("VAR(2)+rank5", dict(nlag=2, rank=5)),
]
base_old = run(make(), 250, 500); base_new = run(make(), nt-250, nt)
for name, kw in variants:
    so = run(make(**kw), 250, 500); sn = run(make(**kw), nt-250, nt)
    both = "YES" if (so >= base_old-2 and sn >= base_new-2) else ""
    print(f"{name:40} {so:9.0f} {sn:9.0f}  {both:>6}")
