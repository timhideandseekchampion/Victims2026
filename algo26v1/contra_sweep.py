"""Sweep the ALGO-contrarian knobs CONTRA_K x CONTRA_WZ on the HL=500 book.

The ALGO leg trades ONE time series (the index) -> only ~1 path of reversion timing exists,
so tuning 2 params on it overfits easily. Judge by BOTH-HALVES stability, not peak @250.
Also decompose: score with the ALGO leg ON vs OFF (CONTRA_DOLLARS=0) = the leg's contribution.
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


def make(K=30, WZ=60, contra=200_000):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]; rev = 0.0
        if contra > 0 and t > K+WZ+2:
            lpA = np.log(prc[0]); mv = lpA[K:]-lpA[:-K]; z = (mv[-1]-mv[-WZ:].mean())/(mv[-WZ:].std()+1e-12)
            rev = float(np.clip(-np.clip(z, -3, 3)*contra/prc[0, -1], -cap, cap))
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


# --- decomposition: ALGO leg ON vs OFF ---
print("=== how much does the ALGO leg contribute? ===")
for lbl, contra in [("ALGO leg OFF (idio book only)", 0), ("ALGO leg ON (K30/WZ60, $200k)", 200_000)]:
    s250 = run(lambda contra=contra: make(contra=contra), nt-250, nt)
    s440 = run(lambda contra=contra: make(contra=contra), nt-440, nt)
    h1 = run(lambda contra=contra: make(contra=contra), 60, 280)
    h2 = run(lambda contra=contra: make(contra=contra), 280, 500)
    print(f"  {lbl:32} S@250 {s250:6.0f}  S@440 {s440:6.0f}  H1 {h1:5.0f}  H2 {h2:5.0f}")

# --- K x WZ sweep, with both-halves stability ---
print("\n=== CONTRA_K x CONTRA_WZ sweep (S@250 / H1 / H2) — want H1 & H2 BOTH strong ===")
Ks = [5, 10, 20, 30, 40]; WZs = [20, 40, 60, 90]
print(f"{'K\\WZ':>6}" + "".join(f"{wz:>18}" for wz in WZs))
for K in Ks:
    row = f"{K:>6}"
    for WZ in WZs:
        s = run(lambda K=K, WZ=WZ: make(K, WZ), nt-250, nt)
        h1 = run(lambda K=K, WZ=WZ: make(K, WZ), 60, 280)
        h2 = run(lambda K=K, WZ=WZ: make(K, WZ), 280, 500)
        row += f"  {s:4.0f}/{h1:4.0f}/{h2:4.0f}"
    print(row)
