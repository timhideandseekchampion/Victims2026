"""Is a robust ~600 reachable? The user reports the field clusters at 600. Search the most
plausible +15-20% levers — fuller capital deployment, different sizing, and the v2 pairs
approach standalone — validated on BOTH windows (250-500 AND 500-750). Anything >=600 on BOTH
is a real edge we were missing; if nothing clears it, our approach genuinely caps lower."""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1)


def ewls(X, Y, hl=500, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my


def make_ridge(conv=0.2, sizing="max", contra=200_000, hedge=True):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = ewls(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        wz = w/(np.std(w)+1e-12)
        keep = np.abs(wz) >= conv
        if sizing == "max":
            sh = np.sign(w)*(10000/prc[1:, -1])
        elif sizing == "convwt":                          # size ∝ |forecast|, top name hits $10k cap
            sh = np.clip(wz/np.max(np.abs(wz)+1e-12), -1, 1)*(10000/prc[1:, -1])
        elif sizing == "rank":                            # rank-proportional
            r = np.argsort(np.argsort(w)); rr = (r-(ni-2)/2)/((ni-1)/2)
            sh = rr*(10000/prc[1:, -1])
        pos[1:] = np.where(keep, sh, 0.0)
        cap = 100000/prc[0, -1]; rev = 0.0
        if contra > 0 and t > 92:
            lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]; z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
            rev = float(np.clip(-np.clip(z, -3, 3)*contra/prc[0, -1], -cap, cap))
        hs = 0.0
        if hedge:
            rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
            betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
            hs = -((pos[1:]*prc[1:, -1])@betas)/prc[0, -1]
        room = max(cap-abs(rev), 0.0); pos[0] = rev+float(np.clip(hs, -room, room))
        return pos.astype(int)
    return gp


def run(gp, S, E):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []; gross = []
    for t in range(S, E+1):
        p = prc[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < E else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > S: pll.append(pl); gross.append(np.abs(cp*cur).sum())
    return score(np.array(pll)), np.mean(gross)


print(f"{'config':44} {'250-500':>9} {'500-750':>9} {'gross$k':>8}")
tests = [
    ("baseline (conv0.2, max, contra200k, hedge)", dict()),
    ("trade ALL names (conv0.0)", dict(conv=0.0)),
    ("conv0.1", dict(conv=0.1)),
    ("conviction-weighted sizing", dict(sizing="convwt")),
    ("conviction-weighted, conv0.0", dict(sizing="convwt", conv=0.0)),
    ("rank sizing", dict(sizing="rank")),
    ("no ALGO leg (idio only)", dict(contra=0)),
    ("no ALGO leg, conv0.0", dict(contra=0, conv=0.0)),
]
for name, kw in tests:
    so, _ = run(make_ridge(**kw), 250, 500)
    sn, g = run(make_ridge(**kw), nt-250, nt)
    flag = "  <-- >=600 BOTH" if (so >= 600 and sn >= 600) else ""
    print(f"{name:44} {so:9.0f} {sn:9.0f} {g/1000:8.0f}{flag}")
