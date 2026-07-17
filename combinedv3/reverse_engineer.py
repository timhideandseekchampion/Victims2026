"""Reverse-engineer what scored high on the now-revealed hidden window (days 500-750).
Score archetypal strategies to decompose WHAT the window rewarded: market direction (beta),
stronger idio edge, more deployment, momentum vs reversion. Distinguish durable edge from
window-specific luck."""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
S, E = nt - 250, nt                                    # the scored hidden window: days 500-750


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu, 0.0
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1), sr


def run(gp):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(S, E+1):
        p = prc[:, :t]; cur = p[:, -1]
        npos = np.clip(gp(p), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < E else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    return score(np.array(pll))


def rfit(X, Y, hl=500, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my

_c = {"t": None, "m": None}
def ridge_book(prc, demean=True, conv=0.2, hedge=True, contra=200_000, longtilt=0.0):
    ni, t = prc.shape; pos = np.zeros(ni)
    if t < 95: return pos.astype(int)
    lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
    if _c["t"] != t: _c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); _c["t"] = t
    B, mx, my = _c["m"]; pred = my+(ret[:, -1]-mx)@B
    w = (pred-pred.mean()) if demean else pred
    s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= conv*(np.std(w)+1e-12), s, 0.0)
    if longtilt != 0:
        pos[1:] += longtilt*10000/prc[1:, -1]           # add net LONG exposure to every name
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


def reset(): _c["t"] = None; _c["m"] = None


# --- context: what did the market do over 500-750? ---
lpA = np.log(prc[0]); algo_ret = lpA[S] , lpA[E-1]
algo_tot = (lpA[E-1]-lpA[S]) * 100
names_tot = (np.log(prc[1:, E-1]) - np.log(prc[1:, S])).mean() * 100
disp_old = np.diff(np.log(prc[1:, 250:500]), axis=1).std(0).mean()
disp_new = np.diff(np.log(prc[1:, S:E]), axis=1).std(0).mean()
print(f"HIDDEN WINDOW 500-750 CONTEXT:")
print(f"  ALGO index total return: {algo_tot:+.1f}%   avg name total return: {names_tot:+.1f}%")
print(f"  cross-sec dispersion: old(250-500) {disp_old:.4f}  vs  new(500-750) {disp_new:.4f}")
print()

def long_only(prc):
    cur = prc[:, -1]; pos = np.zeros(nInst); pos[1:] = 10000/cur[1:]; pos[0] = 100000/cur[0]; return pos.astype(int)
def short_only(prc):
    return -long_only(prc)

print(f"{'archetype (scored on 500-750)':44} {'Score':>7} {'Sharpe':>7}")
tests = [
    ("our primary (neutral, hedged) [=503 anchor]", lambda: (reset(), ridge_book)[1]),
    ("LONG-only (full beta, buy everything)", lambda: long_only),
    ("SHORT-only (full negative beta)", lambda: short_only),
    ("neutral + LONG tilt 0.5", lambda: (reset(), lambda p: ridge_book(p, longtilt=0.5))[1]),
    ("neutral + LONG tilt 1.0", lambda: (reset(), lambda p: ridge_book(p, longtilt=1.0))[1]),
    ("neutral + SHORT tilt 0.5", lambda: (reset(), lambda p: ridge_book(p, longtilt=-0.5))[1]),
    ("max deploy (conv0.1,nohedge,contra1M)", lambda: (reset(), lambda p: ridge_book(p, conv=0.1, hedge=False, contra=1_000_000))[1]),
    ("no ALGO leg (idio ridge only)", lambda: (reset(), lambda p: ridge_book(p, contra=0))[1]),
    ("raw directional (demean off)", lambda: (reset(), lambda p: ridge_book(p, demean=False))[1]),
]
for name, mk in tests:
    gp = mk(); sc, sh = run(gp)
    print(f"{name:44} {sc:7.0f} {sh:7.1f}")
