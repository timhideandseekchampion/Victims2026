"""Test the user's hypothesis: do LOW-score windows coincide with (a) low cross-sectional
dispersion ('not choppy') and (b) a TRENDING ALGO index? Run the book once, then for many
rolling windows correlate window-score with window-dispersion and window-ALGO-trend."""
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


def gp(prc, c):
    ni, t = prc.shape; pos = np.zeros(ni)
    lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
    if c["t"] != t: c["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); c["t"] = t
    B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
    s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
    cap = 100000/prc[0, -1]; lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]
    z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev = float(np.clip(-np.clip(z, -3, 3)*200000/prc[0, -1], -cap, cap))
    rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
    betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
    net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
    pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)


# one run -> daily book PnL + daily dispersion + daily ALGO log-return
c = {"t": None, "m": None}; cash = 0; cp = np.zeros(nInst); val = 0; cm = 0
PL = []; DISP = []; ALGOR = []
for t in range(nt-440, nt+1):
    p = prc_all[:, :t]; cur = p[:, -1]
    npos = np.clip(gp(p, c), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < nt else cp.copy()
    d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
    pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
    if t > nt-440:
        PL.append(pl)
        r = np.log(prc_all[:, t-1]/prc_all[:, t-2])            # that day's returns
        DISP.append(r[1:].std()); ALGOR.append(r[0])
PL = np.array(PL); DISP = np.array(DISP); ALGOR = np.array(ALGOR)


def score(x):
    mu, sd = x.mean(), x.std()
    if mu <= 0 or sd < 1e-10: return mu
    s = np.sqrt(250)*mu/sd; return mu*s**2/(s**2+1)


# rolling windows: window score vs window dispersion & window ALGO trendiness
W, step = 100, 5
rows = []
for a in range(0, len(PL)-W, step):
    sl = slice(a, a+W)
    sc = score(PL[sl])
    disp = DISP[sl].mean()                                     # avg choppiness
    ar = ALGOR[sl]
    trend = abs(ar.sum())/(ar.std()*np.sqrt(W)+1e-12)          # |drift|/RW-scale: high=trending
    rows.append((sc, disp, trend))
rows = np.array(rows)
SC, D, TR = rows[:, 0], rows[:, 1], rows[:, 2]
print(f"{len(rows)} rolling {W}-day windows. Window-score correlations:\n")
print(f"  corr(score, cross-sec DISPERSION) = {np.corrcoef(SC, D)[0,1]:+.2f}   "
      f"(+ => choppier = higher score, supports hypothesis)")
print(f"  corr(score, ALGO TREND strength)  = {np.corrcoef(SC, TR)[0,1]:+.2f}   "
      f"(- => more trending = lower score, supports hypothesis)")
print()
# tercile table
def terc(x, lab):
    lo = SC[x <= np.percentile(x, 33)].mean(); hi = SC[x >= np.percentile(x, 67)].mean()
    print(f"  {lab:22} low-tercile score {lo:5.0f}   high-tercile score {hi:5.0f}")
terc(D, "by DISPERSION")
terc(TR, "by ALGO TREND")
