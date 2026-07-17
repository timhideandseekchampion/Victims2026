"""Test the Two-Sigma-paper idea on OUR data: is the book's edge REGIME-conditional?
I.e. does an observable state (known at decision time) predict whether the next day's book
PnL is strong or weak? If yes -> regime-conditional sizing can add. If no -> dead (as the
constant-vol / no-regime findings predict). Honest test: correlation + quartile split + a
conditional-sizing backtest on BOTH halves."""
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


def gp_and_state(prc, cache):
    """Return (positions, state_dict) — state computed ONLY from prc (causal)."""
    ni, t = prc.shape; pos = np.zeros(ni)
    lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
    if cache["t"] != t: cache["m"] = rfit(ret[:, :-1].T, ret[1:, 1:].T); cache["t"] = t
    B, mx, my = cache["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
    s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
    cap = 100000/prc[0, -1]; rev = 0.0
    lpA = np.log(prc[0]); mv = lpA[30:]-lpA[:-30]
    algoz = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12)
    rev = float(np.clip(-np.clip(algoz, -3, 3)*200000/prc[0, -1], -cap, cap))
    rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
    betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
    net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
    pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room))
    state = dict(
        xs_disp=ret[1:, -1].std(),                 # today's cross-sectional return dispersion
        xs_disp20=ret[1:, -20:].std(),             # 20d avg dispersion
        algo_vol=ret[0, -20:].std(),               # ALGO 20d realized vol
        fc_disp=np.std(w),                          # ridge forecast dispersion (conviction breadth)
        algo_absz=abs(algoz),                       # ALGO contra signal strength
    )
    return pos.astype(int), state


# collect daily book PnL aligned with the state that SET that day's position
cache = {"t": None, "m": None}
cash = 0; cp = np.zeros(nInst); val = 0; cm = 0
rows = []
prev_state = None
for t in range(nt-440, nt+1):
    p = prc_all[:, :t]; cur = p[:, -1]
    if t < nt:
        npos, st = gp_and_state(p, cache)
        npos = np.clip(npos, -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int)
    else:
        npos, st = cp.copy(), None
    d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
    pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
    if t > nt-440 and prev_state is not None:
        rows.append((prev_state, pl))              # pl realized from positions set with prev_state
    prev_state = st
states = list(rows[0][0].keys())
S = {k: np.array([r[0][k] for r in rows]) for k in states}
PL = np.array([r[1] for r in rows])
print(f"n days = {len(PL)}, mean daily PnL = {PL.mean():.0f}\n")
print(f"{'state (causal)':12} {'corr w/ next PnL':>16} {'low-tercile PnL':>16} {'high-tercile PnL':>17}")
for k in states:
    x = S[k]; c = np.corrcoef(x, PL)[0, 1]
    lo = PL[x <= np.percentile(x, 33)].mean(); hi = PL[x >= np.percentile(x, 67)].mean()
    print(f"{k:12} {c:16.3f} {lo:16.0f} {hi:17.0f}")
print("\n(if corr ~0 and low~high tercile PnL, the edge is NOT regime-conditional -> no timing to exploit)")
