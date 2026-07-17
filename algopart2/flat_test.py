"""
flat_test.py — since B is STATIONARY (q̂=0), does EQUAL-weighting all history beat the ships'
forgetting ridge? Compare per-window IC (consistency) and book Score:
  ENS  = SAFE core: mean of forgetting-ridge over hl 250/500/1000/2000
  FLAT = ridge on ALL history, equal weights (the stationary-optimal estimator)
  FLAT500 = ridge on last 500 days, equal weights
"""
import numpy as np, pandas as pd
from scipy import stats
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp_all = np.log(prc); RET = lp_all[:, 1:] - lp_all[:, :-1]
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 1e4); dlr[0] = 1e5
_c = {}
def rz(t, hl, a=0.1):
    k = (t, hl)
    if k in _c: return _c[k]
    r = lp_all[:, :t]; r = r[:, 1:] - r[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]
    w = np.ones(n) if hl == 0 else 0.5 ** (np.arange(n - 1, -1, -1) / hl)
    if hl > 0 and n > hl * 8: pass
    sw = w.sum(); mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(51), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f -= f.mean(); v = f / (f.std() + 1e-12); _c[k] = v; return v
def rz_flat(t, L, a=0.1):
    k = (t, -L)
    if k in _c: return _c[k]
    r = lp_all[:, :t]; r = r[:, 1:] - r[:, :-1]
    if L and r.shape[1] > L + 1: r = r[:, -(L + 1):]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    mx = X.mean(0); my = Y.mean(0); Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ Xc + a * np.eye(51), Xc.T @ Yc)
    f = my + (xin - mx) @ B; f -= f.mean(); v = f / (f.std() + 1e-12); _c[k] = v; return v

FCS = {
    "ENS (forget hl250-2000)": lambda t: np.mean([rz(t, hl) for hl in (250, 500, 1000, 2000)], 0),
    "FLAT (all history, equal)": lambda t: rz_flat(t, 0),
    "FLAT500 (last 500 equal)": lambda t: rz_flat(t, 500),
}
def ic_win(fc, S, E):
    ics = []
    for t in range(max(S, 130), min(E, nt - 1)):
        s = fc(t); fwd = RET[1:, t - 1]
        if s.std() > 1e-12 and fwd.std() > 1e-12: ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics); return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
legs = [(250, 500), (350, 600), (450, 700), (500, 750)]
print("Per-window IC (t):")
print(f"{'forecast':<26}" + "".join(f"{f'{a}-{b}':>14}" for a, b in legs) + f"{'mean':>8}")
for nm, fc in FCS.items():
    row = [ic_win(fc, S, E) for S, E in legs]
    print(f"{nm:<26}" + "".join(f"{ic:8.4f}({t:4.1f})" for ic, t in row) + f"{np.mean([r[0] for r in row]):8.4f}")

def book(fc, Sd, Ed, blend=0.3):
    cash = 0.; cp = np.zeros(nInst); val = 0.; cm = 0.; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 130:
            core = fc(t); rr = lp_all[1:, t - 1] - lp_all[1:, t - 11]; rr -= rr.mean()
            wz = (1 - blend) * core + blend * (-rr / (rr.std() + 1e-12))
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]; lpA = lp_all[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            pos[0] = float(np.clip(-np.clip(z, -3, 3) / 3. * (1e6 / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else: pos = cp.copy()
        d = pos - cp; cash -= cur.dot(d) + cm; cm = np.sum(cur * np.abs(d) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - val; val = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    return mu * (250 * mu**2 / sd**2) / (250 * mu**2 / sd**2 + 1) if mu > 0 and sd > 1e-9 else mu
print("\nBook Score by leg (blend 0.3, hedge off = SAFE):")
print(f"{'forecast':<26}" + "".join(f"{f'{a}-{b}':>10}" for a, b in legs) + f"{'mean':>8}")
for nm, fc in FCS.items():
    row = [book(fc, S, E) for S, E in legs]
    print(f"{nm:<26}" + "".join(f"{s:10.0f}" for s in row) + f"{np.mean(row):8.0f}")
