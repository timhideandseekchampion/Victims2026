"""
sensitivity.py — are RIDGE_A=0.1 and BLEND (0.15 SWING / 0.30 SAFE) robust plateau choices or
fragile overfit peaks? Sweep each across windows and show the score surface. A trustworthy knob
sits on a FLAT plateau (neighbors score similarly); an overfit knob is a sharp spike. Verified
book engine (hedge off, sign-size, matches eval.py).
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp_all = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

_rc = {}
def ridge_z(t, hl, a):
    key = (t, hl, a)
    if key in _rc: return _rc[key]
    lp = lp_all[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(51), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f = f - f.mean(); v = f / (f.std() + 1e-12); _rc[key] = v; return v

def forecast(t, hl, a, blend, ens=False):
    core = np.mean([ridge_z(t, h, a) for h in (250, 500, 1000, 2000)], 0) if ens else ridge_z(t, hl, a)
    rr = lp_all[1:, t - 1] - lp_all[1:, t - 11]; rr = rr - rr.mean()
    return (1 - blend) * core + blend * (-rr / (rr.std() + 1e-12))

def book(hl, a, blend, ens, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 130:
            wz = forecast(t, hl, a, blend, ens)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = lp_all[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            pos[0] = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-9: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1)

legs = [(250, 500), (350, 600), (450, 700), (500, 750)]
def profile(hl, a, blend, ens):
    scs = [book(hl, a, blend, ens, S, E) for S, E in legs]
    return np.mean(scs), min(scs), scs[-1]

print("=== RIDGE_A sweep (SWING core: hl1000, blend0.15, hedge off) ===")
print(f"{'RIDGE_A':>8}{'mean':>8}{'min':>7}{'500-750':>9}")
for a in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    m, mn, last = profile(1000, a, 0.15, False)
    star = "  <- ship" if a == 0.1 else ""
    print(f"{a:>8}{m:8.0f}{mn:7.0f}{last:9.0f}{star}")

print("\n=== BLEND sweep — SWING core (hl1000, a0.1) vs SAFE core (HL-ens, a0.1), hedge off ===")
print(f"{'BLEND':>7}{'SWING mean':>12}{'min':>7}{'500-750':>9}   {'SAFE mean':>10}{'min':>7}{'500-750':>9}")
for b in (0.0, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50):
    sm, smn, sl = profile(1000, 0.1, b, False)
    em, emn, el = profile(None, 0.1, b, True)
    tag = "  <- SWING" if b == 0.15 else ("  <- SAFE" if b == 0.30 else "")
    print(f"{b:>7}{sm:12.0f}{smn:7.0f}{sl:9.0f}   {em:10.0f}{emn:7.0f}{el:9.0f}{tag}")
