"""contra_sweep.py — does the ALGO fade notional (CONTRA_DOL) help or just add variance?
Sweeps CONTRA_DOL on the long windows (250d/500d) holding the stock book fixed (ens, b.20 & b.30).
If the fade has ~0 alpha, less ALGO deployment should raise Sharpe and score."""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)
HLS = [250, 500, 1000, 2000]

_rc = {}
def ridge_z(t, hl, a=0.1):
    key = (t, hl)
    if key in _rc: return _rc[key]
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(nInst), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f = f - f.mean()
    v = f / (f.std() + 1e-12); _rc[key] = v; return v

def revz(t, w):
    rr = logp[1:, t - 1] - logp[1:, t - 1 - w]; rr = rr - rr.mean()
    return -rr / (rr.std() + 1e-12)

def book(blend, contra, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            a = np.mean([ridge_z(t, hl) for hl in HLS], 0)
            wz = a if blend == 0 else (1 - blend) * a + blend * revz(t, 10)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            if contra > 0:
                lpA = logp[0, :t]; mv = lpA[30:] - lpA[:-30]
                z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
                pos[0] = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (contra / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1)

contras = [0, 250_000, 500_000, 1_000_000, 2_000_000]
for blend in (0.20, 0.30):
    print(f"\n########## stock book = ens, BLEND={blend} ##########")
    for L, lo in ((250, 346), (500, 500)):
        ends = list(range(lo, nDays + 1, 20))
        print(f"  -- {L}-day windows ({len(ends)}) --   {'mean':>7}{'std':>7}{'min':>7}{'500-750':>9}")
        for c in contras:
            scs = np.array([book(blend, c, e - L, e) for e in ends])
            leg = book(blend, c, 500, 750)
            print(f"    CONTRA_DOL={c:>9,}          {scs.mean():7.0f}{scs.std():7.0f}{scs.min():7.0f}{leg:9.0f}")
