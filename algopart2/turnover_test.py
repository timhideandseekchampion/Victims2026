"""turnover_test.py — does a no-trade BAND (hysteresis) beat pure sign()? Keeps FULL gross
($10k/name) but only flips a name's sign when |signal| clears a band, so near-zero coin-flip
churn is held instead of paid for. Reports score, floor, and total $-volume (turnover)."""
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

def book(blend, band, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []; dvol = 0.0
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            a = np.mean([ridge_z(t, hl) for hl in HLS], 0)
            wz = a if blend == 0 else (1 - blend) * a + blend * revz(t, 10)
            desired = np.sign(wz) * (dlr[1:] / cur[1:])
            if band > 0:
                # hysteresis: keep yesterday's position on a name unless |signal| clears the band
                prev = cp[1:]
                flip = np.sign(wz) != np.sign(prev)
                hold = flip & (np.abs(wz) < band) & (prev != 0)
                desired = np.where(hold, prev, desired)
            pos[1:] = desired
            cap = dlr[0] / cur[0]
            lpA = logp[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            pos[0] = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        dv = cur * np.abs(dp); dvol += dv.sum(); comm = np.sum(dv * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sc = mu if (mu <= 0 or sd < 1e-10) else mu * (np.sqrt(250)*mu/sd)**2 / ((np.sqrt(250)*mu/sd)**2 + 1)
    return sc, dvol

bands = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
for blend in (0.20, 0.30):
    print(f"\n########## ens, BLEND={blend} ##########")
    for L, lo in ((250, 346), (500, 500)):
        ends = list(range(lo, nDays + 1, 20))
        print(f"  -- {L}-day windows ({len(ends)}) --  {'band':>6}{'mean':>7}{'floor':>7}{'turnover/day':>14}")
        base_turn = None
        for b in bands:
            res = [book(blend, b, e - L, e) for e in ends]
            scs = np.array([r[0] for r in res]); turn = np.mean([r[1] for r in res]) / L
            if base_turn is None: base_turn = turn
            tag = "" if b == 0 else f"  ({turn/base_turn*100:.0f}% of base)"
            print(f"                                  {b:6.2f}{scs.mean():7.0f}{scs.min():7.0f}{turn:14,.0f}{tag}")
