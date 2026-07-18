"""qualifier_sweep.py — the real objective is now a 500-day window (days 1000-1500), then
1500-2000. Over windows that long, variance washes out and SCORE ~= true mean PnL. So rank
configs by MEAN over long windows (250d and the available 500d windows), watching the floor.
This is a different objective than SAFE (worst-short-window) was tuned for."""
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

def forecast(t, kind, blend):
    a = np.mean([ridge_z(t, hl) for hl in HLS], 0) if kind == "ens" else ridge_z(t, int(kind))
    return a if blend == 0 else (1 - blend) * a + blend * revz(t, 10)

def book(kind, blend, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            wz = forecast(t, kind, blend)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = logp[0, :t]; mv = lpA[30:] - lpA[:-30]
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
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1)

configs = [
    ("ens   b.30 (=SAFE)", "ens", 0.30), ("ens   b.20", "ens", 0.20),
    ("ens   b.15", "ens", 0.15), ("ens   b.10", "ens", 0.10), ("ens   b.00", "ens", 0.00),
    ("hl1000 b.15 (=SWING)", "1000", 0.15), ("hl1000 b.10", "1000", 0.10),
    ("hl1000 b.00", "1000", 0.00), ("hl2000 b.10", "2000", 0.10), ("hl500 b.15", "500", 0.15),
]

for L, step, lo in ((250, 20, 346), (500, 20, 500)):
    ends = list(range(lo, nDays + 1, step))
    print(f"\n=== {L}-day windows ({len(ends)} windows, ends {ends[0]}..{ends[-1]}) — rank by MEAN ===")
    print(f"{'config':<22}{'mean':>7}{'std':>7}{'min':>7}{'max':>7}")
    rows = []
    for label, kind, blend in configs:
        scs = np.array([book(kind, blend, e - L, e) for e in ends])
        rows.append((label, scs.mean(), scs.std(), scs.min(), scs.max()))
    for label, m, s, mn, mx in sorted(rows, key=lambda x: -x[1]):
        print(f"{label:<22}{m:7.0f}{s:7.0f}{mn:7.0f}{mx:7.0f}")
