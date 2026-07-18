"""fwd_window.py — how do SAFE vs SWING behave on SHORT growing windows (the live-grading regime)?
Scores each config on rolling windows of length L across all available history (days 0-750),
and prints the distribution. The live grade is a fixed-start(750), growing window, so short-window
variance here is the best proxy we have for what days 750-800 / 750-850 look like."""
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
    return (1 - blend) * a + blend * revz(t, 10)

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
            av = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            pos[0] = av
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

configs = [("SAFE  (ens, b.30)", "ens", 0.30), ("SWING (hl1000, b.15)", "1000", 0.15),
           ("var   (ens, b.45)", "ens", 0.45)]

for L in (50, 100, 250):
    # rolling windows ending every 10 days, need start>=96 for warmup
    ends = list(range(max(96 + L, 200), nDays + 1, 10))
    print(f"\n=== window length {L}d  ({len(ends)} rolling windows, ends {ends[0]}..{ends[-1]}) ===")
    print(f"{'config':<22}{'mean':>7}{'std':>7}{'min':>7}{'p25':>7}{'max':>7}{'>1260':>7}")
    for label, kind, blend in configs:
        scs = np.array([book(kind, blend, e - L, e) for e in ends])
        pct = 100.0 * np.mean(scs >= 1260)
        print(f"{label:<22}{scs.mean():7.0f}{scs.std():7.0f}{scs.min():7.0f}"
              f"{np.percentile(scs,25):7.0f}{scs.max():7.0f}{pct:6.0f}%")
