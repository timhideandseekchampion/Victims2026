"""finalize.py — validate candidate configs across EVERY 250-day window (robustness test)
and print the shippable choice. Reuses push700's engine."""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
logp = np.log(prc)

def ewls(X, Y, hl, a=0.1):
    n, p = X.shape; lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY), mx, my

_fc = {}
def forecast(t, hl):
    key = (t, hl)
    if key in _fc: return _fc[key]
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    B, mx, my = ewls(r[:, :-1].T, r[1:, 1:].T, hl)
    pred = my + (r[:, -1] - mx) @ B
    v = pred - pred.mean(); _fc[key] = v; return v

_rev = {}
def revsig(t, w):
    key = (t, w)
    if key in _rev: return _rev[key]
    lp = logp[1:, :t]; rr = lp[:, -1] - lp[:, -1 - w]; rr = rr - rr.mean()
    v = -rr / (rr.std() + 1e-12); _rev[key] = v; return v

def algo_fade(t, k=30, wz=60):
    lpA = logp[0, :t]; mv = lpA[k:] - lpA[:-k]
    z = (mv[-1] - mv[-wz:].mean()) / (mv[-wz:].std() + 1e-12)
    return -np.clip(z, -3, 3) / 3.0

def score_cfg(cfg, S, E):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(S, E + 1):
        cur = prc[:, t - 1]
        if t < E and t >= 96:
            pos = np.zeros(nInst)
            f = forecast(t, cfg["hl"]); wz = f / (f.std() + 1e-12)
            if cfg["blend"] > 0:
                wz = (1 - cfg["blend"]) * wz + cfg["blend"] * revsig(t, cfg["revw"])
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            av = float(np.clip(algo_fade(t) * (cfg["contra"] / cur[0]), -cap, cap))
            hs = 0.0
            if cfg["hedge"]:
                r = logp[:, 1:t] - logp[:, :t - 1]
                rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
                betas = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
                hs = -((pos[1:] * cur[1:]) @ betas) / cur[0]
            room = max(cap - abs(av), 0.0); pos[0] = av + float(np.clip(hs, -room, room))
            lim = (dlr / cur).astype(int); newPos = np.clip(pos, -lim, lim).astype(int)
        else:
            newPos = cp.copy()
        d = newPos - cp; cash -= cur.dot(d) + comm
        dv = cur * np.abs(d); comm = np.sum(dv * commRate); cp = newPos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1)

candidates = {
    "A max-leg (contra1M, hedge)": dict(hl=500, blend=0.3, revw=10, contra=1_000_000, hedge=True),
    "B plateau  (contra200k, hedge)": dict(hl=500, blend=0.3, revw=10, contra=200_000, hedge=True),
    "C idio-only (contra0)": dict(hl=500, blend=0.3, revw=10, contra=0, hedge=True),
}
# every 250-day window that has full warmup (start>=96): end from 350..750 step 25
ends = list(range(350, 751, 25))
print(f"{'config':<32}" + "".join(f"{e-250}-{e:>7}"[:8] for e in ends) + "   mean   min")
print("windows ->  " + " ".join(f"{e-250}-{e}" for e in ends[:1]) + " ... (250-day windows)")
rows = {}
for name, cfg in candidates.items():
    scs = [score_cfg(cfg, e - 250, e) for e in ends]
    rows[name] = scs
    print(f"{name:<32}" + "".join(f"{s:8.0f}" for s in scs) + f" {np.mean(scs):6.0f} {np.min(scs):6.0f}")

print("\nKey windows for the shippable pick:")
for name, cfg in candidates.items():
    s57 = score_cfg(cfg, 500, 750); s45 = score_cfg(cfg, 400, 500)
    print(f"  {name:<32} 500-750={s57:6.0f}  400-500(100d)={s45:6.0f}  mean-across-all={np.mean(rows[name]):6.0f}  worst={np.min(rows[name]):6.0f}")
