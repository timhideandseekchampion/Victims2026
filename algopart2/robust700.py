"""
robust700.py — (1) dissect the 1054-on-400-500 config, (2) search for a config that
COMFORTABLY clears 700 (rank by the across-window distribution, not one lucky window),
(3) compare the finalists across ALL 250-day legs spanning days 0-750, incl. the early data.

Precomputes forecasts / reversion / betas / ALGO signals ONCE per day so hundreds of
configs x a dozen legs stay fast and causal. eval.py-faithful scoring.
"""
import itertools
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)

HLS = [250, 500, 1000, 2000]
REVWS = [5, 10, 20]

def ewls(X, Y, hl, a=0.1):
    n, p = X.shape; lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY), mx, my

# ---- precompute per-day pieces for t in [96, nDays) --------------------------------
print("precomputing per-day signals ...")
FC = {hl: {} for hl in HLS}; RV = {w: {} for w in REVWS}
BETA = {}; AF = {}; AOLS = {}
for t in range(96, nDays + 1):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    for hl in HLS:
        B, mx, my = ewls(r[:, :-1].T, r[1:, 1:].T, hl)
        pred = my + (r[:, -1] - mx) @ B
        f = pred - pred.mean(); FC[hl][t] = f / (f.std() + 1e-12)
    for w in REVWS:
        rr = lp[1:, -1] - lp[1:, -1 - w]; rr = rr - rr.mean()
        RV[w][t] = -rr / (rr.std() + 1e-12)
    rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
    BETA[t] = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
    lpA = logp[0, :t]; mv = lpA[30:] - lpA[:-30]
    z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
    AF[t] = -np.clip(z, -3, 3) / 3.0
    m = mv[-250:]; zz = (m - m.mean()) / (m.std() + 1e-12)
    fr = (lpA[1:] - lpA[:-1])[-len(zz):]
    b = np.cov(zz[:-1], fr[1:])[0, 1] / (np.var(zz[:-1]) + 1e-12) if len(zz) > 5 else 0.0
    AOLS[t] = float(np.clip(b * z, -1, 1))
print("done.\n")

def score_cfg(cfg, S, E):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    hl, blend, revw, contra, mode, hedge = (cfg["hl"], cfg["blend"], cfg["revw"],
                                            cfg["contra"], cfg["mode"], cfg["hedge"])
    for t in range(S, E + 1):
        cur = prc[:, t - 1]
        if t < E and t >= 96:
            pos = np.zeros(nInst)
            wz = FC[hl][t]
            if blend > 0: wz = (1 - blend) * wz + blend * RV[revw][t]
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            aval = AF[t] if mode == "fade" else (AOLS[t] if mode == "ols" else 0.0)
            av = float(np.clip(aval * (contra / cur[0]), -cap, cap))
            hs = -((pos[1:] * cur[1:]) @ BETA[t]) / cur[0] if hedge else 0.0
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

# legs: proper-warmup 250-day windows spanning the file, S>=250 (grader-representative)
LEGS = [(S, S + 250) for S in range(250, 501, 50)]         # 250-500 ... 500-750
EARLY = [(S, S + 250) for S in [96, 150, 200]]             # early low-warmup legs
FULL = (96, nDays)

# ---- (1) the 1054 config --------------------------------------------------------
c1054 = dict(hl=1000, blend=0.15, revw=10, contra=1_000_000, mode="fade", hedge=False)
print("=" * 90)
print("(1) THE 1054 CONFIG  {hl1000, blend0.15, contra1M, fade, hedge=False}")
print("=" * 90)
scs = [score_cfg(c1054, s, e) for s, e in LEGS]
print("  proper-warmup legs (250d): " + "  ".join(f"{s}-{e}:{score_cfg(c1054,s,e):.0f}" for s, e in LEGS))
print(f"  mean {np.mean(scs):.0f}  min {np.min(scs):.0f}  max {np.max(scs):.0f}   #>=700: {sum(x>=700 for x in scs)}/{len(scs)}")
print(f"  full 0-750 backtest: {score_cfg(c1054, *FULL):.0f}   400-500(100d): {score_cfg(c1054,400,500):.0f}")

# ---- (2) search for a COMFORTABLE-700 config ------------------------------------
print("\n" + "=" * 90)
print("(2) SEARCH for a config that COMFORTABLY clears 700 (ranked by across-leg distribution)")
print("=" * 90)
grid = dict(hl=HLS, blend=[0.15, 0.3, 0.45], revw=REVWS,
            contra=[0, 200_000, 500_000, 1_000_000], mode=["fade", "ols", "off"], hedge=[True, False])
keys = list(grid); seen = set(); configs = []
for c in itertools.product(*[grid[k] for k in keys]):
    cfg = dict(zip(keys, c))
    if cfg["mode"] == "off": cfg["contra"] = 0
    if cfg["blend"] == 0: cfg["revw"] = 10
    k = tuple(sorted(cfg.items()))
    if k in seen: continue
    seen.add(k); configs.append(cfg)

ranked = []
for cfg in configs:
    scs = [score_cfg(cfg, s, e) for s, e in LEGS]
    ranked.append((np.mean(scs), np.min(scs), sum(x >= 700 for x in scs), scs, cfg))
print(f"searched {len(configs)} configs over {len(LEGS)} proper-warmup legs.")
print("\nTOP 10 by MEAN across legs:")
print(f"{'mean':>6}{'min':>6}{'>=700':>7}  {'500-750':>8}  config")
for mean, mn, n700, scs, cfg in sorted(ranked, key=lambda x: -x[0])[:10]:
    cc = {k: cfg[k] for k in ("hl", "blend", "revw", "contra", "mode", "hedge")}
    print(f"{mean:6.0f}{mn:6.0f}{n700:5d}/{len(LEGS)}  {scs[-1]:8.0f}  {cc}")
print("\nTOP 10 by MIN leg (robustness floor):")
for mean, mn, n700, scs, cfg in sorted(ranked, key=lambda x: -x[1])[:10]:
    cc = {k: cfg[k] for k in ("hl", "blend", "revw", "contra", "mode", "hedge")}
    print(f"{mean:6.0f}{mn:6.0f}{n700:5d}/{len(LEGS)}  {scs[-1]:8.0f}  {cc}")

# ---- (3) finalists compared across ALL 0-750 legs -------------------------------
best_mean = max(ranked, key=lambda x: x[0])[4]
shipA = dict(hl=500, blend=0.3, revw=10, contra=1_000_000, mode="fade", hedge=True)
finals = {"ship part2 (A)": shipA, "1054 cfg": c1054, "best-mean search": best_mean}
print("\n" + "=" * 90)
print("(3) FINALISTS across ALL 250-day legs (incl. early low-warmup), days 0-750")
print("=" * 90)
alllegs = EARLY + LEGS
hdr = "".join(f"{s}-{e}"[:8].rjust(9) for s, e in alllegs)
print(f"{'config':<20}{hdr}   {'mean*':>6}{'full':>6}")
for name, cfg in finals.items():
    row = [score_cfg(cfg, s, e) for s, e in alllegs]
    propermean = np.mean([score_cfg(cfg, s, e) for s, e in LEGS])
    full = score_cfg(cfg, *FULL)
    print(f"{name:<20}" + "".join(f"{v:9.0f}" for v in row) + f"   {propermean:6.0f}{full:6.0f}")
print("\n* mean = over proper-warmup legs (S>=250). 'full' = single 0-750 backtest (warmup 96).")
