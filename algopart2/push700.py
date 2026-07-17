"""
push700.py — can we score 700-800 on the 500-750 leg? Large causal knob search.

Runs a big grid (thousands of configs) of the combined book on window 500-750, with:
  * forecast/signal caching so thousands of backtests stay fast and no-look-ahead,
  * IN-SAMPLE fit on 500-750 (what "more tests on the same data" yields),
  * HONEST OOS: the SAME configs re-scored on 400-500 (fit here) -> forward to 500-750,
  * the overfitting gap between the two.

Purpose: find the true causal ceiling on this leg, and quantify how much of any 700 is real
vs an artifact of searching. eval.py-faithful scoring.
"""
import itertools, json
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
logp = np.log(prc)

# ---------------------------------------------------------------- cached signal engine
def ewls(X, Y, hl, a=0.1):
    n, p = X.shape; lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY), mx, my

_fc = {}
def forecast(t, hl):                       # lead-lag forecast using days [0,t) -> 50-vec, demeaned
    key = (t, hl)
    if key in _fc: return _fc[key]
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    B, mx, my = ewls(r[:, :-1].T, r[1:, 1:].T, hl)
    pred = my + (r[:, -1] - mx) @ B
    v = pred - pred.mean(); _fc[key] = v; return v

_rev = {}
def revsig(t, w):                          # -zscore of w-day idio return, demeaned -> 50-vec
    key = (t, w)
    if key in _rev: return _rev[key]
    lp = logp[1:, :t]; rr = lp[:, -1] - lp[:, -1 - w]; rr = rr - rr.mean()
    v = -rr / (rr.std() + 1e-12); _rev[key] = v; return v

_algo = {}
def algo_leg(t, mode, k=30, wz=60):        # ALGO index position fraction in [-1,1]*conviction
    key = (t, mode, k, wz)
    if key in _algo: return _algo[key]
    lpA = logp[0, :t]; mv = lpA[k:] - lpA[:-k]
    z = (mv[-1] - mv[-wz:].mean()) / (mv[-wz:].std() + 1e-12)
    if mode == "fade":   val = -np.clip(z, -3, 3) / 3.0
    elif mode == "ols":                    # rolling OLS fade/follow coef on trailing 250d
        m = mv[-250:]; zz = (m - m.mean()) / (m.std() + 1e-12)
        fr = (lpA[1:] - lpA[:-1])[-len(zz):]
        b = np.cov(zz[:-1], fr[1:])[0, 1] / (np.var(zz[:-1]) + 1e-12) if len(zz) > 5 else 0.0
        val = np.clip(b * z, -1, 1)
    else: val = 0.0
    _algo[key] = val; return val

# ---------------------------------------------------------------- fast eval-faithful loop
def score_cfg(cfg, S, E):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    hl, conv, sizing, blend, revw, mode, contra, hedge = (
        cfg["hl"], cfg["conv"], cfg["sizing"], cfg["blend"], cfg["revw"], cfg["mode"],
        cfg["contra"], cfg["hedge"])
    for t in range(S, E + 1):
        cur = prc[:, t - 1]
        if t < E and t >= 96:
            pos = np.zeros(nInst)
            f = forecast(t, hl); wz = f / (f.std() + 1e-12)
            if blend > 0:
                rv = revsig(t, revw); wz = (1 - blend) * wz + blend * rv
            if sizing == "sign":
                base = np.sign(wz)
            else:  # z-proportional, capped at 1
                base = np.clip(wz, -1, 1)
            take = np.abs(wz) >= conv
            pos[1:] = np.where(take, base, 0.0) * (dlr[1:] / cur[1:])
            # ALGO leg
            cap = dlr[0] / cur[0]
            av = algo_leg(t, mode) * (contra / cur[0])
            av = float(np.clip(av, -cap, cap))
            hs = 0.0
            if hedge:
                r = logp[:, 1:t] - logp[:, :t - 1]
                rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
                betas = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
                hs = -((pos[1:] * cur[1:]) @ betas) / cur[0]
            room = max(cap - abs(av), 0.0)
            pos[0] = av + float(np.clip(hs, -room, room))
            lim = (dlr / cur).astype(int); newPos = np.clip(pos, -lim, lim).astype(int)
        elif t < E:
            newPos = cp.copy()
        else:
            newPos = cp.copy()
        d = newPos - cp; cash -= cur.dot(d) + comm
        dv = cur * np.abs(d); comm = np.sum(dv * commRate); cp = newPos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu, (np.sqrt(250) * mu / sd if sd > 0 else 0.0), pll.sum()
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1), sr, pll.sum()

# ---------------------------------------------------------------- the grid
grid = dict(
    hl=[250, 500, 1000, 2000],
    conv=[0.0, 0.1, 0.2, 0.3],
    sizing=["sign", "zprop"],
    blend=[0.0, 0.15, 0.3, 0.5],
    revw=[5, 10, 20],
    mode=["off", "fade", "ols"],
    contra=[0, 200_000, 500_000, 1_000_000],
    hedge=[True, False],
)
keys = list(grid)
combos = list(itertools.product(*[grid[k] for k in keys]))
# prune redundant: contra irrelevant when mode==off; revw irrelevant when blend==0
seen = set(); configs = []
for c in combos:
    cfg = dict(zip(keys, c))
    if cfg["mode"] == "off": cfg["contra"] = 0
    if cfg["blend"] == 0: cfg["revw"] = 10
    k = tuple(sorted(cfg.items()))
    if k in seen: continue
    seen.add(k); configs.append(cfg)

print(f"searching {len(configs)} unique configs on 500-750 (in-sample) ...")
res = []
for i, cfg in enumerate(configs):
    s_in, sr_in, tot_in = score_cfg(cfg, 500, 750)
    res.append((s_in, sr_in, cfg))
    if (i + 1) % 500 == 0: print(f"  {i+1}/{len(configs)}  best-so-far {max(r[0] for r in res):.0f}")
res.sort(key=lambda x: -x[0])

print("\n" + "=" * 92)
print("IN-SAMPLE fit on 500-750 (this is what 'more tests on the same data' produces)")
print("=" * 92)
print(f"{'S@500-750':>10} {'Sharpe':>7} {'S@400-500':>10} {'gap':>7}  config")
oos_check = []
for s_in, sr_in, cfg in res[:15]:
    s_out, _, _ = score_cfg(cfg, 400, 500)   # same config on a DIFFERENT window
    oos_check.append((s_in, s_out, cfg))
    cc = {k: cfg[k] for k in ("hl", "conv", "sizing", "blend", "mode", "contra", "hedge")}
    print(f"{s_in:10.0f} {sr_in:7.2f} {s_out:10.0f} {s_in - s_out:7.0f}  {cc}")

# ---- HONEST OOS: pick the config that is BEST on 400-500, apply forward to 500-750 -------
print("\n" + "=" * 92)
print("HONEST OOS — select on 400-500, apply FORWARD to 500-750 (no peeking at the answer)")
print("=" * 92)
by_train = sorted(((score_cfg(cfg, 400, 500)[0], cfg) for cfg in configs), key=lambda x: -x[0])
train_best_s, train_best_cfg = by_train[0]
fwd_s, fwd_sr, _ = score_cfg(train_best_cfg, 500, 750)
cc = {k: train_best_cfg[k] for k in ("hl", "conv", "sizing", "blend", "mode", "contra", "hedge")}
print(f"best on 400-500 = {train_best_s:.0f}  ->  FORWARD 500-750 = {fwd_s:.0f} (Sharpe {fwd_sr:.2f})")
print(f"config: {cc}")

# how many configs clear 700 in-sample, and what they average forward
clear700 = [c for s, _, c in res if s >= 700]
if clear700:
    fwd = [score_cfg(c, 400, 500)[0] for c in clear700]
    print(f"\n{len(clear700)} configs clear 700 IN-SAMPLE on 500-750; those SAME configs average "
          f"{np.mean(fwd):.0f} on 400-500 (median {np.median(fwd):.0f}).")
else:
    print(f"\nNO config in the {len(configs)}-config grid reaches 700 causally on 500-750. "
          f"Max achievable = {res[0][0]:.0f}.")

json.dump({"in_sample_top": [(float(s), c) for s, _, c in res[:15]],
           "oos_forward": {"train_best": float(train_best_s), "forward_500_750": float(fwd_s), "cfg": cc},
           "max_in_sample": float(res[0][0]), "n_configs": len(configs)},
          open("push700_results.json", "w"), indent=2)
print("\n[written push700_results.json]")
