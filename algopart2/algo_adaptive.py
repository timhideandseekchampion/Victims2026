"""
algo_adaptive.py — the ALGO leg is REGIME-CHANGING (reverts 400-500, random 500-750). Does an
ADAPTIVE ALGO leg beat the fixed fade? Idio leg held fixed (ridge+revz sign-deploy); only the
ALGO leg logic varies. Compare across every 250-day leg + the two regimes explicitly.

ALGO leg variants (all causal):
  fade   : fixed  pos0 = -clip(z)/3 * $100k        (current ship)
  off    : idio-only (no ALGO leg)
  ols    : rolling OLS predicts next ALGO return from its 30d-move z -> follow the fitted sign
           (beta<0 fade, beta>0 follow, beta~0 flat) -> auto-adapts to the regime
  gate   : fixed fade * w, w = recent realized reversion strength in [0,1] (turn off when not reverting)
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp_all = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

_fc = {}
def idio_forecast(t, blend=0.3, revw=10, hl=1000, a=0.3):
    if t in _fc: return _fc[t]
    lp = lp_all[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(51), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; a1 = f - f.mean()
    rr = lp[1:, -1] - lp[1:, -1 - revw]; rr = rr - rr.mean(); z = -rr / (rr.std() + 1e-12)
    wz = (1 - blend) * a1 / (a1.std() + 1e-12) + blend * z
    _fc[t] = wz; return wz

def algo_leg(soFar, mode, cap, k=30, wz=60, L=250):
    lpA = np.log(soFar[0]); mv = lpA[k:] - lpA[:-k]
    if len(mv) < wz + 5: return 0.0
    z = (mv[-1] - mv[-wz:].mean()) / (mv[-wz:].std() + 1e-12)
    if mode == "off":
        return 0.0
    if mode == "fade":
        return float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / soFar[0, -1]), -cap, cap))
    r = lpA[1:] - lpA[:-1]
    if mode == "ols":
        # predict next-day return from the standardized k-day move, trailing L days
        g = mv - mv.mean()                                   # move series
        gser = g[:-1]; nxt = r[k:][:len(gser)]               # align: move at i -> return i+1
        gL = gser[-L:]; nL = nxt[-L:]
        if len(gL) < 30 or gL.std() < 1e-9: return 0.0
        beta = np.cov(gL, nL)[0, 1] / (gL.var() + 1e-12)
        pred = beta * (g[-1])                                 # predicted next return
        ps = beta * gL                                        # in-window predictions -> scale
        zp = pred / (ps.std() + 1e-12)
        return float(np.clip(np.clip(zp, -3, 3) / 3.0 * (1_000_000 / soFar[0, -1]), -cap, cap))
    if mode == "gate":
        # w = recent realized reversion strength: -corr(k-day move, next return) over last L, in [0,1]
        g = mv - mv.mean(); gser = g[:-1]; nxt = r[k:][:len(gser)]
        gL = gser[-L:]; nL = nxt[-L:]
        if len(gL) < 30 or gL.std() < 1e-9: w_ = 0.0
        else: w_ = float(np.clip(-np.corrcoef(gL, nL)[0, 1] * 3, 0, 1))   # scale corr to gate
        return float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / soFar[0, -1]), -cap, cap)) * w_

def book(mode, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        soFar = prc[:, :t]; cur = soFar[:, -1]; pos = np.zeros(nInst)
        if t < Ed and t >= 130:
            wz = idio_forecast(t)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            av = algo_leg(soFar, mode, cap)
            r = lp_all[:, 1:t] - lp_all[:, :t - 1]; rA = r[0] - r[0].mean()
            bet = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / (rA @ rA + 1e-12)
            hs = -((pos[1:] * cur[1:]) @ bet) / cur[0]
            room = max(cap - abs(av), 0.0); pos[0] = av + float(np.clip(hs, -room, room))
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

modes = ["fade", "off", "ols", "gate"]
legs = [(S, S + 250) for S in range(250, 501, 50)]
print("SCORE by 250-day leg — ALGO leg variants (idio leg fixed):\n")
print(f"{'leg':<12}" + "".join(f"{m:>10}" for m in modes))
tot = {m: 0.0 for m in modes}
for S, E in legs:
    row = ""
    for m in modes:
        sc = book(m, S, E); tot[m] += sc; row += f"{sc:10.0f}"
    print(f"{f'{S}-{E}':<12}{row}")
print(f"{'mean':<12}" + "".join(f"{tot[m]/len(legs):10.0f}" for m in modes))
print("\nWatch: on 500-750 (ALGO random) does adaptive match 'off'? On 250-500/300-550 (ALGO reverts) does it match 'fade'?")
