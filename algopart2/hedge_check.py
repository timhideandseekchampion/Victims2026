"""
hedge_check.py — does the HEDGE actually do anything? Instrument the book on 500-750:
  * residual market-beta $ of the idio leg BEFORE hedging (how much beta is even there?)
  * the DESIRED hedge $ vs the ROOM left in the ALGO cap after the fade vs the ACTUAL hedge applied
  * net beta $ after the (clipped) hedge
Then score hedge ON vs OFF. Uses the SAFE/part2 structure (HL-ensemble, contra=1M, sign-size).
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp_all = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
HLS = [250, 500, 1000, 2000]

_rc = {}
def ridge_z(t, hl, a=0.1):
    key = (t, hl)
    if key in _rc: return _rc[key]
    lp = lp_all[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(51), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f = f - f.mean(); v = f / (f.std() + 1e-12); _rc[key] = v; return v
def forecast(t, blend=0.30):
    core = np.mean([ridge_z(t, hl) for hl in HLS], 0)
    rr = lp_all[1:, t - 1] - lp_all[1:, t - 11]; rr = rr - rr.mean()
    return (1 - blend) * core + blend * (-rr / (rr.std() + 1e-12))

def run(hedge, contra=1_000_000, diag=False):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    D = {"resid_beta$": [], "desired_hedge$": [], "room$": [], "applied_hedge$": [], "net_beta$_after": []}
    for t in range(500, 751):
        soFar = prc[:, :t]; cur = soFar[:, -1]; pos = np.zeros(nInst)
        if t < 750 and t >= 130:
            wz = forecast(t)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = lp_all[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            av = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (contra / cur[0]), -cap, cap))
            r = lp_all[:, 1:t] - lp_all[:, :t - 1]; rA = r[0] - r[0].mean()
            bet = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / (rA @ rA + 1e-12)
            resid_beta_dollar = float((pos[1:] * cur[1:]) @ bet)      # residual market beta $ of idio leg
            hs = -resid_beta_dollar / cur[0]                          # desired hedge shares
            room = max(cap - abs(av), 0.0)
            applied = float(np.clip(hs, -room, room)) if hedge else 0.0
            pos[0] = av + applied
            if diag:
                D["resid_beta$"].append(resid_beta_dollar)
                D["desired_hedge$"].append(hs * cur[0])
                D["room$"].append(room * cur[0])
                D["applied_hedge$"].append(applied * cur[0])
                D["net_beta$_after"].append(resid_beta_dollar + applied * cur[0])
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else:
            pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm
        comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > 500: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sc = mu * (250 * mu**2 / sd**2) / (250 * mu**2 / sd**2 + 1) if mu > 0 and sd > 1e-9 else mu
    sr = np.sqrt(250) * mu / sd
    return sc, sr, D

print("=== hedge mechanics on 500-750 (SAFE structure, contra=$1M) ===")
_, _, D = run(hedge=True, diag=True)
for k, v in D.items():
    v = np.array(v); print(f"  avg |{k:<16}| = ${np.abs(v).mean():>10,.0f}   (median ${np.median(np.abs(v)):>9,.0f})")
print(f"  -> fade pins the $100k cap on {np.mean(np.abs(np.array(D['room$']))<1000)*100:.0f}% of days (room<$1k)")

print("\n=== score: hedge ON vs OFF, at contra=$1M (fade pins cap) and contra=$0 (no fade -> room for hedge) ===")
for contra in (1_000_000, 200_000, 0):
    on, sron, _ = run(True, contra); off, sroff, _ = run(False, contra)
    print(f"  contra=${contra:>9,}:  hedge ON {on:6.1f} (SR{sron:4.1f})   OFF {off:6.1f} (SR{sroff:4.1f})   diff {on-off:+.1f}")
