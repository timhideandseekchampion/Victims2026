"""
ou_speed.py — test Jurek-Yang OU speed-weighted sizing  w_i ~ z_i * sqrt(kappa_i)/sigma_i.
Decisive precheck (per the research): is the per-name reversion speed / vol HETEROGENEOUS?
If half-lives & sigmas cluster tight -> speed-weighting == plain sizing -> skip. If they spread,
it may add Sharpe. Then backtest sizing schemes (sign / z-prop / speed-weighted / inverse-vol)
by SCORE across legs. Marriott-Pope bias-correct kappa (Yu 2012 / RANK-4).
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

# ---- per-name OU fit on the cross-sectional deviation s_i = logP_i - mean_j logP_j ----
def ou_params(t, win=120):
    dev = lp[1:, t - win:t] - lp[1:, t - win:t].mean(0, keepdims=True)   # (50, win)
    phi = np.zeros(50); sig = np.zeros(50)
    for i in range(50):
        x = dev[i]
        if x[:-1].std() < 1e-9: phi[i] = 0.0; sig[i] = x.std() + 1e-9; continue
        b = np.polyfit(x[:-1], x[1:], 1)[0]
        b = b + (1 + 3 * b) / win                      # Marriott-Pope bias correction
        phi[i] = np.clip(b, 1e-4, 0.999)
        sig[i] = (x[1:] - phi[i] * x[:-1]).std() + 1e-9
    kappa = -np.log(phi)
    return kappa, sig

print("Per-name OU heterogeneity on days 400-750 (the decisive precheck):")
hl_all = []; sg_all = []
for t in range(450, 749, 50):
    k, s = ou_params(t)
    hl = np.log(2) / k
    hl_all.append(hl); sg_all.append(s)
    print(f"  day {t}: half-life  median {np.median(hl):5.1f}d  IQR [{np.percentile(hl,25):.1f},{np.percentile(hl,75):.1f}]"
          f"  CV(sqrt(k)/sig) = {np.std(np.sqrt(k)/s)/np.mean(np.sqrt(k)/s):.3f}")
HL = np.concatenate(hl_all); SG = np.concatenate(sg_all)
print(f"  => half-life dispersion: median {np.median(HL):.1f}d, IQR/median = {(np.percentile(HL,75)-np.percentile(HL,25))/np.median(HL):.2f}")
print(f"  => if the speed-weight CV above is small (<~0.2), speed-weighting ~ plain sizing (skip).\n")

# ---- backtest sizing schemes (forecast = ridge+revz blend, same as ship) ----
def _design(soFar, hl=1000):
    r = np.log(soFar)[:, 1:] - np.log(soFar)[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    return X - mx, Y - my, w, mx, my, xin
def forecast(soFar, blend=0.3, revw=10):
    Xc, Yc, w, mx, my, xin = _design(soFar); p = Xc.shape[1]; Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + 0.3 * np.eye(p), Xc.T @ (Wm * Yc))
    f = my + (xin - mx) @ B; a = f - f.mean()
    lpp = np.log(soFar); rr = lpp[1:, -1] - lpp[1:, -1 - revw]; rr = rr - rr.mean(); z = -rr / (rr.std() + 1e-12)
    return (1 - blend) * a / (a.std() + 1e-12) + blend * z

def book(sizing, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        soFar = prc[:, :t]; cur = soFar[:, -1]; pos = np.zeros(nInst)
        if t < Ed and t >= 130:
            wz = forecast(soFar)
            if sizing == "sign":
                raw = np.sign(wz)
            elif sizing == "zprop":
                raw = np.clip(wz / (np.abs(wz).max() + 1e-12), -1, 1)
            elif sizing == "speed":
                k, s = ou_params(t); sw_ = np.sqrt(k) / s
                raw = wz * sw_; raw = raw / (np.abs(raw).max() + 1e-12)
            elif sizing == "invvol":
                s = (np.log(prc[1:, t - 21:t]) - np.log(prc[1:, t - 22:t - 1])).std(1) + 1e-9
                raw = (wz / s); raw = raw / (np.abs(raw).max() + 1e-12)
            pos[1:] = raw * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = np.log(soFar[0]); mv = lpA[30:] - lpA[:-30]
            zz = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            av = float(np.clip(-np.clip(zz, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            r = np.log(soFar)[:, 1:] - np.log(soFar)[:, :-1]; rA = r[0] - r[0].mean()
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
    if mu <= 0 or sd < 1e-10: return mu, 0.0, pll.sum()
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1), sr, pll.sum()

print("SCORE / Sharpe / gross-PnL by sizing scheme (forecast fixed = ridge+revz):")
legs = [(S, S + 250) for S in range(250, 501, 50)]
print(f"{'leg':<12}{'sign':>18}{'zprop':>18}{'speed(JurekYang)':>20}{'invvol':>18}")
tot = {m: 0.0 for m in ("sign", "zprop", "speed", "invvol")}
shp = {m: [] for m in tot}
for S, E in legs:
    cells = ""
    for m in ("sign", "zprop", "speed", "invvol"):
        sc, sr, tp = book(m, S, E); tot[m] += sc; shp[m].append(sr)
        cells += f"{sc:8.0f}(SR{sr:4.1f})"
    print(f"{f'{S}-{E}':<12}{cells}")
print(f"{'mean score':<12}" + "".join(f"{tot[m]/len(legs):18.0f}" for m in ("sign","zprop","speed","invvol")))
print(f"{'mean Sharpe':<12}" + "".join(f"{np.mean(shp[m]):18.2f}" for m in ("sign","zprop","speed","invvol")))
