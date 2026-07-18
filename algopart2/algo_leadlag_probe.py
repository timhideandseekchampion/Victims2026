"""algo_leadlag_probe.py — should ALGO be driven by lead-lag instead of / alongside reversion?
Measures, on ALGO's OWN next-day return, the IC of:
  (a) the lead-lag market signal  = mean of the RAW (pre-demean) ensemble forecast  [common component]
  (b) the tilt                    = sum(sign(wz))                                    [what the idio book leaks]
  (c) the reversion fade z        = the signal ALGO is currently traded on (-30d move z)
Also backtests blending the lead-lag market signal into the ALGO leg vs the shipped reversion-only leg.
Same causal, no-look-ahead ridge as SAFE.py."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = [250, 500, 1000, 2000]

_cache = {}
def ridge(t, hl, a=0.1):
    """returns (raw forecast f before demean, z-scored demeaned v)."""
    key = (t, hl)
    if key in _cache: return _cache[key]
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(nInst), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B
    d = f - f.mean(); v = d / (d.std() + 1e-12)
    _cache[key] = (f, v); return _cache[key]

def revz(t, w=10):
    rr = logp[1:, t-1] - logp[1:, t-1-w]; rr = rr - rr.mean(); return -rr/(rr.std()+1e-12)

def sigs(t, blend=0.3):
    raws, vs = zip(*[ridge(t, hl) for hl in ENS])
    v = np.mean(vs, 0)
    wz = (1-blend)*v + blend*revz(t)
    ll_market = float(np.mean([rw.mean() for rw in raws]))     # (a) common component of forecast
    tilt = float(np.sign(wz).sum())                            # (b) idio-book long/short imbalance
    lpA = logp[0, :t]; mv = lpA[30:]-lpA[:-30]
    revfade = float(-np.clip((mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12), -3, 3))  # (c) fade signal (sign = position dir)
    return ll_market, tilt, revfade

def ic(Sd, Ed):
    a, b, c, y = [], [], [], []
    for t in range(Sd, Ed):
        m, tl, rf = sigs(t)
        a.append(m); b.append(tl); c.append(rf); y.append(float(r_all[0, t]))  # next-day ALGO return
    a, b, c, y = map(np.array, (a, b, c, y))
    return {"ll_market": np.corrcoef(a, y)[0,1],
            "tilt":      np.corrcoef(b, y)[0,1],
            "rev_fade":  np.corrcoef(c, y)[0,1], "n": len(y)}

for lbl, (S, E) in {"500-750": (500, 749), "400-500": (400, 499), "250-400": (250, 399)}.items():
    d = ic(S, E)
    print(f"[{lbl}] IC on ALGO next-day ret (n={d['n']}):  "
          f"lead-lag(common)={d['ll_market']:+.3f}   tilt={d['tilt']:+.3f}   rev_fade={d['rev_fade']:+.3f}")
