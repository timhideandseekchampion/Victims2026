"""
clean_backtest.py — ONE eval-faithful harness (prcSoFar pattern, no look-ahead) to judge any
forecast by BOTH criteria the research demands:
  * per-window IC + t-stat + p-value CONSISTENCY (a real edge is stable across all windows), and
  * eval.py-faithful SCORE across every 250-day leg.
Candidates: plain ridge, RRR-k5, RRR+LoMac blend (user-requested), ridge+revz (ship).
"""
import numpy as np, pandas as pd
from scipy import stats
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

# ---- forecasts built ONLY from prcSoFar (=P[:, :t]); last col is day t-1 (causal) ----
def _design(prcSoFar, hl):
    lp = np.log(prcSoFar); r = lp[:, 1:] - lp[:, :-1]        # (51, T-1)
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]         # predict next return from last observed
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    return X - mx, Y - my, w, mx, my, xin

def f_ridge(prcSoFar, hl=1000, a=0.3):
    Xc, Yc, w, mx, my, xin = _design(prcSoFar, hl); p = Xc.shape[1]; Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + a * np.eye(p), Xc.T @ (Wm * Yc))
    f = my + (xin - mx) @ B; return f - f.mean()

def f_rrr(prcSoFar, hl=1000, a=0.3, k=5):
    Xc, Yc, w, mx, my, xin = _design(prcSoFar, hl); p = Xc.shape[1]; Wm = w[:, None]
    B = np.linalg.solve(Xc.T @ (Wm * Xc) + a * np.eye(p), Xc.T @ (Wm * Yc))
    Yhat = Xc @ B; M = Yhat.T @ (Wm * Yhat)
    ev, V = np.linalg.eigh(M); Pk = V[:, -k:]
    Brr = B @ Pk @ Pk.T; f = my + (xin - mx) @ Brr; return f - f.mean()

def f_lomac(prcSoFar, w=8):
    lp = np.log(prcSoFar); r = lp[:, 1:] - lp[:, :-1]
    acc = np.zeros(nInst - 1)
    for h in range(1, w + 1):
        rr = r[1:, -h]; acc += (rr - rr.mean())
    return -acc / (acc.std() + 1e-12)

def f_rrr_lomac(prcSoFar, lam=0.2):
    a = f_rrr(prcSoFar); b = f_lomac(prcSoFar, 5)
    return (1 - lam) * a / (a.std() + 1e-12) + lam * b

def f_ridge_revz(prcSoFar, blend=0.3, revw=10):
    a = f_ridge(prcSoFar); lp = np.log(prcSoFar)
    rr = lp[1:, -1] - lp[1:, -1 - revw]; rr = rr - rr.mean(); z = -rr / (rr.std() + 1e-12)
    return (1 - blend) * a / (a.std() + 1e-12) + blend * z

# ---- eval-faithful book: idio = sign(forecast)*$10k, + ALGO fade leg + beta hedge ----
def book_score(fc, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        soFar = prc[:, :t]; cur = soFar[:, -1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            wz = fc(soFar)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
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
    if mu <= 0 or sd < 1e-10: return mu, 0.0
    sr = np.sqrt(250) * mu / sd; return mu * sr**2 / (sr**2 + 1), sr

def ic_window(fc, S, E):
    ics = []
    for t in range(max(S, 96), min(E, nt - 1)):
        s = fc(prc[:, :t])
        fwd = np.log(prc[1:, t]) - np.log(prc[1:, t - 1]); fwd = fwd - fwd.mean()
        if s.std() > 1e-12 and fwd.std() > 1e-12: ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics); n = len(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(n))
    p = stats.t.sf(t, n - 1)                                 # one-sided p (IC>0)
    return ics.mean(), t, p

cands = {
    "ridge (hl1000)": f_ridge,
    "RRR k5": f_rrr,
    "RRR+LoMac blend": f_rrr_lomac,
    "ship: ridge+revz(0.3)": f_ridge_revz,
}
legs = [(S, S + 250) for S in range(250, 501, 50)]

print("PER-WINDOW IC / t / p  (consistency test — a real edge is stable across ALL legs)")
for name, fc in cands.items():
    print(f"\n{name}:")
    print(f"  {'leg':<12}{'IC':>9}{'t':>7}{'p':>9}")
    ics = []
    for S, E in legs:
        ic, t, p = ic_window(fc, S, E); ics.append(ic)
        print(f"  {f'{S}-{E}':<12}{ic:9.4f}{t:7.2f}{p:9.4f}")
    print(f"  -> IC mean {np.mean(ics):.4f}  std {np.std(ics):.4f}  min {np.min(ics):.4f}  (lower std = more consistent)")

print("\n\nSCORE by 250-day leg (eval-faithful book):")
print(f"{'leg':<12}" + "".join(f"{n[:14]:>16}" for n in cands))
tot = {n: 0.0 for n in cands}
for S, E in legs:
    cells = ""
    for name, fc in cands.items():
        sc, sr = book_score(fc, S, E); tot[name] += sc; cells += f"{sc:16.0f}"
    print(f"{f'{S}-{E}':<12}{cells}")
print(f"{'mean':<12}" + "".join(f"{tot[n]/len(legs):16.0f}" for n in cands))
