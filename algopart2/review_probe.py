"""review_probe.py — code review checks:
(1) does my engine reproduce eval_safe's official number (612.98) on 500-750? (no-look-ahead sanity)
(2) how imbalanced is sign(wz)? net $ and net index-beta of the book (unhedged with HEDGE=False)
(3) would EXACT dollar-neutral sizing (rank median-split) beat sign(wz)? removes the net-beta leak"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc)
ENS = [250, 500, 1000, 2000]

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
def wzsig(t, blend):
    a = np.mean([ridge_z(t, hl) for hl in ENS], 0)
    return (1 - blend) * a + blend * revz(t, 10)

def book(blend, Sd, Ed, sizing="sign"):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            wz = wzsig(t, blend)
            if sizing == "sign":
                pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            else:  # exact balanced: median split -> 25 long / 25 short
                s = np.where(wz >= np.median(wz), 1.0, -1.0)
                pos[1:] = s * (dlr[1:] / cur[1:])
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
    if mu <= 0 or sd < 1e-10: return mu, sd
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1), sd

# (1) sanity vs official eval_safe (SAFE = ens b.30). Expect ~612.98
s, _ = book(0.30, 500, 750, "sign")
print(f"(1) engine SAFE(ens b.30) 500-750 = {s:.2f}   (eval_safe.py official = 612.98)")

# (2) net exposure / beta of sign(wz) book over 500-750
r_all = logp[:, 1:] - logp[:, :-1]
nets = []; betas_net = []
for t in range(500, 750):
    cur = prc[:, t - 1]; wz = wzsig(t, 0.30)
    stk = np.sign(wz) * ((dlr[1:] / cur[1:]).astype(int))
    net_dollar = float((stk * cur[1:]).sum())
    r = r_all[:, :t - 1]; rA = r[0] - r[0].mean(); den = rA @ rA + 1e-12
    beta = ((r[1:] - r[1:].mean(1, keepdims=True)) @ rA) / den
    net_beta_dollar = float((stk * cur[1:]) @ beta)     # $ index-equivalent exposure of stock book
    nets.append(net_dollar); betas_net.append(net_beta_dollar)
nets = np.array(nets); betas_net = np.array(betas_net)
print(f"(2) sign(wz) stock book net $:  mean {nets.mean():+,.0f}  std {nets.std():,.0f}  "
      f"|net| p90 {np.percentile(np.abs(nets),90):,.0f}   (gross = $500,000)")
print(f"    residual index-beta exposure $: mean {betas_net.mean():+,.0f}  std {betas_net.std():,.0f}  "
      f"|.| p90 {np.percentile(np.abs(betas_net),90):,.0f}  (vs ALGO cap $100k, all used by fade)")

# (3) exact dollar-neutral (median split) vs sign, on long windows
print("(3) sizing comparison (score / pnl-std):")
for blend in (0.20, 0.30):
    for L, lo in ((250, 346), (500, 500)):
        ends = list(range(lo, nDays + 1, 20))
        sg = np.array([book(blend, e - L, e, "sign")[0] for e in ends])
        bl = np.array([book(blend, e - L, e, "bal")[0] for e in ends])
        print(f"    b{blend} {L}d: sign mean {sg.mean():6.0f} floor {sg.min():5.0f} | "
              f"balanced mean {bl.mean():6.0f} floor {bl.min():5.0f}")
