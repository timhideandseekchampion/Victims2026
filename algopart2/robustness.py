"""
robustness.py — (A) one honest IC probe: cross-validated OPTIMAL combination of all signals
(fit weights on 250-500, test forward on 500-750) — can a full multi-signal fit beat 0.079 OOS
with a consistent t? (B) ROBUSTNESS refinement: rank forecast configs by the WORST-window score
(the qualifying-relevant metric), testing HL-ensembling (variance reduction), blend ratios, and
ALGO-gating. Clean no-look-ahead engine.
"""
import numpy as np, pandas as pd
from scipy import stats
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp_all = np.log(prc)
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
HLS = [250, 500, 1000, 2000]

# ---- cached per-day ridge forecasts (per HL) and reversion signals ----
_rc = {}
def ridge_z(t, hl, a=0.1):   # a=0.1 matches the shipped part2/maxEV files
    key = (t, hl)
    if key in _rc: return _rc[key]
    lp = lp_all[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(51), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f = f - f.mean()
    v = f / (f.std() + 1e-12); _rc[key] = v; return v
def revz(t, w):
    rr = lp_all[1:, t - 1] - lp_all[1:, t - 1 - w]; rr = rr - rr.mean()
    return -rr / (rr.std() + 1e-12)

# ================= (A) cross-validated optimal signal combination =================
print("=" * 78)
print("(A) OPTIMAL multi-signal combination — fit weights on 250-500, test on 500-750")
print("=" * 78)
def signal_matrix(t):
    return np.column_stack([ridge_z(t, 500), ridge_z(t, 2000), revz(t, 5), revz(t, 10), revz(t, 40)])
names = ["ridge500", "ridge2000", "revz5", "revz10", "revz40"]
# gather (signal, fwd) over fit window, solve for weights maximizing IC (=OLS of fwd on signals, pooled)
def gather(S, E):
    Xs = []; ys = []
    for t in range(max(S, 96), min(E, nt - 1)):
        s = signal_matrix(t); fwd = lp_all[1:, t] - lp_all[1:, t - 1]; fwd = fwd - fwd.mean()
        Xs.append(s); ys.append(fwd)
    return np.vstack(Xs), np.concatenate(ys)
Xf, yf = gather(250, 500)
wgt = np.linalg.solve(Xf.T @ Xf + 1e-6 * np.eye(5), Xf.T @ yf)   # IC-optimal linear combo (pooled OLS)
wgt = wgt / np.abs(wgt).sum()
print("  fitted weights (250-500):", {n: round(float(w), 3) for n, w in zip(names, wgt)})
def combo_ic(S, E, w):
    ics = []
    for t in range(max(S, 96), min(E, nt - 1)):
        s = signal_matrix(t) @ w; fwd = lp_all[1:, t] - lp_all[1:, t - 1]; fwd = fwd - fwd.mean()
        if s.std() > 1e-12 and fwd.std() > 1e-12: ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics); tt = ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))
    return ics.mean(), tt, stats.t.sf(tt, len(ics) - 1)
for lbl, S, E in [("fit  250-500", 250, 500), ("OOS  500-750", 500, 750), ("all  400-750", 400, 749)]:
    ic, tt, p = combo_ic(S, E, wgt)
    print(f"  {lbl}: IC={ic:.4f}  t={tt:.2f}  p={p:.4g}")
ic5, t5, _ = combo_ic(500, 750, np.array([0.7, 0, 0, 0.3, 0]))
print(f"  (ship-like ridge500+0.3revz10 on 500-750 for reference: IC={ic5:.4f} t={t5:.2f})")
print("  verdict: OOS IC vs 0.079 baseline decides if a fuller combination adds anything.\n")

# ================= (B) robustness: rank by WORST-window score =================
print("=" * 78)
print("(B) ROBUSTNESS — score by leg, ranked by FLOOR (worst window = qualifying metric)")
print("=" * 78)
def forecast(t, kind, blend):
    if kind == "ens":
        a = np.mean([ridge_z(t, hl) for hl in HLS], 0)
    else:
        a = ridge_z(t, int(kind))
    return (1 - blend) * a + blend * revz(t, 10)

def book(kind, blend, algo, Sd, Ed):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        soFar = prc[:, :t]; cur = soFar[:, -1]; pos = np.zeros(nInst)
        if t < Ed and t >= 130:
            wz = forecast(t, kind, blend)
            pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]
            lpA = lp_all[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            av = -np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0])
            if algo == "gate":
                g = mv - mv.mean(); r0 = lpA[1:] - lpA[:-1]; gs = g[:-1]; nx = r0[30:][:len(gs)]
                gL, nL = gs[-250:], nx[-250:]
                wgate = np.clip(-np.corrcoef(gL, nL)[0, 1] * 3, 0, 1) if len(gL) > 30 and gL.std() > 1e-9 else 0.0
                av *= wgate
            av = float(np.clip(av, -cap, cap))
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

legs = [(96, 346), (150, 400), (200, 450), (250, 500), (300, 550), (350, 600), (400, 650), (450, 700), (500, 750)]
configs = [
    ("part2 (hl500 b.30 fade)", "500", 0.30, "fade"),
    ("maxEV (hl1000 b.15 fade)", "1000", 0.15, "fade"),
    ("ENS b.30 fade", "ens", 0.30, "fade"),
    ("ENS b.30 gate", "ens", 0.30, "gate"),
    ("ENS b.45 fade", "ens", 0.45, "fade"),
    ("hl500 b.45 fade", "500", 0.45, "fade"),
]
print(f"{'config':<26}{'cold96':>8}{'min':>7}{'mean':>7}{'std':>7}{'500-750':>9}")
rows = []
for label, kind, blend, algo in configs:
    scs = [book(kind, blend, algo, S, E) for S, E in legs]
    cold = scs[0]; mn = min(scs); mean = np.mean(scs); sd = np.std(scs); last = scs[-1]
    rows.append((label, cold, mn, mean, sd, last))
for label, cold, mn, mean, sd, last in sorted(rows, key=lambda x: -x[2]):
    print(f"{label:<26}{cold:8.0f}{mn:7.0f}{mean:7.0f}{sd:7.0f}{last:9.0f}")
print("\nranked by FLOOR (min). higher min + lower std = more robust for consistent qualifying.")
