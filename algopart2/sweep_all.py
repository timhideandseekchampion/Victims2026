"""sweep_all.py — EXHAUSTIVE grid over EVERY SAFE knob: RIDGE_A, BLEND, REV_W, CONTRA_K,
CONTRA_WZ (HALF_LIVES/CONTRA_DOL/HEDGE established separately). Fast screen via a single
continuous daily-PnL pass per config, scored on the qualifier-relevant horizons. Top configs
then re-verified exactly. Discipline: judge on 500d mean+floor AND 250d AND the clean 500-750
leg; a win that only shows on one horizon on ~1.5 independent windows is overfitting."""
import numpy as np, pandas as pd, itertools, json
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = (250, 500, 1000, 2000)

A_GRID = (0.03, 0.1, 0.3, 1.0)
B_GRID = (0.15, 0.20, 0.25, 0.30, 0.35)
RW_GRID = (5, 10, 20)
CK_GRID = (20, 30, 45)
CWZ_GRID = (40, 60, 90)
CONTRA = 1_000_000
DAYS = range(96, nDays)                       # decision days with a realized next return

# ---- precompute z-scored ensemble ridge forecast per (t, a) ----
def ridge_z_raw(t, hl, a):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(nInst), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; f = f - f.mean(); return f / (f.std() + 1e-12)

print("precomputing ensemble ridge forecasts per (t, RIDGE_A)...")
ENSF = {}                                     # ENSF[a][t] = 50-vec ensemble forecast
for a in A_GRID:
    arr = {}
    for t in DAYS:
        arr[t] = np.mean([ridge_z_raw(t, hl, a) for hl in ENS], 0)
    ENSF[a] = arr
print("  done.")
REVZ = {}
for rw in RW_GRID:
    REVZ[rw] = {t: (lambda rr: -(rr - rr.mean()) / ((rr - rr.mean()).std() + 1e-12))(logp[1:, t - 1] - logp[1:, t - 1 - rw]) for t in DAYS}
AF = {}                                        # ALGO fade fraction per (ck,cwz)
for ck in CK_GRID:
    for cwz in CWZ_GRID:
        d = {}
        for t in DAYS:
            lpA = logp[0, :t]; mv = lpA[ck:] - lpA[:-ck]
            if len(mv) >= cwz:
                z = (mv[-1] - mv[-cwz:].mean()) / (mv[-cwz:].std() + 1e-12); d[t] = -np.clip(z, -3, 3) / 3.0
            else: d[t] = 0.0
        AF[(ck, cwz)] = d

def series(a, blend, rw, ck, cwz):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = np.full(nDays, np.nan)
    ef = ENSF[a]; rz = REVZ[rw]; af = AF[(ck, cwz)]
    for t in DAYS:
        cur = prc[:, t - 1]
        wz = (1 - blend) * ef[t] + blend * rz[t]
        pos = np.zeros(nInst); pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
        cap = dlr[0] / cur[0]; pos[0] = float(np.clip(af[t] * (CONTRA / cur[0]), -cap, cap))
        lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        dp = pos - cp; cash -= cur.dot(dp) + comm; comm = np.sum(cur * np.abs(dp) * commRate); cp = pos
        pll[t] = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
    return pll

def score_win(pll, Sd, Ed):
    seg = pll[Sd + 1:Ed + 1]; seg = seg[~np.isnan(seg)]
    mu, sd = seg.mean(), seg.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1)

W500 = [(e - 500, e) for e in range(500, nDays + 1, 20)]
W250 = [(e - 250, e) for e in range(346, nDays + 1, 25)]
rows = []
combos = list(itertools.product(A_GRID, B_GRID, RW_GRID, CK_GRID, CWZ_GRID))
print(f"screening {len(combos)} configs...")
for i, (a, b, rw, ck, cwz) in enumerate(combos):
    pll = series(a, b, rw, ck, cwz)
    s5 = np.array([score_win(pll, S, E) for S, E in W500])
    s25 = np.array([score_win(pll, S, E) for S, E in W250])
    leg = score_win(pll, 500, 750)
    rows.append(dict(a=a, b=b, rw=rw, ck=ck, cwz=cwz,
                     m5=s5.mean(), f5=s5.min(), m25=s25.mean(), f25=s25.min(), leg=leg))
    if (i + 1) % 100 == 0: print(f"  {i+1}/{len(combos)}")

base = next(r for r in rows if (r['a'], r['b'], r['rw'], r['ck'], r['cwz']) == (0.1, 0.30, 10, 30, 60))
baseq = next(r for r in rows if (r['a'], r['b'], r['rw'], r['ck'], r['cwz']) == (0.1, 0.20, 10, 30, 60))
print(f"\nBASELINE SAFE (a.1 b.30 rw10 ck30 cwz60): 500d mean {base['m5']:.0f} floor {base['f5']:.0f} | leg {base['leg']:.0f}")
print(f"BASELINE QUAL (a.1 b.20 rw10 ck30 cwz60): 500d mean {baseq['m5']:.0f} floor {baseq['f5']:.0f} | leg {baseq['leg']:.0f}")

def top(key, lbl, n=8):
    print(f"\nTop {n} by {lbl}:")
    print(f"  {'a':>5}{'blend':>7}{'revw':>6}{'ck':>4}{'cwz':>5}{'  |':>3}{'500m':>7}{'500f':>7}{'250m':>7}{'250f':>7}{'leg':>7}")
    for r in sorted(rows, key=lambda r: -r[key])[:n]:
        print(f"  {r['a']:>5}{r['b']:>7}{r['rw']:>6}{r['ck']:>4}{r['cwz']:>5}{'  |':>3}{r['m5']:>7.0f}{r['f5']:>7.0f}{r['m25']:>7.0f}{r['f25']:>7.0f}{r['leg']:>7.0f}")
top('m5', '500d MEAN (qualifier EV)')
top('f5', '500d FLOOR (survival)')
# robust: rank by min over the three horizon means (must be good everywhere)
for r in rows: r['robust'] = min(r['m5'], r['m25'], r['leg'])
top('robust', 'ROBUSTNESS (worst of 500m/250m/leg — must win everywhere)')
json.dump(rows, open("sweep_all_results.json", "w"))
print("\nwrote sweep_all_results.json")
