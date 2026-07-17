"""
ic_hunt.py — how high can the cross-sectional IC go on days 400-750? Target: IC ~0.10.

IC = mean over days of the cross-sectional Pearson corr between a signal (known at end of day d)
and the realized next-day return of the 50 idio names. Reports mean IC + t-stat, all CAUSAL.

Also computes the ORACLE ceiling: fit the linear next-day predictor on the WHOLE window with
look-ahead (best possible in-sample) — an upper bound no causal signal can beat. If even the
oracle can't reach 0.10, then 0.10 is not attainable on this data (a hard fact).
"""
import itertools
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]            # (51, nt-2..) col d = move into day d+1
S, E = 400, 749                                           # focus window (signal days)

def ewls(X, Y, hl, a):
    n, p = X.shape; lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY), mx, my

def ic_stats(sig_fn):
    """sig_fn(d) -> 50-vec signal known at end of day d; scored vs RET[1:, d] (next-day move)."""
    ics = []
    for d in range(S, E):
        s = sig_fn(d)
        if s is None: continue
        fwd = RET[1:, d]
        if s.std() < 1e-12 or fwd.std() < 1e-12: continue
        ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics)
    if len(ics) < 5: return np.nan, np.nan, 0
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics))), len(ics)

# ---- causal ridge forecast (cache per (hl,a,pred,d)) --------------------------------
_c = {}
def ridge(d, hl, a, pred="full"):
    if d < 95: return None
    key = (hl, a, pred, d)
    if key in _c: return _c[key]
    X = RET[:, :d - 1].T; Y = RET[1:, 1:d].T                 # predictor at tau -> target at tau+1
    xin = RET[:, d - 1]
    if pred == "no_algo": X = X[:, 1:]; xin = xin[1:]
    B, mx, my = ewls(X, Y, hl, a)
    pred_v = my + (xin - mx) @ B
    v = pred_v - pred_v.mean(); _c[key] = v; return v

def revz(d, w):
    if d < w + 1: return None
    r = lp[1:, d] - lp[1:, d - w]; r = r - r.mean()
    return -r / (r.std() + 1e-12)

print(f"IC battery on days {S}-{E}  (mean daily cross-sectional IC vs next-day return)\n")

# 1. ridge hyperparameter sweep
print("[1] LEAD-LAG RIDGE sweep")
print(f"{'hl':>6}{'alpha':>7}{'pred':>9}{'IC':>9}{'t':>7}")
best_ridge = None
for hl, a, pr in itertools.product([250, 500, 1000, 2000], [0.03, 0.1, 0.3, 1.0], ["full", "no_algo"]):
    ic, t, n = ic_stats(lambda d, hl=hl, a=a, pr=pr: ridge(d, hl, a, pr))
    if best_ridge is None or ic > best_ridge[0]:
        best_ridge = (ic, t, dict(hl=hl, a=a, pred=pr))
    if a in (0.1,) and pr == "full":
        print(f"{hl:>6}{a:>7}{pr:>9}{ic:>9.4f}{t:>7.2f}")
print(f"  -> BEST ridge IC = {best_ridge[0]:.4f} (t={best_ridge[1]:.2f})  {best_ridge[2]}")
bh, ba, bp = best_ridge[2]["hl"], best_ridge[2]["a"], best_ridge[2]["pred"]

# 2. reversion horizons
print("\n[2] CROSS-SECTIONAL REVERSION")
for w in [5, 10, 20, 40]:
    ic, t, n = ic_stats(lambda d, w=w: revz(d, w))
    print(f"  revz({w:>2})  IC={ic:.4f}  t={t:.2f}")

# 3. lead-lag + reversion blend
print("\n[3] BLEND  best-ridge + lambda*revz(5)  (combined signal IC)")
best_blend = (best_ridge[0], 0.0)
for lam in [0.0, 0.1, 0.2, 0.3, 0.5]:
    def blend(d, lam=lam):
        r = ridge(d, bh, ba, bp); z = revz(d, 5)
        if r is None or z is None: return None
        rr = r / (r.std() + 1e-12)
        return (1 - lam) * rr + lam * z
    ic, t, n = ic_stats(blend)
    if ic > best_blend[0]: best_blend = (ic, lam)
    print(f"  lambda={lam:>4}  IC={ic:.4f}  t={t:.2f}")

# 4. multi-HL ensemble
print("\n[4] MULTI-HL ENSEMBLE  (avg of z-scored ridge forecasts across HLs)")
def ens(d):
    fs = []
    for hl in [250, 500, 1000, 2000]:
        r = ridge(d, hl, ba, bp)
        if r is None: return None
        fs.append(r / (r.std() + 1e-12))
    return np.mean(fs, 0)
ic, t, n = ic_stats(ens)
print(f"  ensemble IC={ic:.4f}  t={t:.2f}")

# 5. multi-HL ensemble + reversion blend
def ens_blend(d, lam=best_blend[1] if best_blend[1] > 0 else 0.2):
    e = ens(d); z = revz(d, 5)
    if e is None or z is None: return None
    return (1 - lam) * e + lam * z
ic, t, n = ic_stats(ens_blend)
print(f"  ensemble+revz(5) blend IC={ic:.4f}  t={t:.2f}")

# 6. ORACLE ceiling — fit predictor on the WHOLE window (look-ahead) = upper bound
print("\n[6] ORACLE CEILING (look-ahead in-sample fit — upper bound for ANY linear predictor)")
Xall = RET[:, S - 1:E - 1].T                              # predictors over window
Yall = RET[1:, S:E].T                                     # next-day targets
for a in [0.0, 0.1, 1.0]:
    n, p = Xall.shape
    XtX = Xall.T @ Xall; XtY = Xall.T @ Yall
    B = np.linalg.solve(XtX + a * np.eye(p), XtY)
    ics = []
    for i, d in enumerate(range(S, E)):
        f = (RET[:, d - 1] @ B); f = f - f.mean()
        fwd = RET[1:, d]
        if f.std() > 0: ics.append(np.corrcoef(f, fwd)[0, 1])
    ics = np.array(ics)
    print(f"  oracle ridge(a={a}):  in-sample IC={ics.mean():.4f}  (t={ics.mean()/(ics.std(ddof=1)/np.sqrt(len(ics))):.1f})")

# predictable-fraction ceiling: per-name, corr of fitted vs realized (time-series, in-sample)
n, p = Xall.shape
B = np.linalg.solve(Xall.T @ Xall + 0.1 * np.eye(p), Xall.T @ Yall)
fit = Xall @ B
r2 = 1 - ((Yall - fit) ** 2).sum() / ((Yall - Yall.mean(0)) ** 2).sum()
print(f"  pooled in-sample R^2 of next-day return ~ {r2:.4f}  (predictable fraction; oracle upper bound)")

print("\nSUMMARY")
print(f"  best causal single signal (ridge): IC {best_ridge[0]:.4f}")
print(f"  best causal blend:                 IC {best_blend[0]:.4f} (lambda={best_blend[1]})")
print("  => compare against the 0.10 target and the oracle ceiling above.")
