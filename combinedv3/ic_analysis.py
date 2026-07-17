"""Which SIGNAL actually had cross-sectional edge (IC) on 500-750 vs 250-500? If a different
signal (momentum, a specific reversion horizon, pairs) had notably higher IC on 500-750 than our
ridge, that's how a team could have scored higher — and a real, missable edge. If our ridge has
the highest IC on both, there's no better signal to find."""
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); ret = lp[:, 1:] - lp[:, :-1]            # (51, nt-1)


def ewls(X, Y, hl=500, a=0.1):
    n, p = X.shape; lam = 0.5**(1.0/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc, Yc = X-mx, Y-my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    return np.linalg.solve(XtWX+(eps+a)*np.eye(p), XtWY), mx, my


def daily_ics(signal_fn, dstart, dend):
    """mean & t-stat of daily cross-sectional IC (signal vs next-day return) over [dstart,dend]."""
    ics = []
    for d in range(dstart, dend):
        sig = signal_fn(d)                                # signal on the 50 names, known at end of day d
        fwd = ret[1:, d]                                  # realized next-day return (ret[:,d] = day d+1 move)
        if sig is None or sig.std() < 1e-12 or fwd.std() < 1e-12: continue
        ics.append(np.corrcoef(sig, fwd)[0, 1])
    ics = np.array(ics)
    return ics.mean(), ics.mean()/(ics.std()/np.sqrt(len(ics))+1e-12), len(ics)


# signals (all causal: use data through column d-1 to predict ret[:,d])
def ridge_sig(hl=500):
    cache = {}
    def f(d):
        if d < 95: return None
        key = d
        if key not in cache:
            X = ret[:, :d-1].T; Y = ret[1:, 1:d].T
            B, mx, my = ewls(X, Y, hl)
            cache[key] = (B, mx, my)
        B, mx, my = cache[key]; pred = my + (ret[:, d-1]-mx) @ B
        return pred - pred.mean()
    return f

def rev(h):
    def f(d):
        if d < h+1: return None
        r = ret[1:, d-h:d].sum(1); return -(r - r.mean())
    return f

def mom(h):
    def f(d):
        if d < h+1: return None
        r = ret[1:, d-h:d].sum(1); return (r - r.mean())
    return f

windows = {"250-500": (250, 499), "500-750": (nt-250, nt-1)}
sigs = {"ridge HL500": ridge_sig(500), "ridge HL1000": ridge_sig(1000),
        "rev-1d": rev(1), "rev-3d": rev(3), "rev-5d": rev(5), "rev-10d": rev(10), "rev-20d": rev(20),
        "mom-5d": mom(5), "mom-20d": mom(20)}

print(f"Mean daily cross-sectional IC (t-stat) by signal and window:\n")
print(f"{'signal':16}" + "".join(f"{w:>22}" for w in windows))
for name, fn in sigs.items():
    row = f"{name:16}"
    for w, (a, b) in windows.items():
        if "ridge" in name: fn = ridge_sig(500 if "500" in name else 1000)   # fresh cache per window
        ic, t, n = daily_ics(fn, a, b)
        row += f"    IC {ic:+.4f} (t {t:+.1f})"
    print(row)
print("\nHigher IC on 500-750 than the ridge => a signal a top team could have used there.")
