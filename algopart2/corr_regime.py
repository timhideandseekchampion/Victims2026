"""corr_regime.py — the one untested industry idea from Vol/RV, Multi-Factor, Macro/Carry that this
market CAN express: a realized CORRELATION / DISPERSION regime detector (dispersion-arb family, but
used as a regime validator, not tradeable vol-arb). Does the cross-sectional correlation structure
carry OUT-OF-SAMPLE information about the CHAMPION's future edge health that the book doesn't already
get from watching its own realized IC? If yes -> a second validator axis (orthogonal to xsac's
autocorrelation-sign axis). If no -> confirmed dead, same lesson as HMM/k-means (watch IC directly).
"""
import numpy as np, pandas as pd
import SAFE_rotate as R

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
logp = np.log(prc); r = logp[:, 1:] - logp[:, :-1]        # (51, nDays-1), col j = return into day j+1
W = 40

def rho_bar(t):        # avg pairwise correlation of the 50 names over trailing W (causal, uses <= t-1)
    Rw = r[1:, t - 1 - W:t - 1]
    if Rw.shape[1] < W: return None
    C = np.corrcoef(Rw); iu = np.triu_indices(50, 1); return float(C[iu].mean())

def volratio(t):       # index vol / avg name vol = the dispersion-trade RV metric (low = decorrelated)
    Rw = r[:, t - 1 - W:t - 1]
    if Rw.shape[1] < W: return None
    return float(Rw[0].std() / (Rw[1:].std(1).mean() + 1e-12))

def factorR2(t):       # share of name variance explained by the index (correlation-structure proxy)
    Rw = r[:, t - 1 - W:t - 1]
    if Rw.shape[1] < W: return None
    x = Rw[0] - Rw[0].mean(); vx = x @ x + 1e-12
    r2 = []
    for i in range(1, 51):
        y = Rw[i] - Rw[i].mean(); b = (x @ y) / vx
        ss = y @ y + 1e-12; r2.append(1 - ((y - b * x) @ (y - b * x)) / ss)
    return float(np.mean(r2))

# champion realized IC series (full cache)
R._ensure_cache(prc)
icc = {n: R._ic1("champ", n) for n in range(R.WARMUP, nDays - 1)}
icm = {n: R._ic1("momJT", n) for n in range(R.WARMUP, nDays - 1)}

def oos(feat, name, H=20):
    days = [t for t in range(W + 2, nDays - 1) if feat(t) is not None and t in icc]
    F = np.array([feat(t) for t in days]); dd = np.array(days)
    fut_champ = np.array([np.mean([icc[n] for n in range(t, t + H) if n in icc]) for t in dd])
    fut_diff  = np.array([np.mean([icm[n] - icc[n] for n in range(t, t + H) if n in icc]) for t in dd])
    ok = ~(np.isnan(fut_champ) | np.isnan(fut_diff)); dd, F, fut_champ, fut_diff = dd[ok], F[ok], fut_champ[ok], fut_diff[ok]
    n = len(dd); mid = n // 2
    thr = np.median(F[:mid])                                  # split on TRAIN median only
    te = slice(mid, n); hi = F[te] > thr
    fc, fd = fut_champ[te], fut_diff[te]
    print(f"  {name:<10} range[{F.min():+.3f},{F.max():+.3f}] med {thr:+.3f} | "
          f"future CHAMP IC  hi={fc[hi].mean():+.4f} lo={fc[~hi].mean():+.4f} SEP={abs(fc[hi].mean()-fc[~hi].mean()):.4f}"
          f" | future(momJT-champ) hi={fd[hi].mean():+.4f} lo={fd[~hi].mean():+.4f}")

print(f"(A) REAL data OOS — does the correlation regime predict future edge health? (H=20, W={W})")
print(f"    real-data champ IC mean = {np.mean(list(icc.values())):+.4f}")
for feat, name in [(rho_bar, "rho_bar"), (volratio, "volratio"), (factorR2, "factorR2")]:
    oos(feat, name)

# (B) does the correlation feature even MOVE in a momentum regime? (is it a viable detector at all)
def make_mom(D=550, mom=0.7, seed=1):
    rng = np.random.default_rng(seed); lp = np.log(prc[:, :D + 1]).copy()
    vol = r[1:].std(); names = lp[1:].copy()
    for _ in range(nDays - D):
        tr = names[:, -1] - names[:, -5]; tc = tr - tr.mean(); dr = mom * (tc / (tc.std() + 1e-9)) * vol; dr -= dr.mean()
        no = rng.normal(0, vol, 50); no -= no.mean(); names = np.concatenate([names, (names[:, -1] + dr + no)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :D + 1] = prc[:, :D + 1]; return full

full = make_mom(); rf = np.log(full)[:, 1:] - np.log(full)[:, :-1]
def rho_bar_f(t, RR):
    Rw = RR[1:, t - 1 - W:t - 1]; C = np.corrcoef(Rw); iu = np.triu_indices(50, 1); return C[iu].mean()
D = 550
pre = np.mean([rho_bar_f(t, rf) for t in range(D - 60, D)]); post = np.mean([rho_bar_f(t, rf) for t in range(D + 20, D + 100)])
print(f"\n(B) synthetic momentum regime (D={D}): avg pairwise corr rho_bar  pre {pre:+.3f} -> post {post:+.3f}"
      f"  (shift {post-pre:+.3f}; xsac shifted ~+0.35 for reference)")
print("    -> is correlation a viable regime detector here, or does the regime hide from it?")
