"""test_algo.py — is ALGO (inst 0, the index) predictable? Test on 400-750, split old vs new.
Autocorrelation, k-day reversion regression, variance ratio, and the tradeable zrev(5) Score
with a permutation null — per window."""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
lpA = np.log(prc[0])
ret = lpA[1:] - lpA[:-1]                      # ALGO daily log returns, len 749
RNG = np.random.default_rng(7)

def acf(x, lag):
    a, b = x[:-lag], x[lag:]
    n = len(a); r = np.corrcoef(a, b)[0, 1]
    return r, r * np.sqrt(n)                   # corr, approx t

def kday_reversion(r, k):
    """regress next-day return on the trailing k-day move; negative slope = mean reversion."""
    mv = np.array([r[i - k:i].sum() for i in range(k, len(r))])
    nxt = r[k:]
    b = np.polyfit(mv, nxt, 1)[0]
    # t-stat of slope
    res = nxt - np.polyval(np.polyfit(mv, nxt, 1), mv)
    se = np.sqrt((res @ res) / (len(mv) - 2) / ((mv - mv.mean()) @ (mv - mv.mean())))
    return b, b / se

def variance_ratio(r, k):
    n = len(r); mu = r.mean()
    va1 = ((r - mu) ** 2).sum() / n
    rk = np.array([r[i:i + k].sum() for i in range(n - k + 1)])
    vak = ((rk - k * mu) ** 2).sum() / (n - k + 1) / k
    return vak / va1                          # <1 mean-reverting, >1 trending, =1 random walk

def zrev_score(rseg, w=5, look=60, dol=100_000, comm=2e-5):
    """standalone ALGO zrev(w) book PnL score on a return segment (prices reconstructed)."""
    lp = np.concatenate([[0.0], np.cumsum(rseg)]); P = np.exp(lp) * 100.0
    cash = 0.0; cp = 0.0; val = 0.0; cm = 0.0; pll = []
    for t in range(look + w + 1, len(P)):
        cur = P[t - 1]
        mv = lp[w:t] - lp[:t - w]
        z = (mv[-1] - mv[-look:].mean()) / (mv[-look:].std() + 1e-12)
        pos = int(np.clip(-np.clip(z, -3, 3) / 3.0 * dol / cur, -dol / cur, dol / cur))
        d = pos - cp; cash -= cur * d + cm; cm = cur * abs(d) * comm; cp = pos
        pl = cash + cp * cur - val; val = cash + cp * cur
        pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu
    sr = np.sqrt(250) * mu / sd; return mu * sr ** 2 / (sr ** 2 + 1)

def perm_p(rseg, w=5, n=400):
    obs = zrev_score(rseg, w)
    null = np.array([zrev_score(rseg[RNG.permutation(len(rseg))], w) for _ in range(n)])
    return obs, null.mean(), np.percentile(null, 95), (null >= obs).mean()

windows = {"400-500 (old)": ret[399:499], "500-750 (new/graded)": ret[499:749], "400-750 (all)": ret[399:749]}
print("=== ALGO (index) predictability ===\n")
print("[1] Return autocorrelation (corr, ~t):")
for lbl, r in windows.items():
    cells = "  ".join(f"lag{k}:{acf(r,k)[0]:+.3f}(t{acf(r,k)[1]:+.1f})" for k in (1, 2, 5))
    print(f"  {lbl:<22} {cells}")

print("\n[2] k-day move -> next-day return (slope<0 = mean reversion; t):")
for lbl, r in windows.items():
    cells = "  ".join(f"k{k}:t={kday_reversion(r,k)[1]:+.2f}" for k in (3, 5, 10, 20))
    print(f"  {lbl:<22} {cells}")

print("\n[3] Variance ratio VR(k)  (<1 revert, =1 random walk, >1 trend):")
for lbl, r in windows.items():
    cells = "  ".join(f"VR{k}:{variance_ratio(r,k):.2f}" for k in (2, 5, 10))
    print(f"  {lbl:<22} {cells}")

print("\n[4] Tradeable zrev(5) ALGO book Score + permutation null (is the edge real & monetizable?):")
for lbl, r in windows.items():
    obs, nm, n95, p = perm_p(r)
    star = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else " ns"))
    print(f"  {lbl:<22} Score={obs:7.1f}  null_mean={nm:6.1f}  null95={n95:6.1f}  p={p:.3f} {star}")
