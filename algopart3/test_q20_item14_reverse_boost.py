"""Item 14 [SIGNAL]: reverse-direction boost check. For each stock J's identified significant
leader I (full-sample significance test, same mechanism as the shipped boost), does J's OWN
lagged return ALSO predict I's future return? Much lower bar than "reciprocal pairs" (already
rejected), which required I and J to be MUTUAL best matches -- this only requires the reverse
causal direction to exist at all, for whichever I already leads J.
"""
import numpy as np, pandas as pd
from scipy import stats

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
n, T = rs.shape

ALPHA = 0.05
N_CANDIDATES = 49


def sig_threshold(n_samples):
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


Xi = rs[:, :-1]; Yj = rs[:, 1:]
Xc = Xi - Xi.mean(1, keepdims=True); Yc = Yj - Yj.mean(1, keepdims=True)
Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
C = (Xs @ Ys.T) / Xi.shape[1]
thr = sig_threshold(Xi.shape[1])

pairs = []
for j in range(n):
    col = C[:, j].copy(); col[j] = np.nan
    i = int(np.nanargmax(np.abs(col)))
    if abs(col[i]) > thr:
        pairs.append((i, j, float(col[i])))
print(f"significant full-sample leader pairs (I leads J): {len(pairs)} found (threshold={thr:.4f})")

fwd_ics, rev_ics = [], []
rng = np.random.default_rng(0)
for i, j, ic_fwd in pairs:
    # reverse: does J's lagged return predict I's future return?
    xr = rs[j, :-1]; yr = rs[i, 1:]
    ic_rev = float(np.corrcoef(xr, yr)[0, 1])
    fwd_ics.append(ic_fwd); rev_ics.append(ic_rev)

fwd_ics = np.array(fwd_ics); rev_ics = np.array(rev_ics)
print(f"forward IC:  mean={fwd_ics.mean():+.4f}  |mean|={np.abs(fwd_ics).mean():.4f}")
print(f"reverse IC:  mean={rev_ics.mean():+.4f}  |mean|={np.abs(rev_ics).mean():.4f}")
same_sign = (np.sign(fwd_ics) == np.sign(rev_ics)).mean()
print(f"same-sign fraction (fwd vs rev): {same_sign:.2f}  (0.5 = coin flip / no relationship)")

# permutation test: is the reverse IC magnitude bigger than chance for THESE specific (i,j) pairs?
n_perm = 2000
null_abs_rev = np.empty(n_perm)
for p in range(n_perm):
    vals = []
    for i, j, _ in pairs:
        shift = rng.integers(1, T - 1)
        xr = np.roll(rs[j, :-1], shift)
        yr = rs[i, 1:]
        vals.append(np.corrcoef(xr, yr)[0, 1])
    null_abs_rev[p] = np.mean(np.abs(vals))
obs_abs_rev = np.abs(rev_ics).mean()
pval = float((null_abs_rev >= obs_abs_rev).mean())
print(f"perm test: observed mean|reverse IC|={obs_abs_rev:.4f} vs null mean={null_abs_rev.mean():.4f}  p={pval:.3f}")

n_sig_rev = int((np.abs(rev_ics) > thr).sum())
print(f"reverse pairs individually clearing the SAME significance threshold: {n_sig_rev}/{len(pairs)}")
