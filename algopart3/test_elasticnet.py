"""Elastic Net: blend L1 (Lasso's sparsity/selection) and L2 (ridge's dense shrinkage) to see if a
combination beats either pure approach. Same periodic-checkpoint, exponentially-weighted setup as
the Lasso test, same ridge reference for direct comparison.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
T = r.shape[1]

HL = 500
CHECKPOINTS = list(range(150, T, 30))


def fit_enet_at(cp, alpha, l1_ratio):
    X = r[:, :cp - 1].T
    Y = r[1:, 1:cp].T
    n = X.shape[0]
    lam = 0.5 ** (1.0 / HL)
    w = lam ** np.arange(n - 1, -1, -1)
    mx = (w[:, None] * X).sum(0) / w.sum()
    my = (w[:, None] * Y).sum(0) / w.sum()
    Xc = X - mx
    B = np.zeros((51, 50))
    for j in range(50):
        yc = Y[:, j] - my[j]
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, tol=1e-4)
        m.fit(Xc, yc, sample_weight=w)
        B[:, j] = m.coef_
    return B, mx, my


def pooled_ic(preds_by_day, actual_by_day):
    X = np.concatenate(preds_by_day); Y = np.concatenate(actual_by_day)
    ok = ~np.isnan(X) & ~np.isnan(Y)
    return float(np.corrcoef(X[ok], Y[ok])[0, 1])


print("computing ridge reference on the SAME checkpoints/days ...")
ridge_preds = []; actuals = []
for i, cp in enumerate(CHECKPOINTS[:-1]):
    nxt = CHECKPOINTS[i + 1]
    Br, mxr, myr = SAFE._ewls_ridge(r[:, :cp - 1].T, r[1:, 1:cp].T, HL, SAFE.RIDGE_A)
    for t in range(cp, min(nxt, T - 1)):
        x = r[:, t]
        ridge_preds.append(myr + (x - mxr) @ Br); actuals.append(r[1:, t + 1])
ic_ridge = pooled_ic(ridge_preds, actuals)
print(f"ridge(hl={HL}) IC on these days: {ic_ridge:.4f}\n")

print("sweeping ElasticNet alpha x l1_ratio ...")
print(f"{'alpha':>10}{'l1_ratio':>10}{'IC':>9}{'avg_nonzero':>13}")
best = None
for alpha in (5e-6, 1e-5, 2e-5, 5e-5):
    for l1_ratio in (0.1, 0.3, 0.5, 0.7, 0.9):
        enet_preds = []; actuals2 = []; n_nonzero = 0; n_targets = 0
        for i, cp in enumerate(CHECKPOINTS[:-1]):
            nxt = CHECKPOINTS[i + 1]
            B, mx, my = fit_enet_at(cp, alpha, l1_ratio)
            n_nonzero += int((np.abs(B) > 1e-10).sum()); n_targets += 50
            for t in range(cp, min(nxt, T - 1)):
                x = r[:, t]
                enet_preds.append(my + (x - mx) @ B); actuals2.append(r[1:, t + 1])
        ic = pooled_ic(enet_preds, actuals2)
        avg_nz = n_nonzero / n_targets
        print(f"{alpha:>10}{l1_ratio:>10}{ic:>9.4f}{avg_nz:>13.1f}")
        if best is None or ic > best[0]:
            best = (ic, alpha, l1_ratio, avg_nz)

print(f"\nbest ElasticNet: IC={best[0]:.4f} at alpha={best[1]}, l1_ratio={best[2]} (avg {best[3]:.1f}/51 nonzero)")
print(f"ridge reference: IC={ic_ridge:.4f}")
