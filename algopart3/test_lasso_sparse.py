"""Forecast-construction test: replace ridge's dense L2 shrinkage with Lasso's sparse L1 shrinkage,
since we've established the TRUE cross-sectional structure is sparse (each stock has 1-3 real
leaders among 51 candidates, not a dense relationship to all 50). Lasso can select the few real
predictors instead of uniformly shrinking all 50 coefficients. Fit at periodic checkpoints (every 30
days, causal, exponentially-weighted via sample_weight) rather than every day, since Lasso needs a
separate optimization per target stock (no closed-form multi-output solution like ridge).
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Lasso
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
T = r.shape[1]
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250


def score_fn(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score_fn(tot.mean(), tot.std())}


HL = 500  # single half-life for this test (matching a "medium" ridge memory)
CHECKPOINTS = list(range(150, T, 30))


def fit_lasso_at(cp, alpha):
    """Weighted lasso per target stock, using history through day cp. Returns (50, 51) coef matrix
    + (50,) intercepts-equivalent (my), + predictor means (mx) matching SAFE's _ewls_ridge convention."""
    X = r[:, :cp - 1].T   # (n, 51)
    Y = r[1:, 1:cp].T     # (n, 50)
    n = X.shape[0]
    lam = 0.5 ** (1.0 / HL)
    w = lam ** np.arange(n - 1, -1, -1)
    mx = (w[:, None] * X).sum(0) / w.sum()
    my = (w[:, None] * Y).sum(0) / w.sum()
    Xc = X - mx
    B = np.zeros((51, 50))
    for j in range(50):
        yc = Y[:, j] - my[j]
        lasso = Lasso(alpha=alpha, max_iter=5000, tol=1e-4)
        lasso.fit(Xc, yc, sample_weight=w)
        B[:, j] = lasso.coef_
    return B, mx, my


def pooled_ic(preds_by_day, actual_by_day):
    X = np.concatenate(preds_by_day); Y = np.concatenate(actual_by_day)
    ok = ~np.isnan(X) & ~np.isnan(Y)
    return float(np.corrcoef(X[ok], Y[ok])[0, 1])


print("fitting periodic Lasso ensemble + ridge reference over the SAME checkpoints, several alphas ...")
for alpha in (1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4):
    lasso_preds = []; ridge_preds = []; actuals = []
    n_nonzero_total = 0; n_targets_total = 0
    for i, cp in enumerate(CHECKPOINTS[:-1]):
        nxt = CHECKPOINTS[i + 1]
        B, mx, my = fit_lasso_at(cp, alpha)
        n_nonzero_total += int((np.abs(B) > 1e-10).sum()); n_targets_total += 50
        Br, mxr, myr = SAFE._ewls_ridge(r[:, :cp - 1].T, r[1:, 1:cp].T, HL, SAFE.RIDGE_A)
        for t in range(cp, min(nxt, T - 1)):
            x = r[:, t]
            pred_l = my + (x - mx) @ B
            pred_r = myr + (x - mxr) @ Br
            lasso_preds.append(pred_l); ridge_preds.append(pred_r); actuals.append(r[1:, t + 1])
    ic_l = pooled_ic(lasso_preds, actuals); ic_r = pooled_ic(ridge_preds, actuals)
    avg_nonzero = n_nonzero_total / n_targets_total
    print(f"alpha={alpha}: Lasso IC={ic_l:.4f}  (avg {avg_nonzero:.1f}/51 nonzero coefs per target)   "
          f"ridge(hl={HL}) IC on same days={ic_r:.4f}")
