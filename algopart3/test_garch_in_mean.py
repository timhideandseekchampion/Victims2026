"""GARCH-IN-MEAN: fit a joint model where the conditional variance feeds directly into the
conditional MEAN equation (the formal version of the "vol risk premium" hypothesis SAFE_llvol.py's
docstring already describes informally):
    r_t = mu + lambda * sigma_t + eps_t,      eps_t = sigma_t * z_t
    sigma_t^2 = omega + alpha * eps_{t-1}^2 + beta * sigma_{t-1}^2
This is different from the earlier test (which only modeled volatility as a separate FEATURE for a
heuristic switch) -- here lambda is estimated JOINTLY with the variance dynamics, directly testing
whether elevated conditional vol predicts the return itself, in one coherent model.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r = np.diff(lpA)   # ALGO's daily log returns, length nt-1


def garchm_negloglik(params, x):
    mu, lam, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1: return 1e10
    n = len(x)
    sigma2 = np.empty(n); eps = np.empty(n)
    sigma2[0] = np.var(x)
    eps[0] = x[0] - mu
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = x[t] - mu - lam * np.sqrt(sigma2[t])
    if np.any(sigma2 <= 0) or not np.all(np.isfinite(sigma2)): return 1e10
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps ** 2 / sigma2)
    if not np.isfinite(ll): return 1e10
    return -ll


def fit_garchm(x):
    x0 = [x.mean(), 0.0, np.var(x) * 0.05, 0.05, 0.90]
    res = minimize(garchm_negloglik, x0, args=(x,), method="Nelder-Mead",
                    options={"maxiter": 8000, "xatol": 1e-9, "fatol": 1e-9})
    return res


print("fitting GARCH(1,1)-in-mean on ALGO's full-sample returns ...")
res = fit_garchm(r)
mu, lam, omega, alpha, beta_g = res.x
print(f"  mu={mu:.6f}  lambda={lam:.4f}  omega={omega:.2e}  alpha={alpha:.4f}  beta={beta_g:.4f}  "
      f"persistence={alpha+beta_g:.4f}  converged={res.success}")
print(f"  (lambda = the vol-in-mean coefficient: r_t = mu + lambda*sigma_t + eps_t. "
      f"lambda>0 means elevated vol -> higher expected return, matching the shipped hypothesis)")

half = len(r) // 2
res1 = fit_garchm(r[:half]); res2 = fit_garchm(r[half:])
print(f"\n  H1 fit: lambda={res1.x[1]:+.4f}  persistence={res1.x[3]+res1.x[4]:.4f}")
print(f"  H2 fit: lambda={res2.x[1]:+.4f}  persistence={res2.x[3]+res2.x[4]:.4f}")

quarters = np.array_split(r, 4)
print("  quarter-by-quarter lambda:")
for qi, q in enumerate(quarters):
    rq = fit_garchm(q)
    print(f"    Q{qi+1}: lambda={rq.x[1]:+.4f}  n={len(q)}")
