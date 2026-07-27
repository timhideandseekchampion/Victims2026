"""Different stat-arb angle: is a proper GARCH(1,1) conditional-volatility estimate a better feature
for the ALGO vol-switch signal than the current rolling-20-day std? The 'arch' package isn't
reliably importable in this venv, so GARCH(1,1) is fit by hand via MLE (standard, well-understood
model: sigma^2_t = omega + alpha*r_{t-1}^2 + beta*sigma^2_{t-1}). Fit ONCE on the full history for a
quick "is this feature even better in principle" check, then (if promising) redo properly causally.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r = np.diff(lpA)
ret1 = np.full(len(lpA), np.nan); ret1[:-1] = lpA[1:] - lpA[:-1]


def garch_negloglik(params, x):
    omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1: return 1e10
    n = len(x)
    sigma2 = np.empty(n)
    sigma2[0] = np.var(x)
    for t in range(1, n):
        sigma2[t] = omega + alpha * x[t - 1] ** 2 + beta * sigma2[t - 1]
    ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + x ** 2 / sigma2)
    return -ll


x = r - r.mean()
x0 = [np.var(x) * 0.05, 0.05, 0.90]
res = minimize(garch_negloglik, x0, args=(x,), method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10})
omega, alpha, beta_g = res.x
print(f"GARCH(1,1) fit (full sample): omega={omega:.2e} alpha={alpha:.4f} beta={beta_g:.4f} "
      f"persistence={alpha+beta_g:.4f}  converged={res.success}")

n = len(x)
sigma2 = np.empty(n)
sigma2[0] = np.var(x)
for t in range(1, n):
    sigma2[t] = omega + alpha * x[t - 1] ** 2 + beta_g * sigma2[t - 1]
garch_vol = np.sqrt(sigma2)

T = len(lpA)
garch_vol_full = np.full(T, np.nan); garch_vol_full[1:] = garch_vol

roll_vol = np.full(T, np.nan)
roll_vol[M.VOL_WIN:] = M._roll_std(r, M.VOL_WIN)


def zscore_series(vol, Z):
    out = np.full(T, np.nan)
    for s in range(Z + 5, T):
        w = vol[s - Z:s]
        ok = ~np.isnan(w)
        if ok.sum() < Z // 2: continue
        out[s] = (vol[s] - w[ok].mean()) / (w[ok].std() + 1e-12)
    return out


def xs_ic_report(volz, label):
    xs = volz[:-1]; ys = ret1[:-1]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    icf = np.corrcoef(xs[ok], ys[ok])[0, 1]
    half = ok.sum() // 2
    idxs = np.where(ok)[0]
    h1 = idxs[:half]; h2 = idxs[half:]
    ic1 = np.corrcoef(volz[h1], ret1[h1])[0, 1]
    ic2 = np.corrcoef(volz[h2], ret1[h2])[0, 1]
    print(f"  {label:<28} full IC={icf:+.4f}   H1={ic1:+.4f}  H2={ic2:+.4f}")


print("\ncross-sectional (single-series, ALGO) vol->next-return IC, rolling-std vs GARCH:")
for Z in (60,):
    volz_roll = zscore_series(roll_vol, Z)
    xs_ic_report(volz_roll, f"rolling-20d-std, z-win={Z}")
volz_garch = zscore_series(garch_vol_full, 60)
xs_ic_report(volz_garch, "GARCH(1,1) cond. vol, z-win=60")
# also raw GARCH vol level (not z-scored) vs return, and GARCH vol vs rolling vol correlation
ok2 = ~np.isnan(garch_vol_full[:-1]) & ~np.isnan(roll_vol[:-1])
print(f"\n  corr(GARCH cond.vol, rolling-20d std): {np.corrcoef(garch_vol_full[:-1][ok2], roll_vol[:-1][ok2])[0,1]:.3f}")
print(f"  GARCH alpha+beta (persistence) = {alpha+beta_g:.4f}  (close to 1 => genuine vol clustering; close to 0 => none)")
