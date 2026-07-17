"""Copula dependence between instruments.

Libraries: statsmodels copulas (Gaussian, Student-t, Clayton, Gumbel, Frank),
scipy for rank transforms.

  - Empirical upper/lower TAIL dependence for all 1275 pairs (do names crash
    together more than they rally together? = lower vs upper asymmetry).
  - Copula-family AIC selection (which dependence structure fits) on the
    cointegrated + most-correlated pairs.
  - Kendall's tau matrix summary.
"""
import warnings, itertools, os
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.distributions.copula.api import (GaussianCopula, StudentTCopula,
    ClaytonCopula, GumbelCopula, FrankCopula)
from common import load, log_returns, section, Recorder, RESULTS

rec = Recorder("copula_dependence")
df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape

def pseudo(x):
    return stats.rankdata(x) / (len(x) + 1)

def tail_dep(u, v, q):
    """Empirical lower (q<0.5) tail dependence coefficient."""
    if q < 0.5:
        return np.mean((u < q) & (v < q)) / q
    return np.mean((u > q) & (v > q)) / (1 - q)

section("18A. EMPIRICAL TAIL DEPENDENCE - all 1275 pairs (lower vs upper)")
U = np.column_stack([pseudo(R[:, i]) for i in range(N)])
rows = []
for i, j in itertools.combinations(range(N), 2):
    u, v = U[:, i], U[:, j]
    lo = tail_dep(u, v, 0.10); hi = tail_dep(u, v, 0.90)
    tau = stats.kendalltau(u, v)[0]
    rows.append([tickers[i], tickers[j], tau, lo, hi, lo - hi])
td = pd.DataFrame(rows, columns=["a","b","kendall_tau","lower_TD","upper_TD","asymmetry"])
print(f"Across 1275 pairs: mean lower-tail dep {td.lower_TD.mean():.3f}, "
      f"upper {td.upper_TD.mean():.3f}")
print(f"Mean asymmetry (lower-upper): {td.asymmetry.mean():+.3f} "
      f"({'crash-together bias' if td.asymmetry.mean()>0.02 else 'symmetric - no crash bias'})")
# significance of asymmetry across pairs
tstat, tp = stats.ttest_1samp(td.asymmetry, 0)
print(f"t-test asymmetry vs 0: t={tstat:.2f} p={tp:.3g}")
rec.add("scipy","copula","mean_lower_tail_dep","ALL",td.lower_TD.mean(),np.nan)
rec.add("scipy","copula","mean_upper_tail_dep","ALL",td.upper_TD.mean(),np.nan)
rec.add("scipy","copula","tail_asymmetry_ttest","ALL",tstat,tp)
print("\nHighest lower-tail dependence pairs:")
print(td.sort_values("lower_TD",ascending=False).head(8).round(3).to_string(index=False))

section("18B. COPULA-FAMILY AIC SELECTION (cointegrated + top-corr pairs)")
idx = {t: k for k, t in enumerate(tickers)}
subset = [("AENO","NWIG"),("EORC","NGTE"),("HETT","ULXY"),("SMAH","ILVX"),
          ("HUXZ","ACAC"),("CTGI","EELT")]
# add the 4 most correlated pairs by |tau|
for _, r in td.reindex(td.kendall_tau.abs().sort_values(ascending=False).index).head(4).iterrows():
    subset.append((r.a, r.b))

def fit_aic(u, v):
    tau = stats.kendalltau(u, v)[0]
    data = np.column_stack([u, v])
    out = {}
    # Gaussian
    try:
        rho = 2*np.sin(np.pi*tau/6)  # not exact but init; use pearson of normal scores
        rho = np.corrcoef(stats.norm.ppf(u), stats.norm.ppf(v))[0,1]
        c = GaussianCopula(rho); out["Gaussian"] = (c.logpdf(data).sum(), 1)
    except Exception: pass
    try:
        rho = np.corrcoef(stats.norm.ppf(u), stats.norm.ppf(v))[0,1]
        c = StudentTCopula(rho, df=5); out["StudentT"] = (c.logpdf(data).sum(), 2)
    except Exception: pass
    for name, Cls in [("Clayton",ClaytonCopula),("Gumbel",GumbelCopula),("Frank",FrankCopula)]:
        try:
            c = Cls(); th = c.theta_from_tau(tau); c = Cls(th)
            out[name] = (c.logpdf(data).sum(), 1)
        except Exception: pass
    aic = {k: 2*p - 2*ll for k,(ll,p) in out.items()}
    return aic

print(f"{'pair':<12}{'best_copula':<12}{'tau':>7}  AIC by family")
fam_wins = {}
for a, b in subset:
    u, v = U[:, idx[a]], U[:, idx[b]]
    aic = fit_aic(u, v)
    if not aic: continue
    best = min(aic, key=aic.get); fam_wins[best] = fam_wins.get(best,0)+1
    tau = stats.kendalltau(u, v)[0]
    rec.add("statsmodels","copula",f"best_{best}",f"{a}-{b}",aic[best],np.nan)
    astr = " ".join(f"{k}={v:.0f}" for k,v in sorted(aic.items(), key=lambda x:x[1]))
    print(f"{a+'-'+b:<12}{best:<12}{tau:>7.3f}  {astr}")
print(f"\nBest-fitting copula family counts: {fam_wins}")

section("18C. VERDICT")
print(f"Mean tail asymmetry {td.asymmetry.mean():+.3f} (t={tstat:.2f}): "
      f"{'significant crash co-movement' if tp<0.05 and td.asymmetry.mean()>0 else 'NO significant tail asymmetry'}")
print("Best copula family indicates the dependence type; Gaussian/Frank winning")
print("=> symmetric, no special tail clustering (consistent with Gaussian returns).")
td.to_csv(os.path.join(RESULTS, "copula_tail_dependence.csv"), index=False)
rec.save()
