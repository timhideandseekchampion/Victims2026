"""FULL scipy.stats battery. Every applicable test, one row each in the recorder.

Families:
  - Normality/distribution GoF (7 tests x 51)
  - Skewness & kurtosis tests
  - Distribution fitting (10 families, KS goodness-of-fit) -> best fit
  - Trend estimators (OLS, Theil-Sen, Siegel) on prices
  - Correlation of each name vs ALGO & vs market (Pearson/Spearman/Kendall)
  - Two-sample regime tests first-half vs second-half returns (12 tests)
  - Sign / randomness (binomtest up-days, custom runs, autocorr-of-sign)
  - Contingency of sign transitions (chi2, G-test, Fisher/Barnard/Boschloo)
  - Entropy & differential entropy
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from scipy import stats
from common import load, log_returns, section, stars, Recorder

rec = Recorder("scipy_full")
df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape
mkt = R.mean(axis=1)
algo = R[:, 0]

section("10A. NORMALITY / DISTRIBUTION GOODNESS-OF-FIT (per instrument)")
for i, t in enumerate(tickers):
    r = R[:, i]
    z = (r - r.mean()) / r.std(ddof=0)
    tests = {
        "shapiro":        stats.shapiro(r),
        "normaltest_K2":  stats.normaltest(r),
        "jarque_bera":    stats.jarque_bera(r),
        "kstest_norm":    stats.kstest(z, "norm"),
        "cramervonmises": (lambda c: (c.statistic, c.pvalue))(stats.cramervonmises(z, "norm")),
        "skewtest":       stats.skewtest(r),
        "kurtosistest":   stats.kurtosistest(r),
    }
    for name, res in tests.items():
        s, p = (res.statistic, res.pvalue) if hasattr(res, "statistic") else res
        rec.add("scipy", "normality", name, t, s, p)
    ad = stats.anderson(r, "norm")
    rec.add("scipy", "normality", "anderson_darling", t, ad.statistic, np.nan,
            note=f"crit5%={ad.critical_values[2]:.3f} reject={ad.statistic>ad.critical_values[2]}")

section("10B. DISTRIBUTION FITTING - which family fits returns best (KS)")
families = ["norm", "t", "laplace", "cauchy", "logistic", "skewnorm",
            "johnsonsu", "gennorm", "hypsecant", "genhyperbolic"]
best_counts = {}
for i, t in enumerate(tickers):
    r = R[:, i]
    best, bestp = None, -1
    for fam in families:
        dist = getattr(stats, fam)
        try:
            params = dist.fit(r)
            ks_s, ks_p = stats.kstest(r, fam, args=params)
            rec.add("scipy", "dist_fit", f"ks_{fam}", t, ks_s, ks_p)
            if ks_p > bestp:
                best, bestp = fam, ks_p
        except Exception:
            pass
    best_counts[best] = best_counts.get(best, 0) + 1
print("Best-fitting distribution family across 51 instruments (highest KS p):")
for fam, c in sorted(best_counts.items(), key=lambda x: -x[1]):
    print(f"  {fam}: {c}")

section("10C. TREND ESTIMATORS ON PRICES (is there a deterministic trend?)")
n_lin = n_ts = 0
for i, t in enumerate(tickers):
    y = df.iloc[:, i].values; x = np.arange(len(y))
    lr = stats.linregress(x, y)
    rec.add("scipy", "trend", "linregress_slope", t, lr.slope, lr.pvalue,
            note=f"r2={lr.rvalue**2:.3f}")
    ts = stats.theilslopes(y, x)
    rec.add("scipy", "trend", "theilsen_slope", t, ts[0], np.nan,
            note=f"lo={ts[2]:.4f} hi={ts[3]:.4f} signif={not(ts[2]<0<ts[3])}")
    if lr.pvalue < 0.05: n_lin += 1
    if not (ts[2] < 0 < ts[3]): n_ts += 1
print(f"Significant OLS linear trend (p<0.05): {n_lin}/{N}")
print(f"Theil-Sen slope CI excludes 0:          {n_ts}/{N}")

section("10D. CORRELATION vs ALGO and vs MARKET (3 coefficients each)")
for i, t in enumerate(tickers):
    if i == 0: continue
    r = R[:, i]
    for label, ref in [("vsALGO", algo), ("vsMKT", mkt)]:
        for cname, fn in [("pearson", stats.pearsonr),
                          ("spearman", stats.spearmanr),
                          ("kendall", stats.kendalltau)]:
            res = fn(r, ref)
            rec.add("scipy", f"corr_{label}", cname, t, res[0], res[1])

section("10E. REGIME: FIRST-HALF vs SECOND-HALF RETURNS (12 two-sample tests)")
half = T // 2
sig_counts = {}
for i, t in enumerate(tickers):
    a, b = R[:half, i], R[half:, i]
    two = {
        "ttest_ind":        stats.ttest_ind(a, b),
        "welch_t":          stats.ttest_ind(a, b, equal_var=False),
        "mannwhitneyu":     stats.mannwhitneyu(a, b),
        "ranksums":         stats.ranksums(a, b),
        "kruskal":          stats.kruskal(a, b),
        "brunnermunzel":    stats.brunnermunzel(a, b),
        "ks_2samp":         stats.ks_2samp(a, b),
        "cramervonmises_2s":(lambda c:(c.statistic,c.pvalue))(stats.cramervonmises_2samp(a, b)),
        "epps_singleton":   stats.epps_singleton_2samp(a, b),
        "levene_var":       stats.levene(a, b),
        "bartlett_var":     stats.bartlett(a, b),
        "fligner_var":      stats.fligner(a, b),
        "mood_scale":       stats.mood(a, b),
        "ansari_scale":     stats.ansari(a, b),
    }
    for name, res in two.items():
        s, p = (res.statistic, res.pvalue) if hasattr(res, "statistic") else res
        rec.add("scipy", "regime_2samp", name, t, s, p)
        if p < 0.05:
            sig_counts[name] = sig_counts.get(name, 0) + 1
print("Significant (p<0.05) counts by test (mean/dist/scale shift 1st vs 2nd half):")
for name, c in sorted(sig_counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {c}/{N}")

section("10F. SIGN / RANDOMNESS / CONTINGENCY")
for i, t in enumerate(tickers):
    s = np.sign(R[:, i]); s = s[s != 0]
    up = int((s > 0).sum()); n = len(s)
    bt = stats.binomtest(up, n, 0.5)
    rec.add("scipy", "sign", "binomtest_updays", t, up / n, bt.pvalue)
    # sign-transition contingency (up->up etc.)
    prev, nxt = s[:-1], s[1:]
    a = int(((prev > 0) & (nxt > 0)).sum()); b = int(((prev > 0) & (nxt < 0)).sum())
    c = int(((prev < 0) & (nxt > 0)).sum()); d = int(((prev < 0) & (nxt < 0)).sum())
    table = np.array([[a, b], [c, d]])
    try:
        chi2, p, _, _ = stats.chi2_contingency(table)
        rec.add("scipy", "sign", "chi2_signtransition", t, chi2, p)
        g, gp, _, _ = stats.chi2_contingency(table, lambda_="log-likelihood")
        rec.add("scipy", "sign", "Gtest_signtransition", t, g, gp)
        fp = stats.fisher_exact(table)[1]
        rec.add("scipy", "sign", "fisher_signtransition", t, np.nan, fp)
    except Exception:
        pass
    # entropy of sign distribution & differential entropy of returns
    rec.add("scipy", "info", "differential_entropy", t,
            stats.differential_entropy(R[:, i]), np.nan)

# ANOVA of returns across the 51 instruments (are means jointly different?)
f, p = stats.f_oneway(*[R[:, i] for i in range(N)])
rec.add("scipy", "cross", "anova_means_51inst", "ALL", f, p)
ag = stats.alexandergovern(*[R[:, i] for i in range(N)])
rec.add("scipy", "cross", "alexandergovern_means", "ALL", ag.statistic, ag.pvalue)
kw = stats.kruskal(*[R[:, i] for i in range(N)])
rec.add("scipy", "cross", "kruskal_51inst", "ALL", kw.statistic, kw.pvalue)
# Bartlett/Levene across all 51: equal variances?
rec.add("scipy", "cross", "levene_var_51inst", "ALL", *stats.levene(*[R[:, i] for i in range(N)]))
print(f"ANOVA across 51 instruments: F={f:.2f} p={p:.3g} {stars(p)}")

rec.save()
