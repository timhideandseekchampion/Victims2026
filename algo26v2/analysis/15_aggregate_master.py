"""Master aggregation of every recorded test across all libraries.

Reads results/*_full.csv, pools all p-values, applies Benjamini-Hochberg FDR
(and Bonferroni) GLOBALLY - because with ~5000 tests, raw p<0.05 alone yields
hundreds of false positives. Only findings that survive FDR are "real".
Produces results/MASTER_RESULTS.csv and a printed catalog.
"""
import glob, os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from common import RESULTS, section

files = sorted(glob.glob(os.path.join(RESULTS, "*_full.csv")))
frames = [pd.read_csv(f) for f in files]
allr = pd.concat(frames, ignore_index=True)
allr["pvalue"] = pd.to_numeric(allr["pvalue"], errors="coerce")

section("MASTER AGGREGATION - EVERY TEST RUN")
print(f"CSV files combined: {[os.path.basename(f) for f in files]}")
print(f"Total test rows recorded: {len(allr)}")
withp = allr[allr.pvalue.notna()].copy()
print(f"Rows with a p-value: {len(withp)}")
print(f"Rows reporting only a statistic (no p): {len(allr)-len(withp)}")

# global multiple-testing correction
p = withp.pvalue.clip(1e-300, 1).values
rej_bh, q_bh, _, _ = multipletests(p, alpha=0.05, method="fdr_bh")
rej_bonf, _, _, _ = multipletests(p, alpha=0.05, method="bonferroni")
withp["q_bh"] = q_bh
withp["sig_raw"] = withp.pvalue < 0.05
withp["sig_bh"] = rej_bh
withp["sig_bonf"] = rej_bonf

section("SIGNIFICANCE FUNNEL (honest multiple-testing view)")
print(f"  raw p<0.05:              {withp.sig_raw.sum():>5}  (expected under pure noise ~{0.05*len(withp):.0f})")
print(f"  survive BH-FDR 5%:       {withp.sig_bh.sum():>5}")
print(f"  survive Bonferroni 5%:   {withp.sig_bonf.sum():>5}")

section("WHAT SURVIVES FDR, BROKEN DOWN BY TEST FAMILY")
surv = withp[withp.sig_bh]
tab = (surv.groupby(["library", "family", "test"])
           .size().reset_index(name="n_survive")
           .sort_values("n_survive", ascending=False))
pd.set_option("display.max_rows", 200); pd.set_option("display.width", 200)
print(tab.to_string(index=False))

section("RAW-SIGNIFICANT COUNTS BY FAMILY (before correction, for context)")
rawtab = (withp.groupby(["library","family"])
              .agg(n_tests=("pvalue","size"), n_raw_sig=("sig_raw","sum"),
                   n_bh_sig=("sig_bh","sum")).reset_index())
rawtab["expected_fp"] = (0.05*rawtab.n_tests).round(1)
rawtab["excess"] = rawtab.n_raw_sig - rawtab.expected_fp
print(rawtab.sort_values("excess", ascending=False).to_string(index=False))

section("TOP 40 STRONGEST FINDINGS (lowest q-value)")
top = surv.sort_values("q_bh").head(40)[["library","family","test","target","statistic","pvalue","q_bh","note"]]
print(top.to_string(index=False))

allout = withp.sort_values("q_bh")
allout.to_csv(os.path.join(RESULTS, "MASTER_RESULTS.csv"), index=False)
print(f"\nFull ranked table -> results/MASTER_RESULTS.csv ({len(allout)} rows with p-values)")

section("PLAIN-LANGUAGE SUMMARY")
def cnt(fam_sub, lib=None):
    m = surv.family.str.contains(fam_sub)
    if lib: m &= surv.library.eq(lib)
    return m.sum()
print(f"- Distribution/normality findings surviving FDR: {cnt('normality')}")
print(f"- Cointegration findings surviving FDR:          {cnt('coint')}")
print(f"- Correlation (vs ALGO/market) surviving FDR:    {cnt('corr')}")
print(f"- Trend findings surviving FDR:                  {cnt('trend')}")
print(f"- Autocorrelation findings surviving FDR:        {cnt('autocorr')}")
print(f"- Granger findings surviving FDR:                {cnt('granger')}")
print(f"- Heteroskedasticity findings surviving FDR:     {cnt('hetero')+cnt('garch')}")
print(f"- Regime/2-sample findings surviving FDR:        {cnt('regime')}")
