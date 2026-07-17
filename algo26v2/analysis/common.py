"""Shared data loading / helpers for the Algothon prices analysis.

Data: prices.txt = 501 rows (days) x 51 columns (instruments), whitespace-sep,
header row of tickers. eval.py transposes to (nInst, nt). Instrument 0 (ALGO)
is SPECIAL: 5x lower commission (0.2bp vs 1bp) and 10x higher $ position limit
(100k vs 10k) -> clearly the intended main alpha vehicle.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRICES = os.path.join(ROOT, "prices.txt")
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)


def load():
    """Return (df_prices [T x N], tickers list). Rows=days, cols=instruments."""
    df = pd.read_csv(PRICES, sep=r"\s+", header=0, index_col=None)
    return df, list(df.columns)


def prices_array():
    """(N x T) like eval.py, and (T x N) DataFrame."""
    df, tickers = load()
    P = df.values.T  # N x T
    return P, df, tickers


def log_returns(df):
    """Log returns DataFrame, shape (T-1 x N)."""
    return np.log(df / df.shift(1)).dropna()


def simple_returns(df):
    return df.pct_change().dropna()


COMM_DEFAULT = 0.0001
COMM_INST0 = 0.00002
POSLIM_DEFAULT = 10_000
POSLIM_INST0 = 100_000
N_TEST_DAYS = 250


def section(title):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def stars(p):
    """Significance stars for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


class Recorder:
    """Collects one row per statistical test into a CSV for global aggregation.

    Columns: library, family, test, target, statistic, pvalue, note.
    A NaN/None pvalue means the test has no p-value (report statistic only).
    """
    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, library, family, test, target, statistic=np.nan,
            pvalue=np.nan, note=""):
        self.rows.append(dict(library=library, family=family, test=test,
                              target=str(target), statistic=statistic,
                              pvalue=pvalue, note=note))

    def save(self):
        df = pd.DataFrame(self.rows)
        path = os.path.join(RESULTS, f"{self.name}.csv")
        df.to_csv(path, index=False)
        withp = df[df.pvalue.notna()]
        nsig = (withp.pvalue < 0.05).sum() if len(withp) else 0
        print(f"\n[{self.name}] recorded {len(df)} tests; "
              f"{len(withp)} with p-values; {nsig} raw-significant (p<0.05). "
              f"-> {path}")
        return df
