#!/usr/bin/env python
"""Reverse-engineer the data generator (dev-only planning tool).

Characterises the structure of prices.txt so we can plan against the *process*,
not just fit the sample. Prints a report; the interpretation lives in DGP.md.

    python analyze.py

Uses statsmodels (in the grading sandbox) for ADF / cointegration tests.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint


def load():
    df = pd.read_csv("prices.txt", sep=r"\s+")
    return df.values.T, list(df.columns)   # (nInst, nDays), names


def section(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main():
    prc, names = load()
    N, T = prc.shape
    logp = np.log(prc)
    r = np.diff(logp, axis=1)               # (N, T-1) daily log returns
    r0 = r[0]                               # ALGO / instrument 0

    section("1. FACTOR STRUCTURE — is ALGO (inst 0) the market index?")
    avg_oth = r[1:].mean(axis=0)
    print(f"corr(ALGO returns, equal-weight avg of other {N-1}): "
          f"{np.corrcoef(r0, avg_oth)[0, 1]:+.4f}")
    idx_lvl = (prc / prc[:, [0]]).mean(axis=0)   # equal-weight index level
    print(f"corr(ALGO price level, equal-weight index level): "
          f"{np.corrcoef(prc[0], idx_lvl)[0, 1]:+.4f}")
    print(f"ALGO total move {prc[0, -1]/prc[0, 0]-1:+.1%}   "
          f"index total move {idx_lvl[-1]/idx_lvl[0]-1:+.1%}")
    beta = np.array([np.polyfit(r0, r[k], 1)[0] for k in range(N)])
    r2 = np.array([1 - np.var(r[k] - beta[k] * r0) / np.var(r[k]) for k in range(N)])
    print(f"beta on ALGO (excl ALGO): mean {beta[1:].mean():+.2f}  "
          f"frac>0 {(beta[1:] > 0).mean():.2f}")
    print(f"R^2 on ALGO  (excl ALGO): mean {r2[1:].mean():.3f}  max {r2[1:].max():.3f}")
    Rz = (r - r.mean(1, keepdims=True)) / r.std(1, keepdims=True)
    evals = np.sort(np.linalg.eigvalsh(np.cov(Rz)))[::-1]
    print(f"PCA variance explained, top 5 PCs: {(evals[:5]/evals.sum()).round(3)}")

    section("2. STATIONARITY (ADF) — levels random-walk, spreads mean-revert?")
    def adf_frac(series_list):
        pvals = []
        for s in series_list:
            try:
                pvals.append(adfuller(s, autolag="AIC")[1])
            except Exception:
                pvals.append(1.0)
        pvals = np.array(pvals)
        return np.median(pvals), (pvals < 0.05).mean()
    m, f = adf_frac([logp[k] for k in range(N)])
    print(f"log-price levels : median ADF p {m:.2f}   frac stationary@5% {f:.2f}  (expect NON-stationary)")
    m, f = adf_frac([r[k] for k in range(N)])
    print(f"daily returns    : median ADF p {m:.2f}   frac stationary@5% {f:.2f}  (expect stationary)")
    resid_lvl = [np.cumsum(r[k] - beta[k] * r0) for k in range(1, N)]
    m, f = adf_frac(resid_lvl)
    print(f"residual (stock-mkt) levels: median ADF p {m:.2f}   frac stationary@5% {f:.2f}")

    section("3. COINTEGRATION (Engle-Granger) — does each stock cointegrate with ALGO?")
    coint_p = []
    for k in range(1, N):
        try:
            coint_p.append(coint(prc[k], prc[0])[1])
        except Exception:
            coint_p.append(1.0)
    coint_p = np.array(coint_p)
    print(f"stock~ALGO cointegration: {(coint_p < 0.05).sum()}/{N-1} at 5%,  "
          f"{(coint_p < 0.10).sum()}/{N-1} at 10%  (few => weak stat-arb / slow reversion)")

    section("4. RESIDUAL DYNAMICS — OU half-life & per-stock trend/revert")
    hl = []
    for s in resid_lvl:
        s = s - s.mean()
        phi = np.polyfit(s[:-1], s[1:], 1)[0]
        if 0 < phi < 1:
            hl.append(-np.log(2) / np.log(phi))
    hl = np.array(hl)
    print(f"residual OU half-life (days): median {np.median(hl):.0f}  "
          f"p25 {np.percentile(hl,25):.0f}  p75 {np.percentile(hl,75):.0f}  "
          f"(long => slow reversion / near random walk)")
    ac = np.array([np.corrcoef(r[k][1:], r[k][:-1])[0, 1] for k in range(1, N)])
    print(f"per-stock return lag-1 autocorr: mean {ac.mean():+.3f}  "
          f"frac<0 (revert) {(ac<0).mean():.2f}  frac>0 (trend) {(ac>0).mean():.2f}")

    section("5. INDEX-LEVEL REVERSION — is trading the ALGO index itself an edge?")
    # The index (inst 0) has a 10x position limit ($100k) and 5x lower fee (0.2bp):
    # a fingerprint that it's meant to be traded. Test whether it mean-reverts at a
    # short horizon with a permutation null — shuffle ALGO's daily returns (destroys
    # any temporal reversion, keeps the marginal) and see if the tradeable Score
    # survives. Overlapping-window autocorrelation is biased negative, so we judge
    # by the clean backtest SCORE, not autocorr.
    import backtester as bt
    import strategy as st
    comm, lim = bt.make_grading_params(N)

    def _score(gp, panel):
        return bt.run_backtest(panel, gp, num_test_days=min(250, panel.shape[1] - 1),
                               comm_rate=comm, dlr_pos_limit=lim, inst_names=names).score

    def algo_leg(w=5):
        def f(p):
            nn, t = p.shape; pos = np.zeros(nn, dtype=int)
            if t < w + 2:
                return pos
            z = st.zscore(p, w)[0]
            pos[0] = int(np.sign(-z) * lim[0] / p[0, -1])
            return pos
        return f

    obs = _score(algo_leg(5), prc)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        perm = prc.copy()
        rr = r0.copy(); rng.shuffle(rr)
        perm[0] = prc[0, 0] * np.exp(np.concatenate([[0.0], np.cumsum(rr)]))
        null.append(_score(algo_leg(5), perm))
    null = np.array(null)
    p_ge = float(np.mean(null >= obs))
    print(f"ALGO-leg zrev(5) @ $100k — Score: observed {obs:.0f}   "
          f"shuffled-returns null mean {null.mean():.0f} (95%ile {np.percentile(null,95):.0f})")
    print(f"  P(null >= observed) = {100*p_ge:.0f}%  => index reversion is "
          f"{'REAL (not a random-walk artifact)' if p_ge < 0.05 else 'NOT distinguishable from noise'}")
    print("  SIZING is the dominant lever: Score scales ~linearly with deployed capital\n"
          "  up to the $-limit clip (Sharpe is scale-invariant). The edge is monetised by\n"
          "  running BOTH the index leg ($100k, 0.2bp) and the 50-name idio leg near limits.")

    section("VERDICT")
    print("One market factor (ALGO = the index) + mostly-idiosyncratic names. TWO\n"
          "short-horizon reversions are tradeable and BOTH pay: (a) CROSS-SECTIONAL\n"
          "relative-value reversion across the 50 names, and (b) the ALGO INDEX itself\n"
          "mean-reverts over ~5 days — a real edge (permutation null above), and the\n"
          "designers' 10x-limit / 5x-lower-fee fingerprint says to trade it. The score\n"
          "comes from sizing BOTH legs near the dollar limits. See DGP.md / SIGNALS.md.")


if __name__ == "__main__":
    main()
