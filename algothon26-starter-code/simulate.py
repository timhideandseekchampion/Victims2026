#!/usr/bin/env python
"""DGP simulator + stress test (dev-only).

Reverse-engineering, operationalised: fit a generative model to prices.txt,
check it reproduces the tradeable structure, then stress-test strategies on many
SYNTHETIC 250-day panels — far stronger than one historical window, since the
graded stage is different data from the same process.

Generator (from analyze.py findings) — a ONE-FACTOR market with two reversions:
  * market factor (ALGO) = permanent random-walk drift PLUS a short-horizon
    MEAN-REVERSION toward its own recent moving average (the index k-day return
    autocorrelation is ≈ −0.1, stable across windows) — `theta_m` sets its speed;
  * each stock = beta·market + an idiosyncratic log-price that mean-reverts toward
    the CROSS-SECTIONAL average (relative-value OU) — `kappa` sets its speed.

Both reversions are tradeable and BOTH are needed to reproduce the observed edge.
An earlier version of this file modelled the market as a pure random walk, so its
panels could not contain the index-reversion edge (and wrongly scored the ALGO
leg as noise). This version calibrates `theta_m` to the observed index autocorr.

    python simulate.py                 # fit, calibrate, validate, stress-test
    python simulate.py --paths 200     # more synthetic panels (more stable %s)

Reuses backtester.run_backtest / make_grading_params and strategy signals.
"""
import argparse
import numpy as np
import pandas as pd

import backtester as bt
import strategy as st

M_MA_W = 5          # market moving-average window the index reverts toward


def load():
    df = pd.read_csv("prices.txt", sep=r"\s+")
    return df.values.T, list(df.columns)


def fit(prc):
    r = np.diff(np.log(prc), axis=1)
    r0 = r[0]
    mu_m, sig_m = float(r0.mean()), float(r0.std())
    v0 = float(r0 @ r0)
    beta = np.array([float(r[k] @ r0 / v0) for k in range(prc.shape[0])])
    resid = r - np.outer(beta, r0)
    sigma_e = resid.std(axis=1)
    return {"mu_m": mu_m, "sig_m": sig_m, "beta": beta, "sigma_e": sigma_e,
            "P0": prc[:, 0].copy()}


def simulate(f, ndays, kappa, theta_m, seed):
    """Factor(with MA-reversion) + cross-sectional-OU idiosyncratic. -> (nInst, ndays)."""
    rng = np.random.default_rng(seed)
    N = len(f["P0"])
    beta = beta_clip(f["beta"])
    # market log-level: drift RW + pull back toward its own M_MA_W-day average
    logm = np.zeros(ndays)
    for t in range(1, ndays):
        lo = max(0, t - M_MA_W)
        ma = logm[lo:t].mean()
        pull = -theta_m * (logm[t - 1] - ma)
        logm[t] = logm[t - 1] + f["mu_m"] + pull + rng.normal(0, f["sig_m"])
    # idiosyncratic log-price gaps, mean-reverting to the cross-sectional mean
    S = np.zeros(N)
    out = np.zeros((N, ndays))
    for t in range(ndays):
        S = S - kappa * (S - S.mean()) + rng.normal(0, 1, N) * f["sigma_e"]
        out[:, t] = f["P0"] * np.exp(beta * logm[t] + S)
    return out


def beta_clip(b):
    return np.clip(b, 0.0, 3.0)


def xs_ic(prc, w=20, h=5):
    """Cross-sectional IC of -zscore(w) vs forward h-day return (mean over days)."""
    ics = []
    for t in range(w, prc.shape[1] - h):
        s = -st.zscore(prc[:, :t + 1], w)
        fwd = prc[:, t + h] / prc[:, t] - 1
        m = np.isfinite(s)
        if m.sum() > 3:
            ics.append(np.corrcoef(s[m], fwd[m])[0, 1])
    return float(np.nanmean(ics))


def avg_corr(prc):
    r = np.diff(np.log(prc), axis=1)
    C = np.corrcoef(r)
    return float(C[np.triu_indices(len(C), 1)].mean())


def algo_ac(prc, k=5):
    """Index k-day-return vs next-k-day-return correlation (<0 => reversion).
    NOTE: overlapping windows bias this negative even for a random walk — it is a
    diagnostic only. Calibration targets achievable Score (see below), not this."""
    p0 = prc[0]
    n = prc.shape[1] - 2 * k
    if n < 5:
        return float("nan")
    a = np.array([np.log(p0[i + k] / p0[i]) for i in range(n)])
    b = np.array([np.log(p0[i + 2 * k] / p0[i + k]) for i in range(n)])
    return float(np.corrcoef(a, b)[0, 1])


# --- single-leg probes: the tradeable quantity we calibrate against is SCORE ----
def _leg_algo(w=5):
    """ALGO-only reversion at its $100k limit (the index leg in isolation)."""
    def f(prc):
        nn, t = prc.shape; pos = np.zeros(nn, dtype=int)
        if t < w + 2:
            return pos
        z = st.zscore(prc, w)[0]
        pos[0] = int(np.sign(-z) * st.dollar_limits(nn)[0] / prc[0, -1])
        return pos
    return f


def _leg_idio(w=10, scale=0.10):
    """50-name cross-sectional reversion with the index excluded (the idio leg)."""
    def f(prc):
        nn, t = prc.shape
        if t < w + 2:
            return np.zeros(nn, dtype=int)
        s = (-st.zscore(prc, w)).astype(float)
        s[0] = 0.0
        s[1:] = s[1:] - s[1:].mean()
        return st.size_fraction_of_limit(s, prc, scale)
    return f


def _leg_score(f, kappa, theta, panels, gp, comm, lim, names, seed0):
    out = []
    for s in range(panels):
        x = simulate(f, 251, kappa, theta, seed0 + s)
        out.append(bt.run_backtest(x, gp, num_test_days=min(250, x.shape[1] - 1),
                                   comm_rate=comm, dlr_pos_limit=lim,
                                   inst_names=names).score)
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser(description="DGP simulator + stress test")
    ap.add_argument("--paths", type=int, default=150)
    ap.add_argument("--ndays", type=int, default=300)
    args = ap.parse_args()

    prc, names = load()
    f = fit(prc)
    comm, lim = bt.make_grading_params(prc.shape[0])
    obs_ic, obs_corr, obs_ac = xs_ic(prc), avg_corr(prc), algo_ac(prc, 5)

    real_days = prc.shape[1] - 50            # score all real days minus warm-up (450 of 500)

    def real_score(gp):
        return bt.run_backtest(prc, gp, num_test_days=real_days, comm_rate=comm,
                               dlr_pos_limit=lim, inst_names=names).score
    obs_idio = real_score(_leg_idio())
    obs_algo = real_score(_leg_algo())
    print(f"observed (real prices.txt): idio-leg Score {obs_idio:.0f}   "
          f"ALGO-leg Score {obs_algo:.0f}")
    print(f"  diagnostics: reversion IC@5 {obs_ic:+.3f}  avg corr {obs_corr:+.2f}  "
          f"ALGO k5 autocorr {obs_ac:+.3f}\n")

    # We calibrate to achievable SCORE (what's graded), NOT to IC/autocorr — the OU
    # generator can't match both IC and Score (real reversion is more monetizable),
    # and a random-walk market gives ALGO-leg Score ~0 so P(RW >= observed) ~ 0%:
    # the index edge is real, and theta_m>0 is what puts it into the generator.
    print("=== calibrate kappa to observed idio-leg Score ===")
    bestk = None
    for kappa in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
        m = _leg_score(f, kappa, 0.0, 40, _leg_idio(), comm, lim, names, 5000)
        print(f"  kappa {kappa:>5}: synth idio-leg Score {m:7.1f}")
        if bestk is None or abs(m - obs_idio) < bestk[1]:
            bestk = (kappa, abs(m - obs_idio))
    kappa = bestk[0]
    print(f"-> chosen kappa {kappa}\n")

    print("=== calibrate theta_m to observed ALGO-leg Score ===")
    bestt = None
    for theta in (0.0, 0.1, 0.2, 0.3, 0.4):
        m = _leg_score(f, kappa, theta, 40, _leg_algo(), comm, lim, names, 6000)
        print(f"  theta_m {theta:>4}: synth ALGO-leg Score {m:7.1f}")
        if bestt is None or abs(m - obs_algo) < bestt[1]:
            bestt = (theta, abs(m - obs_algo))
    theta_m = bestt[0]
    print(f"-> chosen theta_m {theta_m} (theta_m=0 => random-walk index => ALGO leg "
          f"un-tradeable; positive => the real index edge)\n")

    # --- stress test across many independent synthetic panels ---
    configs = [
        ("two_leg (submission)", st.make_two_leg(idio_w=10, algo_w=5, idio_scale=0.10,
                                                 algo_frac=1.0, algo_scale=0.10)),
        ("two_leg algoW7", st.make_two_leg(idio_w=10, algo_w=7)),
        ("two_leg algoFrac0.6", st.make_two_leg(idio_w=10, algo_w=5, algo_frac=0.6)),
        ("single zrev10 hot", st.make_get_position(signal_fn=st.zrev(10),
                                                   sizing="fraction", scale=0.10)),
        ("rev_blend (old core)", st.make_get_position(signal_fn=st.alpha_rev_blend,
                                                      sizing="fraction", scale=2.0)),
        ("flat", lambda p: np.zeros(p.shape[0])),
    ]
    print(f"=== stress test: {args.paths} synthetic {real_days}-day panels "
          f"(kappa {kappa}, theta_m {theta_m}) ===")
    print(f"{'config':<24}{'Score 5%':>10}{'median':>9}{'95%':>9}"
          f"{'mean':>8}{'%profit':>9}")
    sims = [simulate(f, real_days + 51, kappa, theta_m, 1000 + s) for s in range(args.paths)]
    for label, gp in configs:
        scores = []
        for x in sims:
            r = bt.run_backtest(x, gp, num_test_days=min(real_days, x.shape[1] - 1),
                                comm_rate=comm, dlr_pos_limit=lim, inst_names=names)
            scores.append(r.score)
        s = np.array(scores)
        print(f"{label:<24}{np.percentile(s,5):>10.1f}{np.median(s):>9.1f}"
              f"{np.percentile(s,95):>9.1f}{s.mean():>8.1f}{100*np.mean(s>0):>8.0f}%")
    print("\n(Judge by EXPECTED graded score: median/mean high AND 5th-percentile / "
          "%profit acceptable. The two-leg book should dominate the single-signal and\n"
          " the old rev_blend core once the index-reversion edge is in the generator.)")


if __name__ == "__main__":
    main()
