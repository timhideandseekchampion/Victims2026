"""Adaptive linear estimator for the peer-lead-lag OLS forecast.

Exponentially-weighted least squares (EWLS): a strict superset of the current
expanding-window OLS (half_life=None / inf reproduces it exactly). Weighting the
recent past more heavily is the pure-linear way to adapt to a drifting / regime-
shifting relationship — no neural nets, no extra capacity.

All matrices here are ROW-MAJOR: X is (n_obs, n_features), Y is (n_obs, n_targets),
and the NEWEST observation is the LAST row. Weight of row i is lambda**(n-1-i), so
the most recent row always has weight 1.
"""
import numpy as np


def half_life_to_lambda(half_life):
    """Decay factor lambda such that a point half_life rows back has weight 1/2.
    half_life=None or inf  ->  lambda=1 (equal weights = expanding window)."""
    if half_life is None or half_life == float("inf"):
        return 1.0
    return 0.5 ** (1.0 / half_life)


def n_eff(half_life, n):
    """Effective sample size of the exponential weights over n rows."""
    lam = half_life_to_lambda(half_life)
    if lam >= 1.0:
        return float(n)
    w = lam ** np.arange(n - 1, -1, -1)
    return float(w.sum() ** 2 / (w ** 2).sum())


def ewls_fit(X, Y, half_life=None, eps_scale=1e-8, alpha=0.0):
    """Weighted multi-output ridge regression with an unpenalised intercept
    (weighted-demean form). Returns (B, mx, my) so a prediction for feature row x* is
        pred = my + (x* - mx) @ B
    - eps_scale adds a *numerical* stabiliser (scaled to the weighted Gram trace).
    - alpha is a real L2 penalty (like sklearn Ridge, penalty on centred coefficients).
      alpha=0 is OLS. A light alpha (~0.1) shrinks the noisy 51x50 fit and empirically
      improves out-of-sample IC / Score; heavy alpha (>=10) destroys the edge.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, p = X.shape
    lam = half_life_to_lambda(half_life)
    w = np.ones(n) if lam >= 1.0 else lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw
    my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx
    Yc = Y - my
    XtWX = Xc.T @ (w[:, None] * Xc)
    XtWY = Xc.T @ (w[:, None] * Yc)
    eps = eps_scale * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + alpha) * np.eye(p), XtWY)
    return B, mx, my


def fit_rows(X, Y, cfg):
    """Fit (B, mx, my) on row-major X, Y under a scheme config:
      {"scheme":"expanding"}            -> equal weights, all rows
      {"scheme":"rolling","window":N}   -> equal weights, last N rows
      {"scheme":"ewls","half_life":h}   -> exponential weights, half-life h
    """
    scheme = cfg.get("scheme", "expanding")
    if scheme == "rolling":
        N = cfg["window"]
        X, Y, hl = X[-N:], Y[-N:], None
    elif scheme == "ewls":
        hl = cfg.get("half_life")
    else:  # expanding
        hl = None
    return ewls_fit(X, Y, hl, cfg.get("eps_scale", 1e-8), cfg.get("alpha", 0.0))


def make_predictor(ret, cfg):
    """Convenience: build (X, Y) from a (n_inst, n_time) return panel exactly as the
    strategy does, fit under cfg, and return the next-day forecast (n_targets,) for
    the day after ret's last column."""
    X = ret[:, :-1].T
    Y = ret[1:, 1:].T
    B, mx, my = fit_rows(X, Y, cfg)
    return my + (ret[:, -1] - mx) @ B
