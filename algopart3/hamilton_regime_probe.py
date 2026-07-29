"""
hamilton_regime_probe.py -- PROTOTYPE: a 2-state Hamilton (1989) Markov-switching model applied to
SAFE_lldollar.py's own champion daily traded PnL series, as a candidate regime detector to compare
against the two detectors already in this repo:
  - xsac        trailing-40d mean cross-sectional lag-1 autocorrelation (SAFE_lldollar.py / SAFE_rotate.py)
  - pnl-sum     trailing-60d summed sign(forecast).realized-return < 0 (the more-sensitive gate this
                repo adopted for SAFE_rotate.py/SAFE_live.py and SAFE_llboost_v11+'s kill switch)

WHY THIS ISN'T A REPEAT of the already-rejected "fit-then-freeze HMM/k-means regime labels add no OOS
validator value" result (memory: ic-vs-score-lesson / algothon-protection-stack):
  1. Proper generative model (regime-conditional mean+variance + an estimated Markov transition
     matrix), fit by direct MLE against the actual Hamilton-filter likelihood -- not a distance-based
     clustering.
  2. Operates on a SINGLE scalar series (the book's own daily champion PnL), not a joint clustering of
     the 50-stock cross-section.
  3. Parameters are re-estimated WALK-FORWARD on a bounded trailing window (MAX_TRAIN days, expanding
     up to that cap), refit every REFIT_EVERY days -- the filtered P(hostile) at day t only ever uses
     data through day t. No full-sample fit-then-freeze leakage.

MODEL: y_t | S_t=k ~ N(mu_k, sigma_k^2), S_t in {0,1}, first-order Markov chain with transition matrix
[[p00, 1-p00],[1-p11, p11]]. State 1 is canonicalized post-fit to be the LOWER-mean ("hostile") state.
Fit by direct MLE (Nelder-Mead in an unconstrained reparametrization: log-sigma, logit-p) maximizing
the standard Hamilton-filter log-likelihood -- simpler than full EM, equivalent at convergence, and
cheap enough at this scale (6 free parameters, <=MAX_TRAIN observations per fit).

SCOPE, stated plainly (first pass, not a finished validator):
  - Single scenario generator (the momentum/flip/noise injector already used by
    test_v14_trend_regime.py / algopart2/stress_momentum.py), single seed per regime.
  - Small optimizer budget (n_restarts=2, maxiter=300) chosen to stay cheap while three other
    backtests are running concurrently in this repo -- not swept for adequacy.

UPDATE (same day): the raw P(state=1)>0.5 crossing (no debounce) gave 58/904 real-data false
positives -- fails the "0 false positives" bar every other detector in this repo is held to, unlike
xsac/pnl-sum which already have their own trailing-window smoothing. Added a HAMILTON_PERSIST=5
persistence gate (same magnitude as this repo's own ROT_P/XSAC_P convention): flag only when the raw
crossing holds for 5 CONSECUTIVE days, exactly like `_xsac_flag`/`_choose` elsewhere. Both the raw
and persistence-gated flag series are reported below so the improvement is visible, not just claimed.

Run: python3 hamilton_regime_probe.py
"""
import numpy as np
from scipy.optimize import minimize
import pandas as pd
import SAFE_lldollar as LL

MAX_TRAIN = 500
MIN_TRAIN = 250
REFIT_EVERY = 60
HAMILTON_PERSIST = 5
N_RESTARTS = 2
MAXITER = 300


def _norm_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _filter(y, mu0, mu1, s0, s1, p00, p11):
    P = np.array([[p00, 1 - p00], [1 - p11, p11]])
    a = 1 - p11; b = 1 - p00; denom = a + b
    xi_pred = np.array([a, b]) / denom if denom > 1e-12 else np.array([0.5, 0.5])
    T = len(y)
    filt = np.empty((T, 2)); loglik = 0.0
    for t in range(T):
        f = np.array([_norm_pdf(y[t], mu0, s0), _norm_pdf(y[t], mu1, s1)])
        joint = xi_pred * f; dens = joint.sum() + 1e-300
        loglik += np.log(dens)
        xi_filt = joint / dens
        filt[t] = xi_filt
        xi_pred = xi_filt @ P
    return filt, loglik, xi_pred


def _neg_loglik(theta, y):
    mu0, mu1, log_s0, log_s1, a00, a11 = theta
    s0 = np.exp(log_s0) + 1e-8; s1 = np.exp(log_s1) + 1e-8
    p00 = 1 / (1 + np.exp(-a00)); p11 = 1 / (1 + np.exp(-a11))
    _, ll, _ = _filter(y, mu0, mu1, s0, s1, p00, p11)
    return -ll


def fit_2state(y, n_restarts=N_RESTARTS, maxiter=MAXITER, seed=0):
    rng = np.random.default_rng(seed)
    mu, sd = y.mean(), y.std() + 1e-8
    best = None
    for _ in range(n_restarts):
        theta0 = np.array([
            mu + 0.3 * sd * rng.standard_normal(),
            mu - 0.3 * sd * rng.standard_normal(),
            np.log(sd), np.log(sd),
            1.5 + 0.5 * rng.standard_normal(),
            1.5 + 0.5 * rng.standard_normal(),
        ])
        res = minimize(_neg_loglik, theta0, args=(y,), method="Nelder-Mead",
                        options=dict(maxiter=maxiter, xatol=1e-5, fatol=1e-5))
        if best is None or res.fun < best.fun:
            best = res
    mu0, mu1, log_s0, log_s1, a00, a11 = best.x
    s0 = np.exp(log_s0) + 1e-8; s1 = np.exp(log_s1) + 1e-8
    p00 = 1 / (1 + np.exp(-a00)); p11 = 1 / (1 + np.exp(-a11))
    if mu1 > mu0:  # canonicalize: state 1 = LOWER mean ("hostile")
        mu0, mu1 = mu1, mu0; s0, s1 = s1, s0; p00, p11 = p11, p00
    return dict(mu0=mu0, mu1=mu1, s0=s0, s1=s1, p00=p00, p11=p11)


def walk_forward_hamilton(y, min_train=MIN_TRAIN, max_train=MAX_TRAIN, refit_every=REFIT_EVERY):
    """Causal: at day t, uses only y[:t+1], and any fit uses only a trailing window ending at t."""
    T = len(y)
    probs = np.full(T, np.nan)
    params = None; xi_pred = None; P = None
    next_refit = min_train
    for t in range(T):
        if t < min_train:
            continue
        if params is None or t >= next_refit:
            lo = max(0, t + 1 - max_train)
            params = fit_2state(y[lo:t + 1])
            P = np.array([[params["p00"], 1 - params["p00"]],
                          [1 - params["p11"], params["p11"]]])
            filt, _, xi_pred = _filter(y[lo:t + 1], **params)
            probs[t] = filt[-1, 1]
            next_refit = t + refit_every
            continue
        f = np.array([_norm_pdf(y[t], params["mu0"], params["s0"]),
                      _norm_pdf(y[t], params["mu1"], params["s1"])])
        joint = xi_pred * f; dens = joint.sum() + 1e-300
        xi_filt = joint / dens
        probs[t] = xi_filt[1]
        xi_pred = xi_filt @ P
    return probs


def champ_pn_and_ret(P_):
    """Un-pruned champion PN series + demeaned realized-return vectors, real (or extended) prices."""
    nInst, nDays = P_.shape
    pn = np.full(nDays, np.nan)
    rc = np.full((nDays, nInst - 1), np.nan)
    for n in range(LL.WARMUP, nDays):
        sig = LL._forecasts(P_[:, :n])["champ"]
        R = np.log(P_[1:, n]) - np.log(P_[1:, n - 1])
        Rc = R - R.mean()
        pn[n] = float((np.sign(sig) * Rc).sum())
        rc[n] = Rc
    return pn, rc


def xsac_series(rc, win=LL.XSAC_W):
    nDays = rc.shape[0]
    xc = np.full(nDays, np.nan)
    for n in range(1, nDays):
        a, b = rc[n - 1], rc[n]
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        d = np.sqrt((a @ a) * (b @ b))
        xc[n] = float(a @ b / d) if d > 1e-18 else 0.0
    xsac = np.full(nDays, np.nan)
    for n in range(nDays):
        lo = n - win + 1
        if lo < 0:
            continue
        seg = xc[lo:n + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= win // 2:
            xsac[n] = seg.mean()
    return xsac


def pnl_sum_series(pn, win=LL.ROT_W):
    nDays = len(pn)
    s = np.full(nDays, np.nan)
    for n in range(nDays):
        lo = n - win + 1
        if lo < 0:
            continue
        seg = pn[lo:n + 1]
        if not np.isnan(seg).any():
            s[n] = seg.sum()
    return s


def make_ext(P_real, kind, T_ext=150, mom=0.6, period=25, K=5, seed=1):
    """Same generator as test_v14_trend_regime.py / algopart2/stress_momentum.py."""
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy()
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        if kind == "noise":
            drift = np.zeros(names.shape[0])
        elif kind == "flip":
            sgn = 1.0 if (step // period) % 2 == 0 else -1.0
            drift = sgn * mom * (tc / (tc.std() + 1e-9)) * vol
        else:
            drift = mom * (tc / (tc.std() + 1e-9)) * vol
        drift -= drift.mean()
        noise = rng.normal(0, vol, names.shape[0]); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :P_real.shape[1]] = P_real
    return full


def first_flag(mask, start):
    idx = np.where(mask[start:])[0]
    return int(idx[0]) if len(idx) else None


def sustained(mask, persist=HAMILTON_PERSIST):
    """True at day t only if mask[t-persist+1 : t+1] are ALL True -- same shape as this repo's own
    `_xsac_flag`/`_choose` persistence convention (ROT_P/XSAC_P)."""
    out = np.zeros(len(mask), dtype=bool)
    run = 0
    for t in range(len(mask)):
        run = run + 1 if mask[t] else 0
        out[t] = run >= persist
    return out


if __name__ == "__main__":
    P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nDays = P_real.shape

    print("=== 1. real prices.txt: false-positive check ===")
    pn, rc = champ_pn_and_ret(P_real)
    xsac = xsac_series(rc)
    pnlsum = pnl_sum_series(pn)
    valid0 = LL.WARMUP
    ham = walk_forward_hamilton(np.nan_to_num(pn[valid0:], nan=0.0))
    ham_full = np.full(nDays, np.nan); ham_full[valid0:] = ham

    xsac_flags = np.nan_to_num(xsac, nan=-1e9) > LL.XSAC_TH
    pnlsum_flags = np.nan_to_num(pnlsum, nan=1e9) < 0
    ham_raw = np.nan_to_num(ham_full, nan=0.0) > 0.5
    ham_flags = sustained(ham_raw)

    print(f"  xsac         : {xsac_flags.sum()} flag-days / {nDays - valid0} (real-data bar: 0)")
    print(f"  pnl-sum      : {pnlsum_flags.sum()} flag-days / {nDays - valid0}")
    print(f"  hamilton-raw : {ham_raw.sum()} flag-days / {nDays - valid0} (no persistence gate)")
    print(f"  hamilton-{HAMILTON_PERSIST}d  : {ham_flags.sum()} flag-days / {nDays - valid0} "
          f"(sustained-{HAMILTON_PERSIST}-day gate)")

    print("\n=== 2. synthetic momentum/flip/noise regime injection ===")
    for kind in ("momentum", "flip", "noise"):
        full = make_ext(P_real, kind)
        pn_e, rc_e = champ_pn_and_ret(full)
        xsac_e = xsac_series(rc_e)
        pnlsum_e = pnl_sum_series(pn_e)
        ham_e_tail = walk_forward_hamilton(np.nan_to_num(pn_e[valid0:], nan=0.0))
        ham_e = np.full(full.shape[1], np.nan); ham_e[valid0:] = ham_e_tail

        xsac_e_flags = np.nan_to_num(xsac_e, nan=-1e9) > LL.XSAC_TH
        pnlsum_e_flags = np.nan_to_num(pnlsum_e, nan=1e9) < 0
        ham_e_raw = np.nan_to_num(ham_e, nan=0.0) > 0.5
        ham_e_flags = sustained(ham_e_raw)

        S = nDays  # start of injected window
        E = full.shape[1]
        fx = first_flag(xsac_e_flags, S); fp = first_flag(pnlsum_e_flags, S)
        fhr = first_flag(ham_e_raw, S); fh = first_flag(ham_e_flags, S)
        nx = int(xsac_e_flags[S:].sum()); np_ = int(pnlsum_e_flags[S:].sum())
        nhr = int(ham_e_raw[S:].sum()); nh = int(ham_e_flags[S:].sum())
        print(f"\n  --- {kind.upper()} (injected window = {E - S} days) ---")
        print(f"  xsac         : first flag day = {fx}  flag-days = {nx}/{E-S}")
        print(f"  pnl-sum      : first flag day = {fp}  flag-days = {np_}/{E-S}")
        print(f"  hamilton-raw : first flag day = {fhr}  flag-days = {nhr}/{E-S}")
        print(f"  hamilton-{HAMILTON_PERSIST}d  : first flag day = {fh}  flag-days = {nh}/{E-S}")
