"""
kalman_tvp.py — the decisive ADAPTATION test (research Rank 1-2): model the lead-lag matrix B_t
as a random-walk state and Kalman-filter it, tuning the drift rate q by MARGINAL LIKELIHOOD.
  * Key diagnostic: is q_hat ~ 0? If so, B is STATIONARY -> adaptation is a trap (Sharpe swings
    are realized-return variance, not regime drift). If q_hat > 0, drift is real and estimable.
  * Efficient: the regressor x_t (lagged 51-return cross-section) is SHARED across all 50 targets,
    so one covariance recursion P (51x51) + a matrix state B (51x50).
  * Compare Kalman-TVP forecast IC (per-window, consistency) vs the fixed forgetting-ridge.
"""
import numpy as np, pandas as pd
from scipy import stats
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]                 # (51, nt-1)
X = RET[:, :-1].T                                              # regressor at tau     (nt-2, 51)
Y = RET[1:, 1:].T                                             # target tau+1 (idio)  (nt-2, 50)
p = 51; ntar = 50

def kalman_run(q, sigma2=None, ret_forecasts=False, upto=None):
    """Run shared-regressor Kalman over rows 0..upto. Returns (loglik) or (loglik, forecasts dict t->f)."""
    n = X.shape[0] if upto is None else upto
    if sigma2 is None: sigma2 = np.var(Y[:200])
    B = np.zeros((p, ntar)); P = np.eye(p) * 1.0
    ll = 0.0; fc = {}
    for tau in range(n):
        x = X[tau]; y = Y[tau]
        P = P + q * np.eye(p)                                 # additive drift inflation
        xP = x @ P                                            # (p,)
        S = x @ xP + sigma2                                   # scalar predictive var
        yhat = x @ B                                          # (50,) prediction
        e = y - yhat
        # marginal loglik (shared S across targets)
        ll += -0.5 * (ntar * np.log(2 * np.pi * S) + (e @ e) / S)
        K = xP / S                                            # (p,)
        B = B + np.outer(K, e)                                # rank-1 update of all targets
        P = P - np.outer(K, xP)
        if ret_forecasts:
            # forecast for the NEXT step uses updated B and the next regressor; store B row usage at decision time
            fc[tau] = (B.copy())
    return (ll, fc) if ret_forecasts else ll

# ---- (1) tune q by marginal likelihood on an early stretch (tau up to ~day 500) ----
sig2 = np.var(Y[100:400])
upto_fit = 498                                                 # ~ up to day 500
print("Marginal-likelihood over drift rate q (relative to sigma^2 = %.2e):" % sig2)
print(f"  {'q/sigma2':>10}{'loglik':>14}")
grid = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
best = None
for r in grid:
    ll = kalman_run(r * sig2, sig2, upto=upto_fit)
    print(f"  {r:>10.0e}{ll:>14.1f}")
    if best is None or ll > best[0]: best = (ll, r)
print(f"  -> q_hat/sigma2 = {best[1]:.0e}   (q_hat~0 => B STATIONARY => adaptation is a trap)")

# refine near best with a finer grid
lo = max(best[1] / 10, 0.0); hi = best[1] * 10 if best[1] > 0 else 1e-3
fine = np.unique(np.concatenate([[0.0], np.geomspace(max(lo,1e-6), max(hi,1e-3), 7)]))
bf = best
for r in fine:
    ll = kalman_run(r * sig2, sig2, upto=upto_fit)
    if ll > bf[0]: bf = (ll, r)
q_hat = bf[1] * sig2
print(f"  refined q_hat/sigma2 = {bf[1]:.2e}\n")

# ---- (2) Kalman-TVP forecast vs fixed ridge: per-window IC consistency ----
def ridge_fc_at(t, hl=2000, a=0.3):
    lpp = lp[:, :t]; r = lpp[:, 1:] - lpp[:, :-1]
    Xt = r[:, :-1].T; Yt = r[1:, 1:].T; xin = r[:, -1]
    n = Xt.shape[0]; lam = 0.5 ** (1.0 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * Xt).sum(0) / sw; my = (w[:, None] * Yt).sum(0) / sw
    Xc = Xt - mx; Yc = Yt - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(51), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B; return f - f.mean()

# precompute Kalman B trajectory once (causal): B after processing tau predicts return at tau+1 (=day tau+2)
_, fcB = kalman_run(q_hat, sig2, ret_forecasts=True)
def kalman_fc_at(t):
    # decision at day t: last observed return is RET[:, t-2] = X row (t-2). B filtered through tau=t-2.
    tau = t - 2
    if tau not in fcB: return None
    Bt = fcB[tau]; x = RET[:, t - 2]                          # most recent observed return
    f = x @ Bt; return f - f.mean()

def ic_window(fc_at, S, E):
    ics = []
    for t in range(max(S, 120), min(E, nt - 1)):
        s = fc_at(t)
        if s is None: continue
        fwd = RET[1:, t]                                      # move into day t+1
        # align: forecast made at day t predicts RET[:, t] (next-day). ridge_fc_at uses xin=RET[:,t-1]? check
        if s.std() > 1e-12 and fwd.std() > 1e-12: ics.append(np.corrcoef(s, fwd)[0, 1])
    ics = np.array(ics); n = len(ics)
    t = ics.mean() / (ics.std(ddof=1) / np.sqrt(n)); pv = stats.t.sf(t, n - 1)
    return ics.mean(), t, pv

legs = [(S, S + 250) for S in range(250, 501, 50)]
print("Per-window IC / t / p — fixed ridge(hl2000) vs Kalman-TVP(q_hat):")
print(f"  {'leg':<12}{'ridge IC':>10}{'t':>6}{'  ':>2}{'TVP IC':>10}{'t':>6}")
ric = []; tic = []
for S, E in legs:
    ri, rt, rp = ic_window(ridge_fc_at, S, E)
    ti, tt, tp = ic_window(kalman_fc_at, S, E)
    ric.append(ri); tic.append(ti)
    print(f"  {f'{S}-{E}':<12}{ri:10.4f}{rt:6.1f}  {ti:10.4f}{tt:6.1f}")
print(f"  ridge: mean {np.mean(ric):.4f} std {np.std(ric):.4f}   TVP: mean {np.mean(tic):.4f} std {np.std(tic):.4f}")
print("\nverdict: if TVP IC <= ridge IC and q_hat~0, adaptation adds nothing (B stationary).")
