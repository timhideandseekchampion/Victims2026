"""Items 1, 2, 4 [MECH] + item 3 [SIGNAL/MECH hybrid]: alternative ridge estimation techniques.
1. Kalman-filter / recursive-least-squares (RLS) coefficients instead of the fixed exp-weighted
   ridge ensemble -- continuously adapting state instead of a fixed half-life blend.
2. Winsorized returns (clip at trailing 1st/99th percentile) before fitting, instead of the
   full rank-transform that failed catastrophically.
3. Quantile-regression forecast (median target) instead of ridge's mean-squared-error target.
4. PCA pre-reduction (project the 51-instrument panel onto top-K components) before the ridge fit.
All compared against the current shipped SAFE_llboost.py on the standard five-metric bar.
"""
import numpy as np, pandas as pd, time
from scipy import stats
from sklearn.linear_model import QuantileRegressor
import SAFE, SAFE_llvol

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]

BOOST_MIN_DAY = 500
ALPHA = 0.05
N_CANDIDATES = 49
BOOST_P = 2.0
BOOST_SCALE_W = 1000
BOOST_IC_L = 190
BOOST_K = 1.5


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== precompute (fixed): significance-gated boost + ALGO leg (shared across all ridge variants) ===")
t0 = time.time()
n = rs.shape[0]
BOOST_AT = {}
for k in range(BOOST_MIN_DAY, nt):
    T = k
    Xi = rs[:, :T - 1]; Yj = rs[:, 1:T]
    n_samples = Xi.shape[1]
    thr = sig_threshold(n_samples)
    C = corrmat(Xi, Yj)
    entry = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        if abs(col[i]) <= thr:
            continue
        lead = rs[i, :T]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        entry[j] = lead_boost[-1]
    BOOST_AT[k] = entry
print(f"  boost map done ({time.time()-t0:.0f}s)")

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("  ALGO leg done")

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def combine_and_score(wz_fn):
    """wz_fn(t) -> wz array (50,) for the idio ridge forecast at day t. Adds boost, scores."""
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = wz_fn(k)
        if k >= BOOST_MIN_DAY:
            for j, bv in BOOST_AT[k].items():
                wz[j] += BOOST_K * bv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<28}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


def wz_shipped(t):
    rr = r[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    return wz


print("\n=== baseline: current shipped ridge (4-half-life ensemble) ===")
base_scs = report("shipped ridge (baseline)", combine_and_score(wz_shipped))

print("\n\n### Item 2 [MECH]: winsorized returns before ridge fit (trailing-window z-clip proxy) ###")


def roll_mean_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return m, np.sqrt(v)


WINS_W = 500


def make_r_wins(K):
    r_w = r.copy()
    for i in range(nInst):
        m, s = roll_mean_std(r[i], WINS_W)
        # m[k]/s[k] = stats over x[k:k+WINS_W]; drop the last so it aligns as a trailing
        # window ending the day BEFORE t (causal, excludes today's own value)
        m, s = m[:-1], s[:-1]
        lo = m - K * s; hi = m + K * s
        seg = r_w[i, WINS_W:]
        r_w[i, WINS_W:] = np.clip(seg, lo, hi)
    return r_w


def wz_winsorized(t, r_w):
    rr = r_w[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    return wz


for K in (2.0, 2.5, 3.0, 4.0):
    r_w = make_r_wins(K)
    POS = combine_and_score(lambda t, r_w=r_w: wz_winsorized(t, r_w))
    report(f"winsorize K={K}", POS, base_scs)

print("\n\n### Item 1 [MECH]: Kalman filter (diagonal covariance, random-walk process noise Q) ###")
print("    genuinely different from EW-ridge: coefficients drift stochastically over time")
print("    rather than being re-estimated fresh each call with a fixed forgetting weighting.")

R_OBS = float(np.var(rs))
print(f"    R_obs (empirical idio-return variance) = {R_OBS:.6f}")


def kalman_wz_series(q_frac, pdiag0=1.0):
    Q = R_OBS * q_frac
    ntgt = rs.shape[0]  # 50
    B = np.zeros((ntgt, nInst))
    Pdiag = np.full((ntgt, nInst), pdiag0)
    WZ = {}
    last_col = r.shape[1] - 1
    for k in range(0, nt - 1):
        x = r[:, k]
        pred = B @ x
        t_out = k + 1
        if t_out >= SAFE.WARMUP:
            fi = pred - pred.mean()
            wz = fi / (fi.std() + 1e-12)
            WZ[t_out] = wz
        if k + 1 > last_col:
            break
        y = r[1:, k + 1]
        Pdiag = Pdiag + Q
        Sigma = (Pdiag * (x ** 2)[None, :]).sum(1) + R_OBS
        Kg = Pdiag * x[None, :] / Sigma[:, None]
        resid = y - pred
        B = B + Kg * resid[:, None]
        Pdiag = (1 - Kg * x[None, :]) * Pdiag
    return WZ


def wz_kalman(t, WZK):
    wz = WZK[t].copy()
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    return wz


for q_frac in (1e-5, 1e-4, 1e-3, 1e-2):
    t0 = time.time()
    WZK = kalman_wz_series(q_frac)
    POS = combine_and_score(lambda t, WZK=WZK: wz_kalman(t, WZK))
    report(f"kalman q_frac={q_frac:g} ({time.time()-t0:.0f}s)", POS, base_scs)

print("\n\n### Item 4 [MECH]: PCA pre-reduction (project 51-name return panel onto top-K PCs) ###")


def make_wz_pca(K, refit_freq=20):
    state = {"V": None, "day": -1}

    def wz_pca(t):
        rr = r[:, :t]
        if state["V"] is None or t - state["day"] >= refit_freq:
            X = rr.T
            Xc = X - X.mean(0)
            cov = Xc.T @ Xc / X.shape[0]
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            state["V"] = eigvecs[:, order[:K]]
            state["day"] = t
        V = state["V"]
        fs = []
        Xp = rr[:, :-1].T @ V
        Yp = rr[1:, 1:].T
        x_today_p = rr[:, -1] @ V
        for hl in SAFE.HALF_LIVES:
            B, mx, my = SAFE._ewls_ridge(Xp, Yp, hl, SAFE.RIDGE_A)
            pred = my + (x_today_p - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if SAFE.BLEND > 0:
            rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
            rv_ = rv_ - rv_.mean()
            rv = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
        return wz
    return wz_pca


for K in (5, 10, 20, 30, 40):
    t0 = time.time()
    fn = make_wz_pca(K)
    POS = combine_and_score(fn)
    report(f"PCA top-{K} PCs ({time.time()-t0:.0f}s)", POS, base_scs)
