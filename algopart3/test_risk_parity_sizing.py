"""Phase 2 (plan: reach 930): cross-sectional inverse-vol / risk-parity sizing. Currently every
traded idio name gets the full $10k position whenever sign(wz) != 0, regardless of its own
volatility -- a volatile name contributes much more portfolio variance than a calm one for the same
dollar exposure. This scales each name's size inversely with its own trailing realized vol (capped
at the existing $10k max, so this can only scale DOWN, never up), aiming to equalize risk
contribution rather than dollar contribution. Since score rewards Sharpe (mu/sd) directly, reducing
avoidable portfolio variance without touching the signal itself is a genuinely different lever from
anything tested this session (confirmed via grep: no risk-parity/inverse-vol/covariance test exists).

Reuses the WZ + significance-boost precompute pattern from test_boost_subparam_sweep.py (fresh
every day, matching production exactly) -- only the SIZING changes, not the signal.
"""
import numpy as np, pandas as pd, time
from scipy import stats
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


print("=== precompute (shared): shipped idio WZ, ALGO leg, significance-boost, per-name trailing vol ===")
t0 = time.time()
WZ = {}
for t in range(SAFE.WARMUP, nt):
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
    WZ[t] = wz
print(f"  WZ done ({time.time()-t0:.0f}s)")

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("  ALGO leg done")

n = rs.shape[0]
BOOST_AT = {}
t0 = time.time()
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
print(f"  significance-boost map done ({time.time()-t0:.0f}s)")

VOL_WIN = 20
vol_all = np.full((n, nt - 1), np.nan)  # vol_all[j, t] = trailing VOL_WIN-day realized vol of stock j, ending day t
for j in range(n):
    for t in range(VOL_WIN, nt - 1):
        vol_all[j, t] = rs[j, t - VOL_WIN:t].std()
print("  per-name trailing vol done")

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def build_pos(risk_parity=False, target_mode="median", vol_win=VOL_WIN):
    POS = np.zeros((nInst, nt))
    vol_local = vol_all
    if vol_win != VOL_WIN:
        vol_local = np.full((n, nt - 1), np.nan)
        for j in range(n):
            for t in range(vol_win, nt - 1):
                vol_local[j, t] = rs[j, t - vol_win:t].std()
    for k in range(SAFE.WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k].copy()
        if k >= BOOST_MIN_DAY:
            for j, bv in BOOST_AT[k].items():
                wz[j] += BOOST_K * bv
        if risk_parity and k - 1 < vol_local.shape[1]:
            vol_today = vol_local[:, k - 1]
            ok = ~np.isnan(vol_today) & (vol_today > 1e-8)
            if ok.sum() > 10:
                tgt = np.median(vol_today[ok]) if target_mode == "median" else np.mean(vol_today[ok])
                mult = np.ones(n)
                mult[ok] = np.clip(tgt / vol_today[ok], 0.0, 1.0)
            else:
                mult = np.ones(n)
        else:
            mult = np.ones(n)
        POS[1:, k] = np.clip(np.sign(wz) * mult * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E) for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<32}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


print("\n=== baseline (current shipped SAFE_llboost, no risk-parity sizing) ===")
base_scs = report("shipped SAFE_llboost", build_pos(risk_parity=False))

print("\n=== risk-parity sizing, target=median vs mean, default 20-day vol window ===")
report("risk-parity target=median", build_pos(risk_parity=True, target_mode="median"), base_scs)
report("risk-parity target=mean", build_pos(risk_parity=True, target_mode="mean"), base_scs)

print("\n=== risk-parity (target=median), sweep vol smoothing window ===")
for vw in (10, 15, 20, 30, 45, 60):
    report(f"risk-parity vol_win={vw}", build_pos(risk_parity=True, target_mode="median", vol_win=vw), base_scs)
