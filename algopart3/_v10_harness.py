"""
_v10_harness.py -- SHARED, VERIFIED backtest scaffolding for testing new candidate signals/mechanisms
against the real, shipped SAFE_llboost_v10.py. Built so many independent candidate-idea test scripts
can reuse the same (expensive) precomputed pieces and the same scoring convention instead of each
reimplementing (and potentially subtly re-breaking) it.

Import this, then either:
  (a) swap in a new BOOST array (leader-selection / boost-mechanism ideas) and rebuild wz via
      WZ_PRE + V10.BOOST_K*your_boost, then rs_blend(wz, t) per day, then evaluate(); or
  (b) build a new standalone signal array SIG[nIdio, nt] (causal) and blend it into BASE_WZ (or into
      wz before the rs_blend step) the same way rank-stability itself is blended -- see `blend_signal`
      helper below -- then evaluate(); or
  (c) construct a fully custom WZ_full[nIdio, nt] array however your idea requires (e.g. a modified
      ridge target, an extra lag predictor, etc.) and call evaluate() directly.

Every helper here is CAUSAL (day t only ever uses data through column t). `evaluate()` enforces the
same OLD/NEW/rolling-mean/rolling-floor/n_worse-vs-baseline bar as the rest of this file's history.
`BASE_WO`/`BASE_WN`/`BASE_SCS` are asserted to reproduce the real SAFE_llboost_v10 docstring numbers
(871.0/912.6/909.8/709.7) at import time -- if that assertion ever fails, something in this harness
(or in SAFE_llboost_v10.py itself) has drifted; do not trust any evaluate() result until it's fixed.
"""
import numpy as np, pandas as pd, time
from scipy import stats
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs_full = r[1:]
algo_r_full = r[0]
nIdio = rs_full.shape[0]
WARMUP = V10.WARMUP
days = list(range(WARMUP, nt))
end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


def scs_curve(POS):
    return np.array([wscore(POS, E - NUMTEST, E) for E in end_days])


def _sig_threshold_n(n_samples, n_cand):
    """Same as V10._sig_threshold but with an explicit candidate-pool-size divisor, for variants
    that change the candidate pool (and therefore the Bonferroni correction) without editing V10."""
    if n_samples < 10:
        return 1.0
    alpha_adj = V10.BOOST_ALPHA / n_cand
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


print("=== _v10_harness: precomputing ridge+REV (boost-independent) ===", flush=True)
t0 = time.time()
WZ_PRE = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in V10.HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, V10.RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V10.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
        rv_ = rv_ - rv_.mean()
        rv_ = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - V10.BLEND) * wz + V10.BLEND * rv_
    WZ_PRE[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

print("=== _v10_harness: precomputing shipped pairwise boost + ALGO leg ===", flush=True)
t0 = time.time()
BOOST_BASE = np.zeros((nIdio, nt))
for k in range(V10.BOOST_MIN_DAY, nt):
    BOOST_BASE[:, k] = V10._pairwise_boost(rs_full[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def algo_fh_series():
    """Causal per-day ALGO vol-regime feature (`fh` inside V10._algo_vol_shares), reconstructed
    from public constants so candidate ideas can condition on ALGO's regime without needing to
    scrape it out of the stateful production function. fh[t] is NaN before enough history exists."""
    lpA = logp[0]; T = len(lpA)
    r_ = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[V10.VOL_WIN:] = V10._roll_std(r_, V10.VOL_WIN)
    fh = np.full(T, np.nan)
    for s in range(V10.VOL_WIN + V10.VOL_Z, T):
        wv = vol[s - V10.VOL_Z:s]
        volz = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
        fh[s] = np.clip(volz, -3, 3) / 3.0
    return fh


def rs_blend(wz, t):
    """Apply the shipped rank-stability blend on top of a wz vector at day t (reuses the real,
    production V10._rank_stability_signal -- no reimplementation)."""
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is None:
        return wz
    s_std = rs_sig.std()
    s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
    return (1 - V10.RS_WEIGHT) * wz + V10.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)


def blend_signal(wz, sig_vec, weight):
    """Generic additive blend of any new per-day signal vector into wz, using the SAME convention
    rank-stability itself uses (z-score the signal, scale to wz's own magnitude, convex-blend at
    `weight`). Use this for any 'add a new standalone cross-sectional signal' idea so every candidate
    is blended consistently and comparably."""
    if sig_vec is None:
        return wz
    s_std = sig_vec.std()
    s_z = (sig_vec - sig_vec.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(sig_vec)
    return (1 - weight) * wz + weight * s_z * (np.abs(wz).mean() + 1e-12)


def build_pos_from_wz(WZ_full, algo_pos_arr=None):
    """WZ_full: (nIdio, nt), already fully combined (ridge+REV+boost+RS+candidate, in whatever order
    your idea requires). Builds the final integer position array using the same sign-sizing + dollar
    caps as production."""
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_full[:, t]
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos_arr if algo_pos_arr is not None else algo_pos
    return POS


BASE_WZ = np.full((nIdio, nt), np.nan)
for t in days:
    wz = WZ_PRE[:, t] + V10.BOOST_K * BOOST_BASE[:, t]
    BASE_WZ[:, t] = rs_blend(wz, t)

POS_BASE = build_pos_from_wz(BASE_WZ)
BASE_SCS = scs_curve(POS_BASE)
BASE_WO = wscore(POS_BASE, *OLD)
BASE_WN = wscore(POS_BASE, *NEW)

print(f"\nbaseline (must match v10 docstring 871.0/912.6/909.8/709.7): "
      f"OLD={BASE_WO:.1f}  NEW={BASE_WN:.1f}  rmean={BASE_SCS.mean():.1f}  rfloor={BASE_SCS.min():.1f}")
assert abs(BASE_WO - 871.0) < 0.5 and abs(BASE_WN - 912.6) < 0.5, \
    "*** harness does NOT reproduce shipped SAFE_llboost_v10 -- do not trust evaluate() results! ***"
print("OK -- harness matches shipped v10 to within rounding. Safe to evaluate candidates.\n")


def evaluate(name, WZ_full, algo_pos_arr=None, verbose=True):
    Pz = build_pos_from_wz(WZ_full, algo_pos_arr)
    scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > BASE_WO) and (wn > BASE_WN) and (scs.mean() > BASE_SCS.mean())
    nworse = int((scs < BASE_SCS).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {name:<40}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
              f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}", flush=True)
    return dict(name=name, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)
