"""
test_batch100_d60_boxcar.py

D60: replace the exponentially-weighted (EW) kernel in the idio ridge with a boxcar (flat trailing
window) kernel, for the SAME nominal half-life values (250, 500, 1000, 2000) reinterpreted as flat
window lengths (capped at whatever history is actually available -- causal). Distinct from the
shipped EW decay (weight = 0.5**(k/hl)); this gives every day inside the window equal weight and
zero weight outside it.

Everything else (BLEND reversal, pairwise boost, rank-stability blend, ALGO leg) is reused verbatim
from SAFE_llboost_v10, following the test_v20cand_idio_deadband.py house convention.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W


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


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

print("=== precompute: REV blend leg, pairwise boost, rank-stability signal, ALGO leg -- all "
      "UNCHANGED / reused verbatim from V10 (independent of the ridge-kernel mechanism under test) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

RS_SIG = np.full((nIdio, nt), np.nan)
for t in days:
    if t < max(RS_SHORT_W, RS_LONG_W) + 5:
        continue
    short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
    long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        continue
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    RS_SIG[:, t] = np.where(disagree, -sz, 0.0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def combine_wz(wz_ridge, t):
    wz = (1 - V10.BLEND) * wz_ridge + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_pos(WZ_RIDGE):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = combine_wz(WZ_RIDGE[:, t], t)
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, WZ_RIDGE, base_wo=None, base_wn=None, base_scs=None, verbose=True):
    Pz = build_pos(WZ_RIDGE); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = None
    if base_wo is not None:
        passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = None if base_scs is None else int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ("  <== fail" if passed is False else "")
        extra = f"  n_worse={nworse}/{len(scs)}" if nworse is not None else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}"
              f"{extra}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


def _ewls_ridge_boxcar(X, Y, W, a):
    """Same closed-form ridge solve as V10._ewls_ridge, but with a FLAT (boxcar) weight over the
    trailing min(W, n) samples instead of an exponential-decay kernel."""
    n, p = X.shape
    w = np.zeros(n)
    w[max(0, n - W):] = 1.0
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc)
    eps = 1e-8 * np.trace(XtWX) / p
    B = np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY)
    return B, mx, my


def build_wz_ridge(kernel):
    """kernel: 'ew' (baseline, must reproduce v10) or 'boxcar' (candidate)."""
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            if kernel == 'ew':
                B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            else:
                B, mx, my = _ewls_ridge_boxcar(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ[:, t] = np.mean(fs, 0)
    return WZ


print("\n=== sanity check: EW kernel (mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = build_wz_ridge('ew')
POS_base = build_pos(WZ_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
    SANITY_OK = False
else:
    print("  OK -- matches v10 to within rounding.")
    SANITY_OK = True


print("\n=== CANDIDATE: boxcar kernel, same nominal window lengths as HALF_LIVES (250,500,1000,2000) ===")
t0 = time.time()
WZ_BOX = build_wz_ridge('boxcar')
c = evaluate("boxcar (all 4 windows)", WZ_BOX, base_wo, base_wn, base_scs)
print(f"  [{time.time()-t0:.0f}s]")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
