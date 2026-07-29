"""
batch100_d6x_shared.py -- shared expensive precompute for the D61/D62/D63/D64 batch (all tested
against SAFE_llboost_v10). NOT a test_*.py file itself (imported by test_batch100_D6*.py scripts),
so it doesn't collide with the repo's test-file naming convention. Follows the same caching pattern
as batch100_shared.py.

Reuses V10's own functions/constants verbatim (per house convention): _ewls_ridge,
_beta_adjusted_target, _pairwise_boost, _algo_vol_shares. Builds, once (cached to disk):
  - REV       : the BLEND reversal leg (identical to V10, independent of every D6x idea -- all 4 ideas
                only touch the ridge stage, not REV/boost/RS/ALGO)
  - BOOST     : the pairwise boost value per idio name/day (V10._pairwise_boost, unchanged)
  - algo_pos  : the ALGO leg, unaffected by anything tested in this batch
  - RS_SIG    : the rank-stability raw signal (pre-standardization), per idio name/day
  - WZ_BASE   : the baseline (unmodified) ridge-ensemble wz, i.e. exactly what SAFE_llboost_v10's
                ridge stage computes -- used by every D6x script's sanity check
  - POS_BASE, base_wo, base_wn, base_scs : the full v10 baseline position + its score, so every D6x
                script's sanity check is a single 3-number comparison, not a re-run.

Every D6x script imports this module, then only recomputes the (idea-specific) ridge stage and calls
combine_wz/build_pos/evaluate on top of the shared REV/BOOST/RS_SIG/algo_pos.
"""
import os
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
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT


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

days = list(range(WARMUP, nt))

CACHE_PATH = "batch100_d6x_cache.npz"
_cache_ok = False
if os.path.exists(CACHE_PATH):
    try:
        _c = np.load(CACHE_PATH)
        if int(_c["nt"]) == nt and int(_c["nIdio"]) == nIdio:
            REV, BOOST, algo_pos, RS_SIG, WZ_BASE = (
                _c["REV"], _c["BOOST"], _c["algo_pos"], _c["RS_SIG"], _c["WZ_BASE"])
            _cache_ok = True
            print(f"=== batch100_d6x_shared: loaded expensive precompute from {CACHE_PATH} (cache hit) ===",
                  flush=True)
    except Exception as e:
        print(f"  cache load failed ({e}), recomputing", flush=True)

if not _cache_ok:
    print("=== batch100_d6x_shared: precompute (BLEND reversion, pairwise boost, rank-stability signal, "
          "ALGO leg, baseline ridge WZ) -- shared across every D6x idea ===", flush=True)
    t0 = time.time()

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

    WZ_BASE = np.full((nIdio, nt), np.nan)
    for t in days:
        rr_ = r[:, :t]
        Y = V10._beta_adjusted_target(rr_)
        X = rr_[:, :-1].T
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ_BASE[:, t] = np.mean(fs, 0)

    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    try:
        np.savez(CACHE_PATH, nt=nt, nIdio=nIdio, REV=REV, BOOST=BOOST, algo_pos=algo_pos,
                 RS_SIG=RS_SIG, WZ_BASE=WZ_BASE)
    except Exception as e:
        print(f"  cache save failed ({e}) -- continuing without cache", flush=True)


def combine_wz(wz_ridge, t):
    """wz_ridge: the (nIdio,) idea-specific ridge-stage output for day t. Applies BLEND reversal,
    pairwise boost, and rank-stability blend exactly as SAFE_llboost_v10.getMyPosition does."""
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


POS_BASE = build_pos(WZ_BASE)
base_scs = scs_curve(POS_BASE)
base_wo, base_wn = wscore(POS_BASE, *OLD), wscore(POS_BASE, *NEW)
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print(f"=== batch100_d6x_shared baseline (WZ_BASE, unmodified ridge): OLD={base_wo:.1f}  NEW={base_wn:.1f}  "
      f"rmean={base_scs.mean():.1f}  rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7) "
      f" SANITY_OK={SANITY_OK} ===", flush=True)
