"""
batch100_common_gi.py -- shared helper for the G81-I88 batch (all tested against SAFE_llboost_v10).

Reuses the pre-built batch100_cache.npz (built by precompute_batch100.py: per-half-life ridge
forecasts FS, REV leg, WZ_RIDGE, WZ_PRE, BOOST, algo_pos, RS_RAW, WZ_V10) instead of recomputing the
expensive ridge-ensemble / pairwise-boost loops from scratch, per house convention (reuse the
expensive shared precompute -- see test_v19cand_boost_ncandidates.py). Every idea script in this
batch imports this module and only recomputes the cheap, idea-specific piece.
"""
import numpy as np, pandas as pd
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
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W, BOOST_ALPHA = (
    V10.BOOST_N_CANDIDATES, V10.BOOST_IC_L, V10.BOOST_P, V10.BOOST_SCALE_W, V10.BOOST_ALPHA)

CACHE = np.load("batch100_cache.npz")
FS = CACHE["FS"]; REV = CACHE["REV"]; WZ_RIDGE = CACHE["WZ_RIDGE"]; WZ_PRE = CACHE["WZ_PRE"]
BOOST = CACHE["BOOST"]; algo_pos = CACHE["algo_pos"]; RS_RAW = CACHE["RS_RAW"]; WZ_V10 = CACHE["WZ_V10"]
days = list(CACHE["days"])

# v9's final wz -- ridge ensemble + BLEND + boost are IDENTICAL between v9 and v10 (v10 only adds the
# RS blend on top), so WZ_PRE + BOOST_K*BOOST (i.e. WZ_V10 before the RS blend step) IS v9's traded wz.
WZ_V9 = WZ_PRE + BOOST_K * BOOST


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


def build_pos_from_wz(WZ):
    """WZ: (nIdio, nt) final per-name signal -> full (nInst, nt) position array (sign-sized, capped),
    with the ALGO leg attached unchanged."""
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


POS_BASE = build_pos_from_wz(WZ_V10)
base_scs = scs_curve(POS_BASE)
base_wo, base_wn = wscore(POS_BASE, *OLD), wscore(POS_BASE, *NEW)
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5


def print_sanity(tag=""):
    print(f"=== sanity check {tag}: must reproduce SAFE_llboost_v10 exactly ===")
    print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
          f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
    print("  OK -- matches v10 to within rounding." if SANITY_OK else
          "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
    return SANITY_OK


def evaluate(nm, POS, verbose=True):
    scs = scs_curve(POS)
    wo = wscore(POS, *OLD); wn = wscore(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<34}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


if __name__ == "__main__":
    print_sanity()
