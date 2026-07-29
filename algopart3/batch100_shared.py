"""
batch100_shared.py -- shared expensive precompute for the B33-B40 batch (all tested against
SAFE_llboost_v10). NOT a test_*.py file itself (imported by test_batch100_<id>.py scripts), so it
doesn't collide with the repo's test-file naming convention.

Reuses V10's own functions/constants verbatim (per house convention): _ewls_ridge,
_beta_adjusted_target, _pairwise_boost, _algo_vol_shares, _rank_stability_signal, and every relevant
constant. Builds, once:
  - WZ_PRE   : ridge ensemble (beta-adjusted target) + BLEND reversion, BEFORE boost / rank-stability
               (independent of anything any single B3x idea changes about boost/RS -- the expensive part)
  - BOOST, LEADER_ID : pairwise boost value AND which candidate index was selected, per idio name/day
               (LEADER_ID needed for B33's persistence check; unchanged from V10._pairwise_boost otherwise)
  - RS_SIG   : the rank-stability raw signal (pre-standardization), per idio name/day
  - algo_pos : the ALGO leg, unaffected by anything tested in this batch
  - WZ_FULL  : the ACTUAL v10 final wz (WZ_PRE + BOOST_K*boost, then RS blend) -- what v10 actually trades
  - POS_BASE : the full (nInst, nt) v10 position array, built from WZ_FULL + algo_pos

Every B3x script imports this module, then only recomputes the (cheap) idea-specific piece.
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
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W = (
    V10.BOOST_N_CANDIDATES, V10.BOOST_IC_L, V10.BOOST_P, V10.BOOST_SCALE_W)
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

CACHE_PATH = "batch100_B33B40_cache.npz"
_cache_ok = False
if os.path.exists(CACHE_PATH):
    try:
        _c = np.load(CACHE_PATH)
        if int(_c["nt"]) == nt and int(_c["nIdio"]) == nIdio:
            REV, algo_pos, WZ_PRE, BOOST, LEADER_ID = (
                _c["REV"], _c["algo_pos"], _c["WZ_PRE"], _c["BOOST"], _c["LEADER_ID"])
            _cache_ok = True
            print(f"=== batch100_shared: loaded expensive precompute from {CACHE_PATH} (cache hit) ===",
                  flush=True)
    except Exception as e:
        print(f"  cache load failed ({e}), recomputing", flush=True)

if not _cache_ok:
    print("=== batch100_shared: precompute (BLEND reversion, ridge WZ w/ beta-adjusted target, boost + "
          "leader identity, rank-stability signal, ALGO leg) -- shared across every B3x idea ===", flush=True)
    t0 = time.time()

    REV = np.zeros((nIdio, nt))
    for t in days:
        rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
        rv_ = rv_ - rv_.mean()
        REV[:, t] = -rv_ / (rv_.std() + 1e-12)

    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

    WZ_PRE = np.full((nIdio, nt), np.nan)  # ridge ensemble + BLEND reversion, BEFORE boost / rank-stability
    for t in days:
        rr_ = r[:, :t]
        X = rr_[:, :-1].T
        Y = V10._beta_adjusted_target(rr_)
        xq = rr_[:, -1]
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        WZ_PRE[:, t] = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    print(f"  WZ_PRE done ({time.time()-t0:.0f}s)", flush=True)

    # --- boost values AND leader identity per day (identical math to V10._pairwise_boost, but also
    # records which candidate index was selected, needed by B33's persistence check) ---
    t0 = time.time()
    BOOST = np.zeros((nIdio, nt))
    LEADER_ID = np.full((nIdio, nt), -1, dtype=int)  # -1 = no significant leader that day
    for t in range(BOOST_MIN_DAY, nt):
        rsl = rs[:, :t]
        n, T = rsl.shape
        Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
        n_samples = Xi_full.shape[1]
        thr = V10._sig_threshold(n_samples)
        vol_causal = np.nanstd(Xi_full, axis=1)
        cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
        Xi = Xi_full[cand_idx]
        C = V10._corrmat(Xi, Yj)
        for j in range(n):
            col = C[:, j].copy()
            cp = np.where(cand_idx == j)[0]
            if len(cp): col[cp[0]] = np.nan
            if np.all(np.isnan(col)): continue
            ci = int(np.nanargmax(np.abs(col)))
            if abs(col[ci]) <= thr: continue
            i = cand_idx[ci]
            lead = rsl[i]
            scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
            lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
            a = max(0, T - 1 - BOOST_IC_L)
            xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
            ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
            if ic <= 0: continue
            BOOST[j, t] = lead_boost[-1]
            LEADER_ID[j, t] = i
    print(f"  BOOST + LEADER_ID done ({time.time()-t0:.0f}s)", flush=True)

    np.savez(CACHE_PATH, nt=nt, nIdio=nIdio, REV=REV, algo_pos=algo_pos, WZ_PRE=WZ_PRE,
             BOOST=BOOST, LEADER_ID=LEADER_ID)
    print(f"  cached precompute to {CACHE_PATH}", flush=True)

# --- rank-stability raw signal (pre-standardization), per idio name/day ---
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


def rs_blend(wz, t):
    """Apply v10's rank-stability blend to a raw wz at day t (matches getMyPosition exactly)."""
    s = RS_SIG[:, t]
    if not np.isfinite(s).all():
        return wz
    sstd = s.std()
    s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
    return (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)


# --- WZ_FULL: the ACTUAL v10 final wz (WZ_PRE + BOOST_K*boost, then RS blend) ---
WZ_FULL = np.full((nIdio, nt), np.nan)
for t in days:
    wz = WZ_PRE[:, t].copy()
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    wz = rs_blend(wz, t)
    WZ_FULL[:, t] = wz


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


POS_BASE = build_pos_from_wz(WZ_FULL)
base_scs = scs_curve(POS_BASE)
base_wo, base_wn = wscore(POS_BASE, *OLD), wscore(POS_BASE, *NEW)
print(f"\n=== sanity check (shared): must reproduce SAFE_llboost_v10 exactly ===")
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


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
