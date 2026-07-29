"""
test_batch100_interactions.py

Batch-100 ideas D55-D56: both ADD interaction-term features to the idio ridge predictor set,
leaving BLEND reversion, the pairwise boost, the rank-stability blend, and the ALGO leg untouched.
Shares one expensive precompute (REV, BOOST via V10._pairwise_boost verbatim, RS_SIG, algo_pos --
none of these depend on the ridge predictor set) across both variants; only the ridge stage itself
(_ewls_ridge call w/ an augmented X) is redone per variant, exactly reusing V10._ewls_ridge and
V10._beta_adjusted_target verbatim -- only the X matrix passed in changes, which is exactly what
these two ideas require changing.

D55: add name_i return * ALGO same-day return interaction features (one per idio name) to the
     ridge predictor set (doubling p from nInst to nInst+nIdio).
D56: add pairwise interaction features (name_i * name_j returns) for just the top-10 highest
     TRAILING-vol names (causal, recomputed at each t) -- C(10,2)=45 extra predictor columns.
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

print("=== precompute: BLEND reversion + pairwise boost (V10._pairwise_boost verbatim) + raw "
      "rank-stability signal + ALGO leg -- all independent of the ridge predictor set ===",
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


def build_pos_from_wz_pre(WZ_PRE):
    """WZ_PRE: ridge ensemble output (BEFORE BLEND) -- apply BLEND, boost, rank-stability, ALGO."""
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = (1 - V10.BLEND) * WZ_PRE[:, t] + V10.BLEND * REV[:, t]
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def ridge_wz_pre_baseline():
    """Verbatim V10 ridge stage (raw X = all-name lag returns), for the sanity check."""
    WZ_PRE = np.full((nIdio, nt), np.nan)
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
        WZ_PRE[:, t] = np.mean(fs, 0)
    return WZ_PRE


def evaluate(nm, WZ_PRE, base_wo=None, base_wn=None, base_scs=None, verbose=True):
    Pz = build_pos_from_wz_pre(WZ_PRE); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = None if base_wo is None else (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum()) if base_scs is not None else None
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<32}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
              f"rfloor={scs.min():7.1f}  n_worse={nworse}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== sanity check: verbatim ridge stage (no interaction features) must reproduce v10 ===")
t0 = time.time()
WZ_PRE_BASE = ridge_wz_pre_baseline()
base = evaluate("baseline (verbatim v10)", WZ_PRE_BASE, verbose=False)
print(f"  baseline: OLD={base['wo']:.1f}  NEW={base['wn']:.1f}  rmean={base['rm']:.1f}  "
      f"rfloor={base['rf']:.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(base['wo'] - 871.0) < 0.5 and abs(base['wn'] - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
base_wo, base_wn = base['wo'], base['wn']
base_scs = scs_curve(build_pos_from_wz_pre(WZ_PRE_BASE))


def report(nm, WZ_PRE):
    return evaluate(nm, WZ_PRE, base_wo, base_wn, base_scs)


# ============================================================================
# D55: name_i return * ALGO same-day return interaction features
# ============================================================================
def ridge_wz_pre_d55():
    WZ_PRE = np.full((nIdio, nt), np.nan)
    for t in days:
        rr_ = r[:, :t]
        X_raw = rr_[:, :-1].T          # (t-1, nInst)
        algo_row = rr_[0, :-1]          # (t-1,)  ALGO's return, same days as X_raw's rows
        X_int = rr_[1:, :-1].T * algo_row[:, None]   # (t-1, nIdio) name_i * ALGO same-day
        X = np.concatenate([X_raw, X_int], axis=1)
        Y = V10._beta_adjusted_target(rr_)
        xq_raw = rr_[:, -1]
        xq_int = rr_[1:, -1] * rr_[0, -1]
        xq = np.concatenate([xq_raw, xq_int])
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ_PRE[:, t] = np.mean(fs, 0)
    return WZ_PRE


print("\n=== D55: add name_i * ALGO same-day return interaction features to ridge predictor set ===")
t0 = time.time()
WZ_D55 = ridge_wz_pre_d55()
d55_result = report("D55 idio*ALGO interactions", WZ_D55)
print(f"    [{time.time()-t0:.0f}s]")


# ============================================================================
# D56: pairwise interaction features among top-10 highest TRAILING-vol names
# ============================================================================
def ridge_wz_pre_d56(topn=10, vol_win=60):
    WZ_PRE = np.full((nIdio, nt), np.nan)
    pairs_cache = None
    for t in days:
        rr_ = r[:, :t]
        X_raw = rr_[:, :-1].T          # (t-1, nInst)
        # causal trailing vol of each idio name, using history up to (not incl.) the query day
        lo = max(0, t - 1 - vol_win)
        trail = rs[:, lo:t - 1]
        if trail.shape[1] < 20:
            X = X_raw
            xq = rr_[:, -1]
        else:
            voln = trail.std(axis=1)
            top_idx = np.argsort(-voln)[:topn]
            pairs = [(a, b) for ii, a in enumerate(top_idx) for b in top_idx[ii + 1:]]
            idio_ret_hist = rr_[1:, :-1]     # (nIdio, t-1)
            X_int = np.stack([idio_ret_hist[a] * idio_ret_hist[b] for (a, b) in pairs], axis=1) \
                if pairs else np.zeros((rr_.shape[1] - 1, 0))
            X = np.concatenate([X_raw, X_int], axis=1)
            idio_ret_q = rr_[1:, -1]
            xq_int = np.array([idio_ret_q[a] * idio_ret_q[b] for (a, b) in pairs]) if pairs \
                else np.zeros(0)
            xq = np.concatenate([rr_[:, -1], xq_int])
        Y = V10._beta_adjusted_target(rr_)
        fs = []
        for hl in HALF_LIVES:
            B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
            pred = my + (xq - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        WZ_PRE[:, t] = np.mean(fs, 0)
    return WZ_PRE


print("\n=== D56: pairwise interaction features (top-10 highest trailing-vol idio names, 45 pairs) "
      "added to ridge predictor set ===")
t0 = time.time()
WZ_D56 = ridge_wz_pre_d56()
d56_result = report("D56 top10-pairwise interactions", WZ_D56)
print(f"    [{time.time()-t0:.0f}s]")


print("\n" + "=" * 90)
print("SUMMARY (all vs v10 baseline OLD=%.1f NEW=%.1f rmean=%.1f)" % (base_wo, base_wn, base_scs.mean()))
print("=" * 90)
for c in [d55_result, d56_result]:
    tag = "PASS" if c["passed"] else "fail"
    print(f"  [{tag}] {c['name']:<32} OLD={c['wo']:7.1f} NEW={c['wn']:7.1f} rmean={c['rm']:7.1f} "
          f"rfloor={c['rf']:7.1f} n_worse={c['nworse']}/61")
