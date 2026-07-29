"""
test_batch100_B32_signedboost.py

B32: Re-test signed boost (invert negative-IC leader pairs instead of discarding) against v10, with
the current N=39 candidate pool and rank-stability layered on top. Originally tested
(test_v13cand_signed_boost.py) against SAFE_llboost_v9 -- BEFORE the rank-stability blend (v10)
existed. Re-verifying directly against the current best.

MECHANISM (identical to test_v13cand_signed_boost.py): `_pairwise_boost` selects each follower's
leader by strongest absolute correlation (already symmetric to sign), but discards the pair if the
realized boost IC is non-positive (`if ic <= 0: continue`). The fix under test: when `ic < 0`, INVERT
the boost (`-lead_boost[-1]`) instead of discarding -- can only ever ADD boosts previously discarded,
so any score change is attributable to what was previously thrown away. allow_negative=False must
reproduce SAFE_llboost_v10 exactly (mandatory sanity check).

REV/rank-stability/ALGO leg cached once (independent of the boost's sign rule) and reused across every
config.
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
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W = (
    V10.BOOST_N_CANDIDATES, V10.BOOST_IC_L, V10.BOOST_P, V10.BOOST_SCALE_W)
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


def _pairwise_boost_signed(rsl, allow_negative, mag_floor=0.0):
    """Identical to V10._pairwise_boost's leader SELECTION and SIGNIFICANCE gate (verbatim, N=39).
    Only the final ic-sign handling changes: allow_negative=False reproduces V10 exactly."""
    n, T = rsl.shape
    boost = np.zeros(n)
    if T < BOOST_MIN_DAY:
        return boost
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = V10._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V10._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if not allow_negative:
            if ic <= 0:
                continue
            boost[j] = lead_boost[-1]
        else:
            if abs(ic) <= mag_floor:
                continue
            boost[j] = lead_boost[-1] if ic > 0 else -lead_boost[-1]
    return boost


print("=== precompute: BLEND reversion, ridge ensemble (beta-adjusted target), rank-stability signal, "
      "ALGO leg -- IDENTICAL regardless of the boost's sign rule, cached once ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_PRE = np.full((nIdio, nt), np.nan)   # ridge ensemble + BLEND reversion, BEFORE boost / rank-stability
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

RS_SIG = np.full((nIdio, nt), np.nan)
for t in days:
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        RS_SIG[:, t] = rs_sig
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def build_pos(allow_negative, mag_floor=0.0):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            boost = _pairwise_boost_signed(rs[:, :t], allow_negative, mag_floor)
            wz = wz + BOOST_K * boost
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: allow_negative=False must reproduce SAFE_llboost_v10 exactly ===")
POS_base = build_pos(False)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, allow_negative, mag_floor=0.0, verbose=True):
    Pz = build_pos(allow_negative, mag_floor); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<24}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, allow_negative=allow_negative, mag_floor=mag_floor, wo=wo, wn=wn,
                rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print("\n=== PRIMARY TEST: allow_negative=True (pure sign rule, matches shipped code's own bar) ===")
evaluate("allow_negative=True", True, 0.0)

print("\n=== sweep: require a minimum |ic| before trusting the flip (avoid near-zero noise flips) ===")
results = [evaluate(f"mag_floor={mf}", True, mf) for mf in (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    best = max(passing, key=lambda c: c["rm"])
    print(f"best by rmean: mag_floor={best['mag_floor']}  rmean={best['rm']:.1f}  n_worse={best['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"]):
        print(f"  mag_floor={c['mag_floor']:<5} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")
