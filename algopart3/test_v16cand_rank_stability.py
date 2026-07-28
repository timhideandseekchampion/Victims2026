"""
test_v16cand_rank_stability.py

CANDIDATE (external suggestion #4, reconstructed from a heavily-truncated description -- flagged
honestly): a signal named "rank_stability_short8_long15" from unspecified prior research, described
only as: "bought medium-term leaders after short-term pullbacks and shorted medium-term laggards
after short-term rebounds." The exact construction (what's ranked, over what universe, how
"short8"/"long15" map to lookback windows) was never visible -- this is a best-effort reconstruction,
NOT a verified replica, and should be read with that caveat throughout.

RECONSTRUCTION: SHORT_W=8, LONG_W=15 (the two numbers in the name). For each idio name:
  long_z[i]  = z-score of its LONG_W-day (15d) return, cross-sectionally across the 50 idio names
               (positive = medium-term "leader", negative = medium-term "laggard")
  short_z[i] = z-score of its SHORT_W-day (8d) return, cross-sectionally
The described trade only fires when the short-term move goes AGAINST the medium-term trend --
leader+pullback (long_z>0, short_z<0) -> buy; laggard+rebound (long_z<0, short_z>0) -> sell. When the
signs AGREE (leader still rising / laggard still falling), the description implies no clear edge, so
no signal. Algebraically, sign(long_z)*|short_z| when sign(long_z) != sign(short_z) reduces exactly
to -short_z (since sign(long_z) = -sign(short_z) in that case) -- i.e. this construction is a PURE
short-term-reversal bet, gated to fire only when it opposes the medium-term trend:
  signal[i] = -short_z[i]  if sign(long_z[i]) != sign(short_z[i])  else 0

Distinct from the shipped reversal leg (BLEND=0.3, REV_W=10, unconditional, no trend-gating and a
different lookback) and from RRR/beta-demean/predictor-shrink/Huber (none of which touch a rank-based
short/long crossover). Tested as an ADDITIONAL blend component layered onto the existing v9 forecast
(wz_new = (1-W)*wz_v9 + W*signal, W swept), not a replacement -- since the description frames it as a
supplementary signal ("this new strategy currently has: lead-lag ridge + short-term reversion [+ this]"),
not a wholesale substitute.

Tested against SAFE_llboost_v9 (current best) -- ALGO leg and the pairwise boost are reused verbatim,
unchanged.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v9 as V9

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V9.WARMUP, V9.BOOST_MIN_DAY, V9.BOOST_K
RIDGE_A = V9.RIDGE_A
HALF_LIVES = V9.HALF_LIVES


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

print("=== precompute: reversal leg, boost, ALGO leg, ridge WZ (unchanged -- reused verbatim from v9) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V9.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V9._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V9._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_V9 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V9._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V9._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    wz_v9 = (1 - V9.BLEND) * wz + V9.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz_v9 = wz_v9 + BOOST_K * BOOST[:, t]
    WZ_V9[:, t] = wz_v9
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def rank_stability_signal(short_w, long_w):
    """signal[j, t] = -short_z[j] where sign(long_z[j]) != sign(short_z[j]), else 0. Both z-scores
    cross-sectional (across the 50 idio names) at day t, using log-price moves through day t --
    causal, no look-ahead."""
    sig = np.full((nIdio, nt), np.nan)
    for t in days:
        if t < max(short_w, long_w) + 5:
            continue
        short_ret = logp[1:, t] - logp[1:, t - short_w]
        long_ret = logp[1:, t] - logp[1:, t - long_w]
        sz = short_ret - short_ret.mean(); sz = sz / (sz.std() + 1e-12)
        lz = long_ret - long_ret.mean(); lz = lz / (lz.std() + 1e-12)
        disagree = np.sign(lz) != np.sign(sz)
        sig[:, t] = np.where(disagree, -sz, 0.0)
    return sig


print("=== precompute: rank_stability signal (short8/long15 reconstruction) ===")
t0 = time.time()
RS_SIGNAL = rank_stability_signal(8, 15)
print(f"  done ({time.time()-t0:.0f}s)")
active = np.isfinite(RS_SIGNAL) & (RS_SIGNAL != 0)
print(f"  fires on {int(active[:, 500:].sum())} stock-days from day 500+ "
      f"({100*active[:,500:].mean():.1f}% of stock-days)")


def build_pos(sig_arr, weight):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V9[:, t].copy()
        if weight > 0 and np.isfinite(sig_arr[:, t]).all():
            s = sig_arr[:, t]
            if s.std() > 1e-12:
                s_z = (s - s.mean()) / (s.std() + 1e-12)
            else:
                s_z = np.zeros(nIdio)
            wz = (1 - weight) * wz + weight * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: weight=0 must reproduce SAFE_llboost_v9 exactly ===")
POS_base = build_pos(RS_SIGNAL, 0.0)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v9 docstring: 848.8/893.3/894.1/708.6)")
if not (abs(base_wo - 848.8) < 0.5 and abs(base_wn - 893.3) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v9 -- do not trust results below. ***")
else:
    print("  OK -- matches v9 to within rounding.")


def evaluate(nm, sig_arr, weight, verbose=True):
    Pz = build_pos(sig_arr, weight); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, weight=weight, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse,
                passed=passed, scs=scs)


print("\n=== SWEEP: blend weight for rank_stability_short8_long15 ===")
WEIGHTS = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
results = [evaluate(f"weight={w}", RS_SIGNAL, w) for w in WEIGHTS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} weights beat v9 on OLD+NEW+rmean jointly.")
if not passing:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  weight={c['weight']:<5} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} "
              f"rmean={c['rm']:>7.1f} rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

print("\n=== robustness: neighboring (short_w, long_w) pairs around the 8/15 reconstruction ===")
for sw, lw in ((5, 10), (8, 15), (8, 20), (10, 20), (10, 25)):
    sig2 = rank_stability_signal(sw, lw)
    best_w = min(passing, key=lambda c: -c["rm"])["weight"] if passing else 0.1
    evaluate(f"  short{sw}_long{lw} @ w={best_w}", sig2, best_w)
