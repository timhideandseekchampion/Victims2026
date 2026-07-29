"""
validate_postjumpfade_full.py

Full getMyPosition-pathway validation of the "post-jump fixed-size fade" candidate found while
running the batch100-style signal search (test_batch100_catHIJK_vol_event_misc.py, item 4), on top
of the REAL, shipped SAFE_llboost_v10 -- not the backtest-approximation harness (_v10_harness.py)
used to discover and sweep it. Mirrors validate_llboost_v10_full.py's convention exactly.

CANDIDATE MECHANISM: after v10's full idio forecast wz is built (ridge+beta-demean, BLEND reversal,
pairwise boost, rank-stability blend -- unchanged, byte-identical to v10), on any name whose most
recent daily return exceeded K_SIGMA * (trailing W-day stdev computed strictly BEFORE that return)
add a fixed-size extra term EXTRA_W * (-sign(that return)) * (mean|wz| that day) -- i.e. a discrete,
event-triggered fade on top of the existing CONTINUOUS reversal leg (BLEND=0.3, REV_W=10), sized
relative to today's own forecast magnitude so it scales sensibly across the book rather than being
a fixed absolute number.

Chosen params: W=40, K_SIGMA=2.0, EXTRA_W=0.06 -- the best-by-rolling-mean point in a 5x4x7=140
config neighbor grid (W in {30,35,40,45,50} x K_SIGMA in {1.75,2.0,2.25,2.5} x EXTRA_W in
{0.02..0.08}) that ALSO hits n_worse=0/61, not just the single best raw number. 56/140 configs in
that grid clear the strict OLD+NEW+rolling-mean bar -- a broad, multi-dimensional plateau, not an
isolated spike (contrast with the rejected Huber IRLS candidate in this file's history, which had
exactly one passing point surrounded by failing neighbors).
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250

W = 40
K_SIGMA = 2.0
EXTRA_W = 0.06


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return float(score(tot.mean(), tot.std()))


def getMyPosition_candidate(prcSoFar):
    """Byte-for-byte identical to V10.getMyPosition up to the rank-stability blend (calls V10's own
    private helpers directly, so module-level state like the ALGO HOLD-deadband cache and _DLR live
    in V10's namespace exactly as they do for real v10 -- no reimplementation, no state drift), plus
    the one new term."""
    prcSoFar = np.asarray(prcSoFar, dtype=float)
    nInst_, t = prcSoFar.shape
    dlr_ = V10._limits(nInst_)
    cur = prcSoFar[:, -1]
    pos = np.zeros(nInst_)
    if t < V10.WARMUP:
        return pos.astype(int)

    logp = np.log(prcSoFar)
    r = logp[:, 1:] - logp[:, :-1]

    Y = V10._beta_adjusted_target(r)
    fs = []
    for hl in V10.HALF_LIVES:
        B, mx, my = V10._ewls_ridge(r[:, :-1].T, Y, hl, V10.RIDGE_A)
        pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if V10.BLEND > 0:
        rr = logp[1:, -1] - logp[1:, -1 - V10.REV_W]
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        wz = (1 - V10.BLEND) * wz + V10.BLEND * rv

    boost = V10._pairwise_boost(r[1:])
    wz = wz + V10.BOOST_K * boost

    rs_sig = V10._rank_stability_signal(logp)
    if rs_sig is not None:
        s_std = rs_sig.std()
        s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros_like(rs_sig)
        wz = (1 - V10.RS_WEIGHT) * wz + V10.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)

    # --- NEW: post-jump fixed-size fade ---
    idio_r = r[1:]
    if idio_r.shape[1] >= W + 1:
        sigma = idio_r[:, -1 - W:-1].std(axis=1)
        jump = idio_r[:, -1]
        flagged = np.abs(jump) > K_SIGMA * (sigma + 1e-12)
        if flagged.any():
            scale = np.abs(wz).mean() + 1e-12
            fade_dir = -np.sign(jump)
            wz = wz.copy()
            wz[flagged] = wz[flagged] + EXTRA_W * fade_dir[flagged] * scale

    pos[1:] = np.sign(wz) * (dlr_[1:] / cur[1:])
    pos[0] = V10._algo_vol_shares(logp[0], cur[0], dlr_[0])
    lim = (dlr_ / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int)


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
FIRST_DAY = 148

print("building shipped SAFE_llboost_v10 positions (baseline, real getMyPosition) ...", flush=True)
t0 = time.time()
POS_v10 = np.zeros((nInst, nt))
for k in range(FIRST_DAY, nt):
    POS_v10[:, k] = V10.getMyPosition(P[:, :k + 1])
print(f"  done in {time.time()-t0:.0f}s", flush=True)

print(f"building candidate positions (v10 + post-jump fade, W={W} K_SIGMA={K_SIGMA} EXTRA_W={EXTRA_W}) ...",
      flush=True)
t0 = time.time()
POS_cand = np.zeros((nInst, nt))
for k in range(FIRST_DAY, nt):
    POS_cand[:, k] = getMyPosition_candidate(P[:, :k + 1])
print(f"  done in {time.time()-t0:.0f}s", flush=True)

wo_v10, wn_v10 = window(POS_v10, *OLD), window(POS_v10, *NEW)
wo_c, wn_c = window(POS_cand, *OLD), window(POS_cand, *NEW)
scs_v10 = np.array([window(POS_v10, E - NUMTEST, E) for E in end_days])
scs_c = np.array([window(POS_cand, E - NUMTEST, E) for E in end_days])

print(f"\nREAL getMyPosition, shipped v10:     OLD={wo_v10:.2f}  NEW={wn_v10:.2f}  "
      f"rmean={scs_v10.mean():.2f}  rfloor={scs_v10.min():.2f}")
print("  (v10 docstring: 871.0 / 912.6 / 909.8 / 709.7)")
print(f"REAL getMyPosition, +postjump fade:  OLD={wo_c:.2f}  NEW={wn_c:.2f}  "
      f"rmean={scs_c.mean():.2f}  rfloor={scs_c.min():.2f}")

nworse = int((scs_c < scs_v10).sum())
passed = (wo_c > wo_v10) and (wn_c > wn_v10) and (scs_c.mean() > scs_v10.mean())
print(f"\nn_worse={nworse}/{len(scs_c)}   passed(OLD+NEW+rmean jointly)={passed}")

d = POS_cand - POS_v10
n_diff_days = int((np.abs(d).sum(0) > 0).sum())
print(f"\npositions differ on {n_diff_days}/{nt - FIRST_DAY} days from FIRST_DAY (sanity: should be >0, "
      f"consistent with the ~5% flag incidence measured in the backtest sweep)")

print("\n=== turnover/commission sanity (NEW window) ===")
def commission(POS, S, E):
    curPos = np.zeros(nInst); tot = 0.0
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = POS[:, tt - 1] if tt < E else curPos
        dP = newPos - curPos
        tot += float((commRate * np.abs(dP) * cur).sum())
        curPos = newPos
    return tot
c_v10 = commission(POS_v10, *NEW)
c_c = commission(POS_cand, *NEW)
print(f"  v10 commission (NEW):      ${c_v10:,.0f}")
print(f"  candidate commission (NEW):${c_c:,.0f}  (delta ${c_c - c_v10:+,.0f})")
