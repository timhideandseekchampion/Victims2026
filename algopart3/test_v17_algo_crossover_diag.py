"""
test_v17_algo_crossover_diag.py

Follow-up to test_v17cand_algo_crossover.py's rejection. That test found a suspicious pattern: at
small windows, NEW consistently improves as blend weight rises while OLD consistently degrades --
opposite-direction movement, unlike v9/v10's own discoveries where both windows moved together as
their effect strengthened. Before concluding this is definitely an artifact (rather than just
eyeballing the score pattern), this runs the same rigor already applied to v9 and v10:

  1. RAW, MODEL-FREE IC of the crossover vote against ALGO's own next-day return, by era -- is the
     underlying "fade a countermove against the medium-term trend" relationship itself directionally
     stable over time, or does its sign of predictive power flip between OLD and NEW (which would
     directly explain, not just correlate with, the observed score divergence)? Mirrors
     test_algo_ic_regime_drivers.py's quartile-breakdown methodology from earlier this session.
  2. WALK-FORWARD check on the blended book: select (short_w, long_w, weight) using ONLY the OLD
     window, check the untouched NEW window -- and the reverse -- matching the exact diligence
     already applied to v9's beta-demean and v10's rank-stability.
  3. A SMARTER construction: gate the crossover vote by its OWN trailing realized IC (only trust it
     when it's been working recently), mirroring the double-IC philosophy already used elsewhere in
     ALGO's own `_side()` -- does a regime-adaptive version fix what a fixed-weight blend can't?
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
lpA = logp[0]
T = len(lpA)


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


def crossover_vote(short_w, long_w):
    """length-T vote array (causal): sign of the medium-term trend on days the short-term move
    opposes it, 0 (no vote) otherwise / on days without enough history."""
    vote = np.zeros(T)
    for k in range(max(short_w, long_w) + 5, T):
        long_ret = lpA[k] - lpA[k - long_w]
        short_ret = lpA[k] - lpA[k - short_w]
        if long_ret == 0 or short_ret == 0:
            continue
        if np.sign(long_ret) != np.sign(short_ret):
            vote[k] = np.sign(long_ret)
    return vote


ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]   # ret1[k] = return realized AFTER day k

# ==================================================================================================
# 1) raw, model-free IC of the vote against ALGO's own next-day return, by era
# ==================================================================================================
print("=" * 96)
print("1) RAW IC: crossover vote vs ALGO's next-day return, by era (model-free, no blend/weight)")
print("=" * 96)
QUARTS = [(0, 250), (250, 500), (500, 750), (750, 1000)]
for sw, lw in ((5, 10), (5, 15), (8, 22), (10, 30)):
    vote = crossover_vote(sw, lw)
    active = vote != 0
    print(f"\n  short{sw}_long{lw}:")
    for lo, hi in QUARTS:
        m = active[lo:hi] & np.isfinite(ret1[lo:hi])
        n = int(m.sum())
        if n < 20:
            print(f"    days {lo:4d}-{hi:4d}: n={n:3d} (too few to estimate)")
            continue
        hit = float((np.sign(ret1[lo:hi][m]) == vote[lo:hi][m]).mean())
        ic = float(np.corrcoef(vote[lo:hi][m], ret1[lo:hi][m])[0, 1])
        print(f"    days {lo:4d}-{hi:4d}: n={n:3d}  hit_rate={100*hit:5.1f}%  IC={ic:+.4f}")
    # OLD/NEW specifically (the two windows the score test actually used)
    for nm, (lo, hi) in (("OLD (500-750)", OLD), ("NEW (750-1000)", NEW)):
        m = active[lo:hi] & np.isfinite(ret1[lo:hi])
        n = int(m.sum())
        if n < 20:
            continue
        hit = float((np.sign(ret1[lo:hi][m]) == vote[lo:hi][m]).mean())
        ic = float(np.corrcoef(vote[lo:hi][m], ret1[lo:hi][m])[0, 1])
        print(f"    {nm:<16}: n={n:3d}  hit_rate={100*hit:5.1f}%  IC={ic:+.4f}")

print("\n" + "=" * 96)
print("full-file zero-crossing check: does the vote's OWN trailing IC change sign over time, like")
print("ALGO's vol-timing edge did earlier this session (days ~100-500 transition)?")
print("=" * 96)
for sw, lw in ((5, 10), (8, 22)):
    vote = crossover_vote(sw, lw)
    trailing_ic = np.full(T, np.nan)
    W = 250
    for k in range(W + 50, T - 1):
        seg_v = vote[k - W:k]; seg_r = ret1[k - W:k]
        m = (seg_v != 0) & np.isfinite(seg_r)
        if m.sum() < 30:
            continue
        trailing_ic[k] = float(np.corrcoef(seg_v[m], seg_r[m])[0, 1])
    valid = np.isfinite(trailing_ic)
    signs = np.sign(trailing_ic[valid])
    crossings = int((np.diff(signs) != 0).sum())
    print(f"  short{sw}_long{lw}: {crossings} sign changes in the trailing-{W}d IC over the file "
          f"(days {np.where(valid)[0][0]}-{np.where(valid)[0][-1]})")
