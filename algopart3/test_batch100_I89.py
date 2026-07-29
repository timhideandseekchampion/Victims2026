"""
test_batch100_I89.py

I89 (DIAGNOSTIC): block-bootstrap stress test of SAFE_llboost_v10, resampling contiguous BLOCKS of
the REAL multivariate price path (not a parametric generator), complementing the existing parametric
stress tests (stress_test_synthetic.py, test_v10_stress_synthetic.py / _v2.py) which fit a one-factor
market model + calibrated idio/ALGO processes. This is the standard non-parametric block bootstrap for
dependent time series: resample WHOLE DAYS (blocks of contiguous days) with replacement, preserving
the cross-sectional (across-instrument) co-movement within each block exactly as it occurred, while
breaking up the specific historical sequencing/autocorrelation across block boundaries.

Method: block length L=20 trading days. For each draw, build a synthetic nt-day return panel by
concatenating randomly-chosen (with replacement) length-L blocks of the REAL log-return matrix
`r = diff(log(prices))` until reaching the full length, then reconstruct a synthetic price path by
cumulating those returns from the real starting price. Run the ACTUAL SAFE_llboost_v10.getMyPosition
walk-forward (full sequential simulation, matching validate_llboost_v10_full.py's convention) on each
synthetic path, and score the last 250 days (the standard numTestDays convention).
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

np.random.seed(0)

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250

logp = np.log(P_)
r = np.diff(logp, axis=1)  # (nInst, nt-1)
nRet = r.shape[1]

L = 20
N_BOOT = 25


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore_lastN(POS, n_test):
    S, E = nt - n_test, nt
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_synth[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


def make_synthetic_returns():
    starts = np.arange(0, nRet - L + 1)
    blocks = []
    total = 0
    while total < nRet:
        s = np.random.choice(starts)
        blocks.append(r[:, s:s + L])
        total += L
    r_synth = np.concatenate(blocks, axis=1)[:, :nRet]
    return r_synth


def full_walk_score(P_path):
    """Simulate SAFE_llboost_v10.getMyPosition sequentially on P_path (nInst, nt), full walk, then
    score the last NUMTEST days."""
    global P_synth
    P_synth = P_path
    V10._DLR = None
    V10._PREV_ALGO_SHARES = 0
    V10._PREV_T = -1
    POS = np.zeros((nInst, nt))
    lim_all = None
    for t in range(1, nt):
        prcSoFar = P_path[:, :t]
        p = V10.getMyPosition(prcSoFar)
        lim = (dlr / prcSoFar[:, -1]).astype(int)
        POS[:, t - 1] = np.clip(np.asarray(p, dtype=float), -lim, lim)
    return wscore_lastN(POS, NUMTEST)


print("=== sanity check: full walk on the REAL price path must reproduce v10's official-style "
      "score (250-day window) ===", flush=True)
t0 = time.time()
real_score = full_walk_score(P_)
print(f"  real-data full-walk score (last {NUMTEST} days): {real_score:.1f}  "
      f"(v10 NEW=750-1000 docstring score: 912.6 -- similar window, not bit-identical convention)  "
      f"[{time.time()-t0:.0f}s]")

print(f"\n=== block bootstrap: L={L} days, {N_BOOT} draws, real starting prices, full v10 walk-forward "
      f"each draw ===", flush=True)
scores = []
t0 = time.time()
for b in range(N_BOOT):
    tb = time.time()
    r_synth = make_synthetic_returns()
    logp_synth = np.zeros((nInst, nt))
    logp_synth[:, 0] = logp[:, 0]
    logp_synth[:, 1:] = logp[:, 0:1] + np.cumsum(r_synth, axis=1)
    P_synth_path = np.exp(logp_synth)
    sc = full_walk_score(P_synth_path)
    scores.append(sc)
    print(f"  draw {b+1:2d}/{N_BOOT}: score={sc:8.1f}   [{time.time()-tb:.0f}s]", flush=True)

scores = np.array(scores)
print(f"\n  total time {time.time()-t0:.0f}s", flush=True)
print("\n=== block-bootstrap score distribution (real price path, L=20, N=25) ===")
print(f"  mean={scores.mean():.1f}  median={np.median(scores):.1f}  std={scores.std():.1f}")
print(f"  min={scores.min():.1f}  p5={np.percentile(scores,5):.1f}  p25={np.percentile(scores,25):.1f}  "
      f"p75={np.percentile(scores,75):.1f}  p95={np.percentile(scores,95):.1f}  max={scores.max():.1f}")
print(f"  real-data (unshuffled) full-walk score sits at percentile: "
      f"{100*(scores < real_score).mean():.0f}")
