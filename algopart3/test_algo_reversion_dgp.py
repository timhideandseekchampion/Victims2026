"""Does the ALGO leg have a short-horizon MEAN-REVERSION edge (fade its own recent zscore),
the mechanism the sibling algothon26-starter-code/DGP.md reverse-engineered and validated with a
permutation test on a DIFFERENT prices.txt draw — instead of / alongside SAFE_llvol's vol-level
regime-switch bet? Section 1 factor-structure diagnostics (corr to equal-weight avg ~0.98, R^2~0.20,
single dominant PC) matched that sibling draw almost exactly, suggesting the same generator family.

Tests, using the EXACT eval-mirroring accounting from compute_diagnostics.py:
  1. ALGO-leg-ONLY score (idio off) for zrev(w) at a few windows, vs a shuffled-returns permutation
     null (same methodology as analyze.py section 5) -> is index reversion real here too?
  2. FULL portfolio score (SAFE idio + zrev ALGO leg) vs the shipped SAFE_llvol score, on the same
     OLD/NEW/rolling windows already used throughout algopart3.
"""
import numpy as np, pandas as pd
import SAFE, SAFE_llvol

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250


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
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


def zscore_last(px_row, w):
    """(last price - rolling mean)/rolling std over trailing w days incl today."""
    win = px_row[-w:]
    sd = win.std()
    return 0.0 if sd < 1e-9 else (px_row[-1] - win.mean()) / sd


def algo_only_pos(prc_algo_row, w):
    """ALGO-leg-only integer position series (idio always 0), causal."""
    T = len(prc_algo_row)
    pos = np.zeros(T)
    for t in range(w + 2, T):
        z = zscore_last(prc_algo_row[:t + 1], w)
        cap_shares = int(dlr[0] / prc_algo_row[t])
        pos[t] = int(np.clip(np.sign(-z), -1, 1) * cap_shares)
    return pos


OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

print("=== 1. ALGO-leg-ONLY score: zrev(w) vs shuffled-returns null (idio off) ===")
rng = np.random.default_rng(0)
r0 = np.diff(np.log(P[0]))
for w in (3, 5, 7, 10, 15, 20):
    POS = np.zeros((nInst, nt)); POS[0, :] = algo_only_pos(P[0], w)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    null = []
    for _ in range(150):
        rr = r0.copy(); rng.shuffle(rr)
        perm_algo = P[0, 0] * np.exp(np.concatenate([[0.0], np.cumsum(rr)]))
        POSn = np.zeros((nInst, nt)); POSn[0, :] = algo_only_pos(perm_algo, w)
        null.append(window(POSn, *NEW)["score"])
    null = np.array(null)
    p_ge = float((null >= wn["score"]).mean())
    print(f"  w={w:>3}: OLD {wo['score']:>7.1f}  NEW {wn['score']:>7.1f}  roll_mean {np.mean(scs):>7.1f}  "
          f"roll_floor {min(scs):>7.1f}   null(NEW) mean {null.mean():>6.1f} p95 {np.percentile(null,95):>6.1f}  "
          f"P(null>=obs)={100*p_ge:.0f}%")

print("\n=== 2. FULL portfolio: SAFE idio + zrev(w) ALGO leg, vs shipped SAFE_llvol ===")
safe_idio_only = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1]))
    p = full.copy(); p[0] = 0
    safe_idio_only[:, k] = np.clip(p, -lim, lim).astype(int)

llvol_pos = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    llvol_pos[:, k] = np.clip(np.asarray(SAFE_llvol.getMyPosition(P[:, :k + 1])), -lim, lim).astype(int)
wo = window(llvol_pos, *OLD); wn = window(llvol_pos, *NEW)
scs = [window(llvol_pos, E - NUMTEST, E)["score"] for E in end_days]
print(f"  shipped SAFE_llvol : OLD {wo['score']:>7.1f}  NEW {wn['score']:>7.1f}  roll_mean {np.mean(scs):>7.1f}  roll_floor {min(scs):>7.1f}")

for w in (3, 5, 7, 10, 15, 20):
    POS = safe_idio_only.copy()
    algo_row = algo_only_pos(P[0], w)
    for k in range(130, nt):
        cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
        POS[0, k] = int(np.clip(algo_row[k], -lim0, lim0))
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"  idio + zrev({w:>2}) : OLD {wo['score']:>7.1f}  NEW {wn['score']:>7.1f}  roll_mean {np.mean(scs):>7.1f}  roll_floor {min(scs):>7.1f}")
