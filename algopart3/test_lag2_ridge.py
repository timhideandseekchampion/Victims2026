"""Does adding LAG-2 (two days ago) returns as extra ridge features improve on the shipped
lag-1-only cross-sectional ridge? SAFE.py's ridge uses r[:, t-1] (yesterday's full 51-vector) to
predict r[:, t] (today's). This tests whether r[:, t-2] carries independent predictive power once
lag-1 is already in the model -- a principled extension of the validated mechanism, not a new
signal class. Tested two ways: (a) partial correlation of lag-2 vs the ridge's OWN residual
(does lag-2 explain what lag-1 residualizes), (b) an actual ridge refit with [lag-1, lag-2]
concatenated as features, scored properly (not just IC) against the lag-1-only baseline.
"""
import numpy as np, pandas as pd
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)
r = np.diff(logp, axis=1)
T = r.shape[1]


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


print("=== 1. partial-correlation check: does lag-2 explain the shipped ridge's residual? ===")
CHECKPOINTS = list(range(200, T, 100))
pooled_x, pooled_y = [], []
for cp in CHECKPOINTS:
    X1 = r[:, :cp - 2].T; Y = r[:, 2:cp].T  # lag-1 features, aligned so row t predicts r[:,t+2]... 
    # simpler: refit shipped-style single-hl ridge on lag-1 (hl=500) up to cp, get residual at cp
    Xtr = r[:, :cp - 1].T; Ytr = r[:, 1:cp].T
    B, mx, my = SAFE._ewls_ridge(Xtr, Ytr, hl=500, a=SAFE.RIDGE_A)
    pred = my + (r[:, cp - 1] - mx) @ B  # predicts r[:, cp]
    resid = r[:, cp] - pred
    lag2 = r[:, cp - 2]  # two days before cp
    pooled_x.append(lag2); pooled_y.append(resid)
X = np.concatenate(pooled_x); Y = np.concatenate(pooled_y)
ic = np.corrcoef(X, Y)[0, 1]
print(f"pooled corr(lag-2 return, lag-1-ridge residual) = {ic:+.4f}  (n={len(X)})")

print("\n=== 2. actual scored test: lag-1-only ridge vs lag-1+lag-2 concatenated-feature ridge ===")


def build_pos_lag1(hls=SAFE.HALF_LIVES, ridge_a=SAFE.RIDGE_A, blend=SAFE.BLEND, rev_w=SAFE.REV_W):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        rr = r[:, :k]
        fs = []
        for hl in hls:
            B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, ridge_a)
            pred = my + (rr[:, -1] - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if blend > 0:
            rv_ = logp[1:, k] - logp[1:, k - rev_w]
            rv_ = rv_ - rv_.mean()
            rv = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - blend) * wz + blend * rv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


def build_pos_lag12(hls=SAFE.HALF_LIVES, ridge_a=SAFE.RIDGE_A, blend=SAFE.BLEND, rev_w=SAFE.REV_W):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP + 1, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        rr = r[:, :k]
        fs = []
        for hl in hls:
            # X = concat(lag-1, lag-2) at each t -> predict r[:,t]
            X1 = rr[:, 1:-1].T; X2 = rr[:, :-2].T  # lag-1, lag-2 for the same target index
            Xc = np.concatenate([X1, X2], axis=1)
            Yc = rr[1:, 2:].T
            B, mx, my = SAFE._ewls_ridge(Xc, Yc, hl, ridge_a)
            xin = np.concatenate([rr[:, -1], rr[:, -2]])
            pred = my + (xin - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if blend > 0:
            rv_ = logp[1:, k] - logp[1:, k - rev_w]
            rv_ = rv_ - rv_.mean()
            rv = -rv_ / (rv_.std() + 1e-12)
            wz = (1 - blend) * wz + blend * rv
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = np.array([window(POS, E - NUMTEST, E) for E in end_days])
    line = f"{nm:<20}OLD={wo:>7.1f}  NEW={wn:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


base_scs = report("lag-1 only (shipped)", build_pos_lag1())
report("lag-1 + lag-2 concat", build_pos_lag12(), base_scs)

print("\n=== 3. is it salvageable? re-sweep RIDGE_A for the concat lag1+lag2 model (curse-of-dimensionality check) ===")
for ridge_a in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    report(f"lag1+lag2 concat, ridge_a={ridge_a}", build_pos_lag12(ridge_a=ridge_a), base_scs)
