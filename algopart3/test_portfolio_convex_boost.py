"""Does the validated DUCT->AMRP convex-boost trick generalize: apply the SAME mechanism (each
stock gets a boost term from its OWN best-leader stock, using a FIXED p,K chosen from the AMRP
tuning -- not re-tuned per pair, to avoid re-introducing the exact overfitting risk flagged earlier
tonight) to every one of the 49 idio names, and check the FULL PORTFOLIO score (idio book + shipped
ALGO leg), not just one isolated name.
"""
import numpy as np, pandas as pd
import SAFE, SAFE_llvol

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)
r = logp[:, 1:] - logp[:, :-1]


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


print("finding each stock's best leader (full-sample |corr|, causal-safe since only used to pick WHICH stock, gain fixed not tuned) ...")
n = nInst - 1
Xi = r[1:, :-1]; Yj = r[1:, 1:]
def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]
C = corrmat(Xi, Yj)
best_leader = {}   # follower_idx (1..50) -> leader_idx (1..50)
for j in range(n):
    col = C[:, j].copy(); col[j] = np.nan
    i = int(np.nanargmax(np.abs(col)))
    best_leader[j + 1] = i + 1

print("computing shipped SAFE idio wz series (all 49 names) ...")
WZ = {}
for t in range(SAFE.WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, SAFE.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    WZ[t] = wz
print("done")

leader_scale = {i: np.nanstd(r[i, :500]) for i in range(1, nInst)}

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos(p, K, algo_pos):
    POS = np.zeros((nInst, nt))
    POS[0, :] = algo_pos
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            leader_i = best_leader[j]
            lret = r[leader_i, k - 1]
            scale = leader_scale[leader_i]
            boost = np.sign(lret) * (abs(lret) / scale) ** p if K > 0 else 0.0
            sig = wz + K * boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


print("computing shipped ALGO leg (SAFE_llvol, unchanged) ...")
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("done")

print(f"\n{'p':>5}{'K':>6}{'idio OLD':>10}{'idio NEW':>10}{'idio rmean':>12}{'idio rfloor':>13}"
      f"{'FULL OLD':>10}{'FULL NEW':>10}{'FULL rmean':>12}{'FULL rfloor':>13}")
for p, K in [(0, 0), (3.0, 1.0), (3.0, 2.0), (4.0, 2.0), (4.0, 3.0), (5.0, 3.0), (5.0, 5.0)]:
    POS = build_pos(p, K, algo_pos)
    POS_idio_only = POS.copy(); POS_idio_only[0, :] = 0
    wo_i = window(POS_idio_only, *OLD); wn_i = window(POS_idio_only, *NEW)
    scs_i = [window(POS_idio_only, E - NUMTEST, E)["score"] for E in end_days]
    wo_f = window(POS, *OLD); wn_f = window(POS, *NEW)
    scs_f = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    mark = "  <-- shipped (K=0)" if K == 0 else ""
    print(f"{p:>5}{K:>6}{wo_i['score']:>10.1f}{wn_i['score']:>10.1f}{np.mean(scs_i):>12.1f}{min(scs_i):>13.1f}"
          f"{wo_f['score']:>10.1f}{wn_f['score']:>10.1f}{np.mean(scs_f):>12.1f}{min(scs_f):>13.1f}{mark}")
