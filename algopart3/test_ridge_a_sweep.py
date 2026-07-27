"""Given the pairwise lead-lag network is real (validated: permutation p=0%, H1/H2 persistence
+0.31 across ALL pairs, quarter-stable, near-universal across 49/50 names) and the CURRENT ridge
already picks the right leader for each name, is the shrinkage level (RIDGE_A=0.1) actually well
tuned, or could a different value extract more of this now-validated signal? Sweep RIDGE_A, idio
book only (ALGO leg off, exact eval-mirroring accounting), everything else identical to SAFE.py.
"""
import numpy as np, pandas as pd
import SAFE

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


logp = np.log(P)
r = logp[:, 1:] - logp[:, :-1]


def idio_wz(t, ridge_a):
    rr = r[:, :t]
    fs = []
    for hl in SAFE.HALF_LIVES:
        B, mx, my = SAFE._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, ridge_a)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if SAFE.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - SAFE.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - SAFE.BLEND) * wz + SAFE.BLEND * rv
    return wz


def build_idio_pos(ridge_a):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        wz = idio_wz(k, ridge_a)
        pos = np.zeros(nInst)
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
        POS[:, k] = np.clip(pos, -lim, lim).astype(int)
    return POS


OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

print(f"{'RIDGE_A':>9}{'OLD':>9}{'NEW':>9}{'roll_mean':>11}{'roll_floor':>12}")
for ridge_a in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
    POS = build_idio_pos(ridge_a)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    mark = "  <-- shipped" if abs(ridge_a - 0.1) < 1e-9 else ""
    print(f"{ridge_a:>9}{wo['score']:>9.1f}{wn['score']:>9.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}{mark}")
