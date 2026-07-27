"""Test: idio book currently sizes as sign(wz) * full $10k for every name every day (no magnitude
scaling at all - a barely-nonzero wz gets the exact same $10k bet as a high-conviction wz). Does
scaling position size continuously with |wz| (soft conviction weighting) beat the current all-or-
nothing sign rule? ALGO leg is left exactly as SAFE_llvol computes it in every variant.
"""
import numpy as np, pandas as pd
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
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


def wz_at(logp, r, t):
    rr_hist = r[:, :t]
    fs = []
    for hl in M.HALF_LIVES:
        B, mx, my = M._ewls_ridge(rr_hist[:, :-1].T, rr_hist[1:, 1:].T, hl, M.RIDGE_A)
        pred = my + (rr_hist[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if M.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - M.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - M.BLEND) * wz + M.BLEND * rv
    return wz


print("precomputing wz series ...")
logp = np.log(P)
r = logp[:, 1:] - logp[:, :-1]
WZ = {}
for t in range(M.WARMUP, nt):
    WZ[t] = wz_at(logp, r, t)
print(f"done: {len(WZ)} days")


def build_pos(mode, K=1.0):
    POS = np.zeros((nInst, nt))
    for k in range(130, nt):
        cur = P[:, k]
        lim = (dlr / cur).astype(int)
        algo_pos = M._algo_vol_shares(logp[0, :k + 1], cur[0], dlr[0])
        if k in WZ:
            wz = WZ[k]
            if mode == "sign":
                w = np.sign(wz)
            elif mode == "clip":                       # linear ramp to full size by |wz|=K, capped at 1
                w = np.clip(wz / K, -1, 1)
            elif mode == "tanh":
                w = np.tanh(wz / K)
            idio_pos = w * (dlr[1:] / cur[1:])
        else:
            idio_pos = np.zeros(nInst - 1)
        pos = np.concatenate(([algo_pos], idio_pos))
        POS[:, k] = np.clip(pos, -lim, lim).astype(int)
    return POS


OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

print(f"\n{'mode':>8} {'K':>6} {'OLD':>8} {'NEW':>8} {'roll_mean':>10} {'roll_floor':>11}")
for mode, K in [("sign", 0), ("clip", 0.3), ("clip", 0.5), ("clip", 0.75), ("clip", 1.0), ("clip", 1.5), ("clip", 2.0),
                ("tanh", 0.3), ("tanh", 0.5), ("tanh", 1.0)]:
    POS = build_pos(mode, K)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{mode:>8} {K:>6.2f} {wo['score']:>8.1f} {wn['score']:>8.1f} {np.mean(scs):>10.1f} {min(scs):>11.1f}")
