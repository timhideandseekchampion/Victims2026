"""Does the DUCT->AMRP convex/threshold effect (correlation stronger for big DUCT moves, validated:
permutation p=0%, H1/H2 persistent) actually move the SCORE, not just the correlation? Build AMRP's
forecast two ways: (a) SAFE.py's shipped ridge-ensemble forecast for AMRP alone, (b) the same plus a
convex boost term on DUCT's return (sign(DUCT_ret) * |DUCT_ret|^p, amplifying big moves more than
proportionally). Score AMRP's OWN isolated PnL (not diluted by the other 48 names), same
eval-mirroring accounting used throughout tonight. Sweep the boost gain K and convexity power p.
"""
import numpy as np, pandas as pd
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)
r = logp[:, 1:] - logp[:, :-1]

AMRP_I = names.index("AMRP")
DUCT_I = names.index("DUCT")


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window_1name(pos_row, S, E, idx):
    curPos = 0.0; comm = 0.0; prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[idx, tt - 1]
        newPos = (pos_row[tt - 1] if tt < E else curPos)
        if tt > S:
            tot.append(float(curPos * (cur - prevCur) - comm))
        dP = newPos - curPos
        comm = commRate[idx] * abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


def window_full(POS, S, E):
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

# scale reference: typical |DUCT daily return| so the boost term's magnitude is comparable to wz's own scale
duct_scale = np.nanstd(r[DUCT_I, :500])


def convex_boost(k_day, p, scale):
    duct_ret = r[DUCT_I, k_day - 1]     # DUCT's most recent realized return (causal)
    return np.sign(duct_ret) * (abs(duct_ret) / scale) ** p


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))

print(f"\n{'p':>5}{'K':>7}{'AMRP OLD':>10}{'AMRP NEW':>10}{'AMRP rmean':>12}{'AMRP rfloor':>12}")
best_pos_row = None
for p in (1.0, 1.5, 2.0, 3.0):
    for K in (0.0, 0.3, 0.6, 1.0, 1.5, 2.0):
        if p != 1.0 and K == 0.0:
            continue   # baseline (K=0) is p-independent, only print once
        pos_row = np.zeros(nt)
        for k in range(SAFE.WARMUP, nt):
            wz = WZ[k][AMRP_I - 1]
            boost = convex_boost(k, p, duct_scale) if K > 0 else 0.0
            sig = wz + K * boost
            cur = P[AMRP_I, k]
            lim = int(dlr[AMRP_I] / cur)
            pos_row[k] = np.clip(np.sign(sig) * (dlr[AMRP_I] / cur), -lim, lim)
        wo = window_1name(pos_row, *OLD, AMRP_I); wn = window_1name(pos_row, *NEW, AMRP_I)
        scs = [window_1name(pos_row, E - NUMTEST, E, AMRP_I)["score"] for E in end_days]
        mark = "  <-- shipped (K=0)" if K == 0.0 else ""
        print(f"{p:>5}{K:>7}{wo['score']:>10.1f}{wn['score']:>10.1f}{np.mean(scs):>12.1f}{min(scs):>12.1f}{mark}")
