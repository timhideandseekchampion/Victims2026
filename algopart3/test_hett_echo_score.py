"""Does the ULXY 2-day echo actually move the SCORE, regardless of whether its correlation survives
a strict significance test? Build HETT's forecast two ways -- (a) SAFE.py's shipped ridge-ensemble
forecast for HETT alone, (b) the same forecast plus a blend of ULXY's 2-day-lagged return -- and
score HETT's OWN isolated PnL (not the whole 49-name book, so a single-name tweak isn't swamped by
the other 48) using the same eval-mirroring OLD/NEW/rolling-mean/rolling-floor methodology used
throughout. Sweep the blend weight; also check the full 49-name portfolio for completeness.
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

HETT_I = names.index("HETT")
ULXY_I = names.index("ULXY")


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
            pl = curPos * (cur - prevCur) - comm
            tot.append(float(pl))
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

def ulxy_2day_z(k, zwin=60):
    if k - 2 - zwin < 0: return 0.0
    hist = r[ULXY_I, k - 2 - zwin:k - 2]
    val = r[ULXY_I, k - 2]
    sd = hist.std() + 1e-12
    return (val - hist.mean()) / sd

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))

print(f"\n{'blend k':>9}{'HETT OLD':>10}{'HETT NEW':>10}{'HETT rmean':>12}{'HETT rfloor':>12}")
for K in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0):
    pos_row = np.zeros(nt)
    for k in range(SAFE.WARMUP, nt):
        wz = WZ[k][HETT_I - 1]              # HETT's index into the 50-wide idio target vector
        z2 = ulxy_2day_z(k)
        sig = wz + K * z2
        cur = P[HETT_I, k]
        lim = int(dlr[HETT_I] / cur)
        pos_row[k] = np.clip(np.sign(sig) * (dlr[HETT_I] / cur), -lim, lim)
    wo = window_1name(pos_row, *OLD, HETT_I); wn = window_1name(pos_row, *NEW, HETT_I)
    scs = [window_1name(pos_row, E - NUMTEST, E, HETT_I)["score"] for E in end_days]
    mark = "  <-- shipped (K=0)" if K == 0.0 else ""
    print(f"{K:>9}{wo['score']:>10.1f}{wn['score']:>10.1f}{np.mean(scs):>12.1f}{min(scs):>12.1f}{mark}")
