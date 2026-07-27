"""User's hypothesis: the pairwise convex-boost catastrophe wasn't from including weak pairs, it's
from applying a FIXED boost at ALL times -- exactly the mistake the ALGO leg already avoided (fixed-
direction reversion/momentum bets failed there; the adaptive trailing-IC switch is what worked).
Test the same fix here: gate each pair's boost by ITS OWN trailing IC (only apply when currently
paying, per name, per day), instead of a constant multiplier always on.
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


n = nInst - 1
Xi = r[1:, :-1]; Yj = r[1:, 1:]
def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]
C = corrmat(Xi, Yj)
best_leader = {}; best_corr = {}
for j in range(n):
    col = C[:, j].copy(); col[j] = np.nan
    i = int(np.nanargmax(np.abs(col))); best_leader[j + 1] = i + 1; best_corr[j + 1] = col[i]
STRONG = {j: i for j, i in best_leader.items() if abs(best_corr[j]) > 0.10}
print(f"{len(STRONG)}/49 names in the STRONG subset")

print("computing shipped SAFE idio wz series ...")
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

# precompute each STRONG pair's boost-signal series and target-return series, for trailing-IC gating
ret1_target = {}         # j -> ret1[t] = r[j, t]  (the return we're trying to predict, day t)
boost_series = {}        # j -> boost[t] using leader's return realized at day t-1 (causal, aligns with predicting r[j,t])
for j, i in STRONG.items():
    scale = leader_scale[i]
    b = np.full(nt, np.nan)
    for t in range(1, nt - 1):
        lret = r[i, t - 1]
        b[t] = np.sign(lret) * (abs(lret) / scale) ** 3.0
    boost_series[j] = b
    tgt = np.full(nt, np.nan); tgt[:nt - 1] = r[j, :nt - 1]
    ret1_target[j] = tgt


def trailing_ic(feat, ret1, tnow, L):
    a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
    ok = ~np.isnan(xs) & ~np.isnan(ys)
    if ok.sum() < 60: return None
    xs, ys = xs[ok], ys[ok]
    if xs.std() < 1e-12: return None
    return float(np.corrcoef(xs, ys)[0, 1])


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos(K, IC_L, gate=True):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            if j in STRONG and K > 0:
                b = boost_series[j][k]
                if gate:
                    ic = trailing_ic(boost_series[j], ret1_target[j], k, IC_L)
                    gsign = 1.0 if (ic is not None and ic > 0) else 0.0
                else:
                    gsign = 1.0
                boost = gsign * b if not np.isnan(b) else 0.0
            else:
                boost = 0.0
            sig = wz + K * boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


print(f"\n{'K':>6}{'IC_L':>6}{'gate':>6}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}{'worst window':>16}")
POS0 = build_pos(0, 120)
wo = window(POS0, *OLD); wn = window(POS0, *NEW)
scs = [(E, window(POS0, E - NUMTEST, E)["score"]) for E in end_days]
print(f"{'0 (baseline)':>18}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean([s for _,s in scs]):>8.1f}{min(s for _,s in scs):>9.1f}")

for K in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
    for IC_L in (90, 120, 180):
        POS = build_pos(K, IC_L, gate=True)
        wo = window(POS, *OLD); wn = window(POS, *NEW)
        scs = [(E, window(POS, E - NUMTEST, E)["score"]) for E in end_days]
        worst = min(scs, key=lambda x: x[1])
        print(f"{K:>6}{IC_L:>6}{'Y':>6}{wo['score']:>8.1f}{wn['score']:>8.1f}"
              f"{np.mean([s for _,s in scs]):>8.1f}{worst[1]:>9.1f}   days {worst[0]-250}-{worst[0]}")
