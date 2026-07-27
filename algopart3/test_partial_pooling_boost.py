"""Partial-pooling (empirical Bayes shrinkage) version: each pair keeps its OWN independent gate
(preserving the idio book's diversification -- different pairs can be on/off on different days,
unlike the shared network gate that just failed), but that gate is a shrinkage blend of the pair's
own trailing IC and the population-wide average IC, weighted by how much of ITS OWN history that
specific pair has accumulated (n / (n + tau)). Early on with thin per-pair history, a pair leans on
the population prior (avoiding the "50 noise-driven pairs at day 200" problem); as its own track
record grows, it relies more on itself. Still fully causal throughout.
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


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


CHECKPOINTS = list(range(150, nt, 50))
P_POWER = 2.0


def boost_arr(i, upto, p):
    scale = np.nanstd(r[i, max(0, upto - 500):upto]) + 1e-12
    lret = r[i]
    return np.sign(lret) * (np.abs(lret) / scale) ** p


print("building candidate leader map at each checkpoint (causal, expanding window) ...")
LEADER_AT = {}
for cp in CHECKPOINTS:
    Xi = r[1:, :cp - 1]; Yj = r[1:, 1:cp]
    n = nInst - 1
    C = corrmat(Xi, Yj)
    leader = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col))); leader[j + 1] = i + 1
    LEADER_AT[cp] = leader


def leader_for_day(k):
    valid = [c for c in CHECKPOINTS if c <= k]
    if not valid: return None, {}
    return max(valid), LEADER_AT[max(valid)]


BOOST_CACHE = {}
for cp, leader in LEADER_AT.items():
    for j, i in leader.items():
        if (i, cp) not in BOOST_CACHE:
            BOOST_CACHE[(i, cp)] = boost_arr(i, cp, P_POWER)

print("building the daily pooled network edge series (population prior) ...")
pooled_edge = np.full(nt, np.nan)
for k in range(150, nt - 1):
    cp, leader = leader_for_day(k)
    if cp is None: continue
    vals = []
    for j, i in leader.items():
        b = BOOST_CACHE[(i, cp)][k - 1]
        if not np.isnan(b):
            vals.append(b * r[j, k])
    if vals:
        pooled_edge[k] = float(np.mean(vals))


def trailing_mean(feat, tnow, L):
    a = max(0, tnow - L); xs = feat[a:tnow]
    ok = ~np.isnan(xs)
    return (float(xs[ok].mean()), int(ok.sum())) if ok.sum() > 0 else (None, 0)


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

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt); FULL = (SAFE.WARMUP, nt)

# pair-specific edge history cache: edge_j(t) = boost_ij computed from the leader's return AS OF
# t-1 (the same value that would actually be traded on day t) times r[j,t] (the return that trade
# earns) -- FIXED: must use b[t-1], not b[t], or this measures same-day co-movement instead of the
# 1-day-lagged predictive relationship (same bug class as everything else needing causal alignment).
PAIR_EDGE_CACHE = {}
def pair_edge_series(j, i, cp):
    key = (j, i, cp)
    if key in PAIR_EDGE_CACHE: return PAIR_EDGE_CACHE[key]
    b = BOOST_CACHE[(i, cp)]
    e = np.full(nt, np.nan)
    for t in range(1, nt - 1):
        bv = b[t - 1]
        if not np.isnan(bv):
            e[t] = bv * r[j, t]
    PAIR_EDGE_CACHE[key] = e
    return e


def build_idio_pos(K, IND_L, TAU):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        cp, leader = leader_for_day(k)
        pop_mean, _ = trailing_mean(pooled_edge, k, 90) if cp is not None else (None, 0)
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            boost = 0.0
            if K > 0 and cp is not None and j in leader:
                i = leader[j]
                b = BOOST_CACHE[(i, cp)][k - 1]
                if not np.isnan(b):
                    e = pair_edge_series(j, i, cp)
                    ind_mean, n_ind = trailing_mean(e, k, IND_L)
                    if ind_mean is None: ind_mean = 0.0
                    w = n_ind / (n_ind + TAU)
                    pm = pop_mean if pop_mean is not None else 0.0
                    shrunk = w * ind_mean + (1 - w) * pm
                    if shrunk > 0:
                        boost = b
            sig = wz + K * boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


print("computing shipped ALGO leg (unchanged) ...")
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("done")

POS0 = build_idio_pos(0, 120, 90); POS0[0, :] = algo_pos
scs0 = np.array([window(POS0, E - NUMTEST, E)["score"] for E in end_days])
wo0 = window(POS0, *OLD); wn0 = window(POS0, *NEW); wf0 = window(POS0, *FULL)
print(f"\n{'K':>6}{'IND_L':>7}{'TAU':>6}{'FULL1000':>10}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse/61':>12}")
print(f"{'baseline':>19}{wf0['score']:>10.1f}{wo0['score']:>9.1f}{wn0['score']:>9.1f}{scs0.mean():>9.1f}{scs0.min():>9.1f}")
for K in (0.5, 1.0, 1.5, 2.0):
    for IND_L, TAU in [(90, 60), (120, 90), (150, 120), (200, 150)]:
        POS = build_idio_pos(K, IND_L, TAU); POS[0, :] = algo_pos
        scs = np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])
        nworse = int((scs < scs0).sum())
        wo = window(POS, *OLD); wn = window(POS, *NEW); wf = window(POS, *FULL)
        print(f"{K:>6}{IND_L:>7}{TAU:>6}{wf['score']:>10.1f}{wo['score']:>9.1f}{wn['score']:>9.1f}{scs.mean():>9.1f}{scs.min():>9.1f}{nworse:>12}")
