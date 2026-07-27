"""Robustness check for the min_cp/K=1.5 significance-gated pairwise boost found in
test_boost_floor_mitigation.py: does it survive changing the checkpoint RETRAIN CADENCE (the
original plan's section-4 robustness bar), not just the min-history threshold at one fixed
cadence? Reruns the full mechanism at checkpoint steps 25/50/75/100, each with its own min_day=500
gate (day-based, not checkpoint-index-based, so it's meaningful across cadences) and K=1.5.
"""
import numpy as np, pandas as pd
from scipy import stats
import SAFE, SAFE_llvol

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
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


IC_L = 220
ALPHA = 0.05
N_CANDIDATES = 49


def sig_threshold(n_samples, alpha=ALPHA, n_tests=N_CANDIDATES):
    if n_samples < 10: return 1.0
    alpha_adj = alpha / n_tests
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def boost_arr(i, upto, p):
    scale = np.nanstd(r[i, max(0, upto - 500):upto]) + 1e-12
    lret = r[i]
    return np.sign(lret) * (np.abs(lret) / scale) ** p


print("computing shipped SAFE idio wz series (shared across cadences) ...")
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

print("computing shipped ALGO leg (shared) ...")
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("done")

end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def build_strong_at(cp_step):
    checkpoints = list(range(200, nt, cp_step))
    strong_at = {}
    for cp in checkpoints:
        Xi = r[1:, :cp - 1]; Yj = r[1:, 1:cp]
        n_samples = Xi.shape[1]
        thr = sig_threshold(n_samples)
        n = nInst - 1
        C = corrmat(Xi, Yj)
        best_leader = {}; best_corr = {}
        for j in range(n):
            col = C[:, j].copy(); col[j] = np.nan
            i = int(np.nanargmax(np.abs(col))); best_leader[j + 1] = i + 1; best_corr[j + 1] = col[i]
        entry = {}
        for j, i in best_leader.items():
            if abs(best_corr[j]) <= thr:
                continue
            b = boost_arr(i, cp, 2.0)
            a = max(0, cp - IC_L)
            xs = b[a:cp - 1]; ys = r[j, a + 1:cp]
            ok = ~np.isnan(xs) & ~np.isnan(ys)
            if ok.sum() < 60 or xs[ok].std() < 1e-12:
                continue
            ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
            entry[j] = (i, ic > 0)
        strong_at[cp] = entry
    return checkpoints, strong_at


def scs_curve(POS):
    return np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])


def report(nm, POS, base_scs=None):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = scs_curve(POS)
    line = f"{nm:<28}OLD={wo['score']:>7.1f}  NEW={wn['score']:>7.1f}  rmean={scs.mean():>7.1f}  rfloor={scs.min():>7.1f}"
    if base_scs is not None:
        nworse = int((scs < base_scs).sum())
        line += f"  n_worse={nworse}/{len(scs)}"
    print(line)
    return scs


for cp_step in (25, 50, 75, 100):
    print(f"\n=== checkpoint cadence = every {cp_step} days ===")
    checkpoints, strong_at = build_strong_at(cp_step)
    boost_cache = {}
    for cp, entry in strong_at.items():
        for j, (i, gate) in entry.items():
            if (i, cp) not in boost_cache:
                boost_cache[(i, cp)] = boost_arr(i, cp, 2.0)

    def strong_for_day(k, checkpoints=checkpoints, strong_at=strong_at):
        valid = [c for c in checkpoints if c <= k]
        if not valid: return None, {}
        cp = max(valid)
        return cp, strong_at[cp]

    def build_pos(K, min_day, checkpoints=checkpoints, strong_at=strong_at, boost_cache=boost_cache,
                  strong_for_day=strong_for_day):
        POS = np.zeros((nInst, nt))
        for k in range(SAFE.WARMUP, nt):
            cur = P[:, k]; lim = (dlr / cur).astype(int)
            cp, entry = strong_for_day(k)
            for j in range(1, nInst):
                wz = WZ[k][j - 1]
                boost = 0.0
                if K > 0 and cp is not None and k >= min_day and j in entry:
                    i, gate = entry[j]
                    if gate:
                        boost = K * boost_cache[(i, cp)][k - 1]
                sig = wz + boost
                POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
        POS[0, :] = algo_pos
        return POS

    base_scs = report("baseline", build_pos(0.0, 0))
    report("min_day=500 K=1.5", build_pos(1.5, 500), base_scs)
