"""Fix for the causal version's failure: the |corr|>0.10 threshold was itself picked by eyeballing
the FULL-SAMPLE correlation distribution (a subtler look-ahead in threshold-choice), and it doesn't
account for sample size -- at day 200 with only ~198 return-pairs, a raw 0.10 correlation is nowhere
near as reliable as the same 0.10 measured over 950 days. Fix: use a proper t-test for correlation
significance, Bonferroni-corrected for testing 49 candidate leaders per follower (49 comparisons),
re-evaluated at each causal checkpoint using ONLY the data available then. This should naturally be
much more conservative early (excluding most/all pairs when data is too thin to support them) and
only include pairs once genuinely statistically justified by the sample size actually available.
"""
import numpy as np, pandas as pd
from scipy import stats
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


CHECKPOINTS = list(range(200, nt, 50))
IC_L = 220
ALPHA = 0.05
N_CANDIDATES = 49


def sig_threshold(n_samples, alpha=ALPHA, n_tests=N_CANDIDATES):
    """min |corr| to be significant at alpha, Bonferroni-corrected for n_tests, given n_samples."""
    if n_samples < 10: return 1.0
    alpha_adj = alpha / n_tests
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def boost_arr(i, upto, p):
    scale = np.nanstd(r[i, max(0, upto - 500):upto]) + 1e-12
    lret = r[i]
    return np.sign(lret) * (np.abs(lret) / scale) ** p


print("re-estimating leader map with a SAMPLE-SIZE-ADJUSTED significance threshold, at each checkpoint ...")
STRONG_AT = {}
for cp in CHECKPOINTS:
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
    STRONG_AT[cp] = entry
    print(f"  day {cp}: n={n_samples} sig_thr={thr:.3f}  -> {len(entry)} pairs pass both filters")

BOOST_CACHE = {}
for cp, entry in STRONG_AT.items():
    for j, (i, gate) in entry.items():
        if (i, cp) not in BOOST_CACHE:
            BOOST_CACHE[(i, cp)] = boost_arr(i, cp, 2.0)


def strong_for_day(k):
    cp = max([c for c in CHECKPOINTS if c <= k], default=CHECKPOINTS[0])
    return cp, STRONG_AT[cp]


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
OLD = (500, 750); NEW = (750, nt)


def build_idio_pos(K):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        cp, entry = strong_for_day(k)
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            boost = 0.0
            if K > 0 and j in entry:
                i, gate = entry[j]
                if gate:
                    boost = BOOST_CACHE[(i, cp)][k - 1]
            sig = wz + K * boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


print("computing shipped ALGO leg (unchanged) ...")
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print("done")

POS0 = build_idio_pos(0); POS0[0, :] = algo_pos
scs0 = np.array([window(POS0, E - NUMTEST, E)["score"] for E in end_days])
wo0 = window(POS0, *OLD); wn0 = window(POS0, *NEW)
print(f"\n{'K':>6}{'FULL OLD':>10}{'FULL NEW':>10}{'FULL mean':>11}{'FULL floor':>12}{'n_worse/61':>12}")
print(f"{'baseline':>16}{wo0['score']:>10.1f}{wn0['score']:>10.1f}{scs0.mean():>11.1f}{scs0.min():>12.1f}")
for K in (0.5, 1.0, 1.5, 2.0, 3.0):
    POS = build_idio_pos(K); POS[0, :] = algo_pos
    scs = np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])
    nworse = int((scs < scs0).sum())
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    print(f"{K:>6}{wo['score']:>10.1f}{wn['score']:>10.1f}{scs.mean():>11.1f}{scs.min():>12.1f}{nworse:>12}")

# --- also report the FULL 1000-day continuous window, and fix the same pre-checkpoint look-ahead ---
FULL = (SAFE.WARMUP, nt)
print("\n(full-1000-day window, using the FIXED strong_for_day that returns nothing before day 200)")
def strong_for_day_fixed(k):
    valid = [c for c in CHECKPOINTS if c <= k]
    if not valid: return None, {}
    cp = max(valid)
    return cp, STRONG_AT[cp]

def build_idio_pos_fixed(K):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        cp, entry = strong_for_day_fixed(k)
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            boost = 0.0
            if K > 0 and cp is not None and j in entry:
                i, gate = entry[j]
                if gate:
                    boost = BOOST_CACHE[(i, cp)][k - 1]
            sig = wz + K * boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS

POS0f = build_idio_pos_fixed(0); POS0f[0, :] = algo_pos
wf0 = window(POS0f, *FULL)
print(f"{'K':>6}{'FULL(1000d)':>12}")
print(f"{'baseline':>16}{wf0['score']:>12.1f}")
for K in (0.5, 1.0, 1.5, 2.0, 3.0):
    POSf = build_idio_pos_fixed(K); POSf[0, :] = algo_pos
    wf = window(POSf, *FULL)
    print(f"{K:>6}{wf['score']:>12.1f}")
