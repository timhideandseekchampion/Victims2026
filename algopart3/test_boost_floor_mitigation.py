"""Follow-up to test_significance_adjusted_boost.py: that mechanism (boost the idio book's
pairwise signal only when a sample-size-adjusted, Bonferroni-corrected significance test passes)
genuinely lifted OLD/NEW/rolling-mean at every K tested, but monotonically WORSENED the rolling
floor (563.8 -> 531.9 -> 522.3 -> 506.0 -> 497.7 -> 479.9 as K rises 0->0.5->1.0->1.5->2.0->3.0).
This script:
  1. diagnoses whether the floor drop is concentrated in a few rolling windows or broad-based
  2. sweeps FINER/SMALLER K to find where the floor stops degrading
  3. tests a GRADUAL/continuous version (boost weight scales with how far a pair's correlation
     exceeds the significance threshold, instead of a hard binary in/out gate)
  4. tests VOL-REGIME gating (shrink the boost during high ALGO-vol stress periods, using the
     same causal volz construction SAFE_llvol's own vol-switch already relies on)
All precomputation (leader map, WZ, ALGO leg) is done ONCE and reused across every variant.
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
    if n_samples < 10: return 1.0
    alpha_adj = alpha / n_tests
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def boost_arr(i, upto, p):
    scale = np.nanstd(r[i, max(0, upto - 500):upto]) + 1e-12
    lret = r[i]
    return np.sign(lret) * (np.abs(lret) / scale) ** p


print("=== precompute (shared across every variant below) ===")
print("1. significance-gated leader map, at each checkpoint, plus a MARGIN (corr/threshold ratio) ===")
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
        margin = abs(best_corr[j]) / thr  # >1 once significant, how far past threshold
        entry[j] = (i, ic > 0, margin)
    STRONG_AT[cp] = entry
print("done")

print("2. shipped SAFE idio wz series ...")
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

print("3. shipped ALGO leg (unchanged) + ALGO causal vol-regime z-score (for vol-gating variant) ...")
algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(SAFE_llvol._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

VOL_WIN, VOL_Z = 20, 60
vol0 = np.full(nt - 1, np.nan)
vol0[VOL_WIN - 1:] = SAFE_llvol._roll_std(r[0], VOL_WIN)
volz_algo = np.full(nt - 1, np.nan)
for s in range(VOL_WIN + VOL_Z, nt - 1):
    wv = vol0[s - VOL_Z:s]
    volz_algo[s] = (vol0[s] - wv.mean()) / (wv.std() + 1e-12)
print("done")

BOOST_CACHE = {}
for cp, entry in STRONG_AT.items():
    for j, (i, gate, margin) in entry.items():
        if (i, cp) not in BOOST_CACHE:
            BOOST_CACHE[(i, cp)] = boost_arr(i, cp, 2.0)


def strong_for_day(k):
    valid = [c for c in CHECKPOINTS if c <= k]
    if not valid: return None, {}
    cp = max(valid)
    return cp, STRONG_AT[cp]


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)


def build_pos(mode, K, vol_gain=0.0):
    """mode: 'hard' (binary gate, fixed K) or 'gradual' (K scaled by margin, capped at 2x)."""
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        cp, entry = strong_for_day(k)
        vg = 1.0
        if vol_gain > 0 and k < len(volz_algo) and not np.isnan(volz_algo[k]):
            vg = float(np.clip(1.0 - vol_gain * max(volz_algo[k], 0.0), 0.0, 1.0))
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            boost = 0.0
            if K > 0 and cp is not None and j in entry:
                i, gate, margin = entry[j]
                if gate:
                    eff_k = K if mode == "hard" else K * min(margin, 2.0)
                    boost = eff_k * vg * BOOST_CACHE[(i, cp)][k - 1]
            sig = wz + boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    POS[0, :] = algo_pos
    return POS


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


print("\n=== baseline (K=0) ===")
base_POS = build_pos("hard", 0.0)
base_scs = report("baseline", base_POS)

print("\n=== 1. WINDOW CONCENTRATION diagnostic (hard gate, K=0.5 and K=1.5) ===")
for K in (0.5, 1.5):
    POS = build_pos("hard", K)
    scs = scs_curve(POS)
    diffs = scs - base_scs
    worse_idx = np.where(diffs < 0)[0]
    print(f"K={K}: {len(worse_idx)}/{len(base_scs)} windows worse. "
          f"worst 5 windows (end_day, baseline, candidate, diff):")
    order = np.argsort(diffs)[:5]
    for idx in order:
        print(f"    end_day={end_days[idx]:>4}  base={base_scs[idx]:>7.1f}  cand={scs[idx]:>7.1f}  diff={diffs[idx]:>+7.1f}")
    print(f"  spread of diffs across ALL worse windows: min={diffs[worse_idx].min():.1f} "
          f"max={diffs[worse_idx].max():.1f} mean={diffs[worse_idx].mean():.1f}  "
          f"(broad-based if not dominated by 1-2 outliers)")

print("\n=== 2. FINER/SMALLER K sweep (hard gate) ===")
for K in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4):
    report(f"hard K={K}", build_pos("hard", K), base_scs)

print("\n=== 3. GRADUAL (margin-scaled) boost, various base K ===")
for K in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    report(f"gradual K={K}", build_pos("gradual", K), base_scs)

print("\n=== 4. VOL-REGIME gated boost (hard gate K=0.5/1.5, shrink boost when ALGO volz > 0) ===")
for K in (0.5, 1.5):
    for vg in (0.5, 1.0, 1.5):
        report(f"hard K={K} volgain={vg}", build_pos("hard", K, vol_gain=vg), base_scs)

print("\n=== 5. COMBINED: gradual + vol-gating ===")
for K in (1.0, 1.5, 2.0, 3.0):
    for vg in (0.5, 1.0):
        report(f"gradual K={K} volgain={vg}", build_pos("gradual", K, vol_gain=vg), base_scs)

print("\n=== 6. MIN-CHECKPOINT gate: no boost at all until the leader map has enough history ===")


def build_pos_mincp(K, min_cp, mode="hard"):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        cp, entry = strong_for_day(k)
        for j in range(1, nInst):
            wz = WZ[k][j - 1]
            boost = 0.0
            if K > 0 and cp is not None and cp >= min_cp and j in entry:
                i, gate, margin = entry[j]
                if gate:
                    eff_k = K if mode == "hard" else K * min(margin, 2.0)
                    boost = eff_k * BOOST_CACHE[(i, cp)][k - 1]
            sig = wz + boost
            POS[j, k] = np.clip(np.sign(sig) * (dlr[j] / cur[j]), -lim[j], lim[j])
    POS[0, :] = algo_pos
    return POS


for min_cp in (400, 500, 600, 700, 800):
    for K in (0.5, 1.5, 2.0):
        report(f"min_cp={min_cp} K={K}", build_pos_mincp(K, min_cp), base_scs)

print("\n=== 7. FINE neighbor-stability sweep around min_cp=500, K=1.5 (plateau check, not a lucky point) ===")
for min_cp in (450, 480, 500, 520, 550):
    for K in (1.0, 1.25, 1.5, 1.75, 2.0):
        report(f"min_cp={min_cp} K={K}", build_pos_mincp(K, min_cp), base_scs)
