"""Portfolio-construction angle (not signal research): the idio book trades all 49 names at a flat
$10k regardless of how much of each name's risk is SYSTEMATIC (shared ALGO-beta exposure) vs
IDIOSYNCRATIC (the actual diversifying part). Recall R^2-to-ALGO varies from near-0 to 0.373 across
names -- a high-beta name's "independent" bet is secretly more correlated with the other 48 than a
low-beta name's, creating a hidden concentrated factor bet inside what looks like pure breadth. Test
down-weighting (never up-weighting, respecting the $10k cap) position size by causal rolling beta/R^2
to ALGO, to see if reducing that hidden shared-risk concentration improves the aggregate Sharpe.
"""
import numpy as np, pandas as pd
import SAFE

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
Praw = P.values.T.astype(float)
nInst, nt = Praw.shape
logp = np.log(Praw)
r = np.diff(logp, axis=1)
T = r.shape[1]
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def score_fn(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = Praw[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score_fn(tot.mean(), tot.std())}


def full_scs(POS):
    return np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days])


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

print("computing causal rolling beta/R^2 to ALGO per stock (250-day trailing window) ...")
BETA_W = 250
r2_series = np.full((nInst, T), np.nan)
for j in range(1, nInst):
    for t in range(BETA_W, T):
        r0w = r[0, t - BETA_W:t]; rjw = r[j, t - BETA_W:t]
        b = np.polyfit(r0w, rjw, 1)[0]
        resid_var = np.var(rjw - b * r0w)
        tot_var = np.var(rjw)
        r2_series[j, t] = 1 - resid_var / (tot_var + 1e-18) if tot_var > 1e-18 else 0.0
print("done")


def build_pos(weight_fn):
    POS = np.zeros((nInst, nt))
    for k in range(SAFE.WARMUP, nt):
        cur = Praw[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[k]
        for j in range(1, nInst):
            r2 = r2_series[j, k - 1] if k - 1 < T and not np.isnan(r2_series[j, k - 1]) else 0.0
            w = weight_fn(r2)
            POS[j, k] = np.clip(np.sign(wz[j - 1]) * w * (dlr[j] / cur[j]), -lim[j], lim[j])
    return POS


base_POS = build_pos(lambda r2: 1.0)
base_scs = full_scs(base_POS)
wo0 = window(base_POS, *OLD); wn0 = window(base_POS, *NEW)
print(f"\nshipped (flat $10k, no risk-weighting): OLD={wo0['score']:.1f} NEW={wn0['score']:.1f} "
      f"rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")

print("\n--- linear down-weight by R^2: weight = 1 - k*R^2 ---")
for k_ in (0.3, 0.5, 0.7, 1.0):
    POS = build_pos(lambda r2, k_=k_: max(1.0 - k_ * r2, 0.2))
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"k={k_}: OLD={wo['score']:>7.1f} NEW={wn['score']:>7.1f} "
          f"rmean={scs.mean():>7.1f} rfloor={scs.min():>7.1f}  n_worse={nworse}/{len(scs)}")

print("\n--- threshold cut: full size unless R^2 > thresh, then downweight to frac ---")
for thresh, frac in [(0.15, 0.5), (0.15, 0.7), (0.2, 0.5), (0.25, 0.5), (0.1, 0.6)]:
    POS = build_pos(lambda r2, thresh=thresh, frac=frac: frac if r2 > thresh else 1.0)
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"thresh={thresh},frac={frac}: OLD={wo['score']:>7.1f} NEW={wn['score']:>7.1f} "
          f"rmean={scs.mean():>7.1f} rfloor={scs.min():>7.1f}  n_worse={nworse}/{len(scs)}")
