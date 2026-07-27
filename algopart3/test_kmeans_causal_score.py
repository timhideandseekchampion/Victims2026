"""Causal version: periodically refit K-means (every 100 days, expanding window) on the 4-feature
regime-state vector, identify the "high-vol, strong-IC" cluster at each refit by its center's volz
value (not by label, which is arbitrary across refits), and test whether being in that cluster today
(boost) vs not (no boost / reduced size) improves the ALGO leg's score over the shipped mechanism.
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
lpA = logp[0]
r = np.diff(logp, axis=1)
r0 = np.diff(lpA)
T = r.shape[1]
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
ret1 = np.full(len(lpA), np.nan); ret1[:-1] = lpA[1:] - lpA[:-1]
ret1_T = ret1[:T]

vol20 = np.full(T, np.nan); vol20[19:] = M._roll_std(r0, 20)
volz = np.full(T, np.nan)
for s in range(80, T):
    w = vol20[s - 60:s]; volz[s] = (vol20[s] - w.mean()) / (w.std() + 1e-12)
vol_of_vol = np.full(T, np.nan)
for s in range(80, T):
    w = vol20[s - 60:s]; ok = ~np.isnan(w)
    if ok.sum() > 20: vol_of_vol[s] = w[ok].std() / (w[ok].mean() + 1e-12)
mom_raw = np.full(T, np.nan)
for t in range(10, T):
    mom_raw[t] = lpA[t] - lpA[t - 10]
momz = np.full(T, np.nan)
for s in range(80, T):
    w = mom_raw[s - 60:s]; ok = ~np.isnan(w)
    if ok.sum() > 20: momz[s] = (mom_raw[s] - w[ok].mean()) / (w[ok].std() + 1e-12)
dispersion = np.array([r[1:, t].std() for t in range(T)])
dispz = np.full(T, np.nan)
for s in range(80, T):
    w = dispersion[s - 60:s]; dispz[s] = (dispersion[s] - w.mean()) / (w.std() + 1e-12)

feat = np.column_stack([volz, vol_of_vol, momz, dispz])
valid = ~np.any(np.isnan(feat), axis=1)

print("periodic causal K-means refit (every 100 days, expanding window) ...")
CHECKPOINTS = list(range(300, T, 100))
HIGH_VOL_CLUSTER = {}   # checkpoint -> (kmeans model, index of the high-volz cluster)
for cp in CHECKPOINTS:
    mask = valid[:cp]
    if mask.sum() < 100: continue
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(feat[:cp][mask])
    high_idx = int(np.argmax(km.cluster_centers_[:, 0]))  # cluster with highest mean volz
    HIGH_VOL_CLUSTER[cp] = (km, high_idx)


def in_high_vol_cluster(t):
    valid_cps = [c for c in CHECKPOINTS if c <= t]
    if not valid_cps or not valid[t]: return False
    cp = max(valid_cps)
    if cp not in HIGH_VOL_CLUSTER: return False
    km, high_idx = HIGH_VOL_CLUSTER[cp]
    lbl = km.predict(feat[t:t + 1])[0]
    return lbl == high_idx


print("computing shipped SAFE idio book ...")
import SAFE
idio_pos = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_pos[:, k] = np.clip(p, -lim, lim).astype(int)
print("done")


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


OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos(boost_gain, low_frac):
    """When in the high-vol cluster: shipped ALGO signal * boost_gain (extra conviction on top of
    SWITCH_GAIN, still capped at $100k). Otherwise: shipped ALGO signal * low_frac (never flip sign,
    just modulate size -- same lesson as tonight's other desize tests: never override direction)."""
    POS = idio_pos.copy()
    for k in range(300, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        algo_shares = M._algo_vol_shares(lpA[:k + 1], cur[0], dlr[0])
        t = k - 1
        mult = boost_gain if (t < T and in_high_vol_cluster(t)) else low_frac
        av = algo_shares * cur[0] * mult
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    return POS


def full_scs(POS):
    return np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days if E >= 550])


base_POS = build_pos(1.0, 1.0)
base_scs = full_scs(base_POS)
wo0 = window(base_POS, *OLD); wn0 = window(base_POS, *NEW)
print(f"\n{'config':<28}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}{'n_worse':>10}")
print(f"{'shipped (no cluster info)':<28}{wo0['score']:>8.1f}{wn0['score']:>8.1f}{base_scs.mean():>8.1f}{base_scs.min():>9.1f}")

for boost, lowf in [(1.3, 1.0), (1.5, 1.0), (2.0, 1.0), (1.0, 0.7), (1.5, 0.7), (1.3, 0.5), (2.0, 0.5)]:
    POS = build_pos(boost, lowf)
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"{'boost='+str(boost)+',low='+str(lowf):<28}{wo['score']:>8.1f}{wn['score']:>8.1f}{scs.mean():>8.1f}{scs.min():>9.1f}{nworse:>10}/{len(scs)}")
