"""Different mechanism from every size-modulation attempt tonight: use Markov self-persistence of
the CURRENT regime cluster as a signal-trust indicator for TURNOVER, not SIZE. In low-persistence
("transitional/unstable") clusters, HOLD yesterday's ALGO position rather than reacting to today's
signal reading; in high-persistence ("stable") clusters, trade reactively as normal. This targets
entries/exits (when to act on a new reading) rather than conviction (how big to size it).
"""
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
import SAFE, SAFE_llvol as M

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

print("periodic causal K-means refit + causal Markov transition matrix (every 100 days) ...")
CHECKPOINTS = list(range(300, T, 100))
MODELS = {}
for cp in CHECKPOINTS:
    mask = valid[:cp]
    if mask.sum() < 100: continue
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit(feat[:cp][mask])
    lbls = np.full(cp, -1); lbls[mask] = km.labels_
    trans = np.zeros((4, 4))
    for t in range(cp - 1):
        a, b = lbls[t], lbls[t + 1]
        if a >= 0 and b >= 0: trans[a, b] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    persist = np.where(row_sums[:, 0] > 0, np.diag(trans) / row_sums[:, 0], 0.0)
    MODELS[cp] = (km, persist)


def cluster_persistence(t):
    valid_cps = [c for c in CHECKPOINTS if c <= t]
    if not valid_cps or not valid[t]: return None
    cp = max(valid_cps)
    if cp not in MODELS: return None
    km, persist = MODELS[cp]
    lbl = km.predict(feat[t:t + 1])[0]
    return persist[lbl]


print("computing shipped SAFE idio book ...")
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


def build_pos(persist_thresh):
    POS = idio_pos.copy()
    prev_algo_pos = 0
    n_hold = 0; n_days = 0
    for k in range(300, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        algo_shares = M._algo_vol_shares(lpA[:k + 1], cur[0], dlr[0])
        t = k - 1
        p = cluster_persistence(t) if t < T else None
        n_days += 1
        if persist_thresh is not None and p is not None and p < persist_thresh:
            new_algo = prev_algo_pos  # hold: unstable regime, don't react to today's reading
            n_hold += 1
        else:
            new_algo = algo_shares
        POS[0, k] = int(np.clip(new_algo, -lim[0], lim[0]))
        prev_algo_pos = POS[0, k]
    return POS, n_hold, n_days


def full_scs(POS):
    return np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days if E >= 550])


base_POS, _, _ = build_pos(None)
base_scs = full_scs(base_POS)
wo0 = window(base_POS, *OLD); wn0 = window(base_POS, *NEW)
print(f"\n{'config':<28}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}{'n_worse':>10}")
print(f"{'shipped (no hold logic)':<28}{wo0['score']:>8.1f}{wn0['score']:>8.1f}{base_scs.mean():>8.1f}{base_scs.min():>9.1f}")

for thresh in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    POS, n_hold, n_days = build_pos(thresh)
    scs = full_scs(POS)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"{'hold if persist<'+str(thresh):<28}{wo['score']:>8.1f}{wn['score']:>8.1f}{scs.mean():>8.1f}{scs.min():>9.1f}{nworse:>10}/{len(scs)}   hold_days={n_hold}/{n_days}")
