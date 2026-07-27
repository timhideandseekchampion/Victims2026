"""Fair version: give the tree model (a) stock identity as a feature so it can learn PER-STOCK
structure instead of one pooled rule, (b) a richer feature set (multiple lags, ALGO's own vol
regime, momentum at two horizons) so it has access to at least as much information as SAFE.py's
ridge implicitly uses, and (c) a real hyperparameter search via causal validation, not eyeballed
defaults. Uses HistGradientBoostingRegressor (native categorical support, sklearn 1.9).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import SAFE, SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
Praw = P.values.T.astype(float)
nInst, nt = Praw.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(Praw)
r = np.diff(logp, axis=1)
T = r.shape[1]


def score(mu, sd):
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
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


CHECKPOINTS_LEADER = list(range(150, T, 50))
LEADER_AT = {}
for cp in CHECKPOINTS_LEADER:
    Xi = r[1:, :cp - 1]; Yj = r[1:, 1:cp]
    n = nInst - 1
    C = corrmat(Xi, Yj)
    leader = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col))); leader[j + 1] = i + 1
    LEADER_AT[cp] = leader


def leader_for_day(k):
    valid = [c for c in CHECKPOINTS_LEADER if c <= k]
    return LEADER_AT[max(valid)] if valid else {}


print("building a richer per-stock-aware panel ...")
VOL_W = 20
own_vol = np.full((nInst, T), np.nan)
for j in range(nInst):
    for t in range(VOL_W, T):
        own_vol[j, t] = r[j, t - VOL_W:t].std()
mom5 = np.full((nInst, T), np.nan); mom20 = np.full((nInst, T), np.nan)
for j in range(nInst):
    mom5[j, 5:] = np.array([r[j, t - 5:t].sum() for t in range(5, T)])
    mom20[j, 20:] = np.array([r[j, t - 20:t].sum() for t in range(20, T)])
xs_mean_others = np.array([r[1:, t].mean() for t in range(T)])
cross_rank = np.full((nInst, T), np.nan)
for t in range(T):
    cross_rank[1:, t] = pd.Series(r[1:, t]).rank(pct=True).values
# ALGO's own vol-regime feature (same volz used by the shipped model's ALGO leg)
vol0 = np.full(T, np.nan); vol0[M.VOL_WIN - 1:] = M._roll_std(r[0], M.VOL_WIN)
volz0 = np.full(T, np.nan)
for s in range(M.VOL_WIN + M.VOL_Z, T):
    wv = vol0[s - M.VOL_Z:s]; volz0[s] = (vol0[s] - wv.mean()) / (wv.std() + 1e-12)

rows = []
for j in range(1, nInst):
    for t in range(200, T - 1):
        leader = leader_for_day(t)
        li = leader.get(j, 0)
        lret = r[li, t] if li else 0.0
        lret_lag1 = r[li, t - 1] if li else 0.0
        rows.append([j, t, r[j, t], r[j, t - 1], own_vol[j, t], r[0, t], cross_rank[j, t],
                      mom5[j, t], mom20[j, t], xs_mean_others[t], lret, lret_lag1,
                      volz0[t] if not np.isnan(volz0[t]) else 0.0, r[j, t + 1]])
panel = pd.DataFrame(rows, columns=["stock", "t", "own_ret", "own_ret_lag1", "own_vol", "algo_ret",
                                     "xs_rank", "mom5", "mom20", "xs_mean", "leader_ret",
                                     "leader_ret_lag1", "algo_volz", "target"]).dropna()
panel["stock"] = panel["stock"].astype("category")
print(f"panel shape: {panel.shape}")
FEATS = ["stock", "own_ret", "own_ret_lag1", "own_vol", "algo_ret", "xs_rank", "mom5", "mom20",
         "xs_mean", "leader_ret", "leader_ret_lag1", "algo_volz"]

# --- hyperparameter search via causal validation: train < 500, validate on 500-650 ---
train_hp = panel[panel["t"] < 500]
val_hp = panel[(panel["t"] >= 500) & (panel["t"] < 650)]
print("\nhyperparameter search (causal validation on days 500-650) ...")
best = None
for max_iter in (50, 100, 200):
    for max_depth in (2, 3, 4, None):
        for lr in (0.02, 0.05, 0.1):
            for l2 in (0.0, 1.0, 5.0):
                m = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth,
                                                   learning_rate=lr, l2_regularization=l2,
                                                   categorical_features=["stock"], random_state=0)
                m.fit(train_hp[FEATS], train_hp["target"])
                pred = m.predict(val_hp[FEATS])
                ic = np.corrcoef(pred, val_hp["target"])[0, 1]
                if best is None or ic > best[0]:
                    best = (ic, max_iter, max_depth, lr, l2)
print(f"best hyperparams by validation IC: IC={best[0]:.4f}  max_iter={best[1]} max_depth={best[2]} lr={best[3]} l2={best[4]}")

MAX_ITER, MAX_DEPTH, LR, L2 = best[1], best[2], best[3], best[4]

CHECKPOINTS = list(range(300, T - 1, 100))
gbm_sig = {}
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    gbm = HistGradientBoostingRegressor(max_iter=MAX_ITER, max_depth=MAX_DEPTH, learning_rate=LR,
                                         l2_regularization=L2, categorical_features=["stock"],
                                         random_state=0).fit(train[FEATS], train["target"])
    gp = gbm.predict(test[FEATS])
    for (stk, t), gv in zip(zip(test["stock"], test["t"]), gp):
        gbm_sig[(stk, t)] = gv

full_ic = np.corrcoef([gbm_sig[k] for k in gbm_sig], [panel.set_index(["stock","t"]).loc[k,"target"] for k in gbm_sig])[0,1]
print(f"\nfull walk-forward OOS IC (tuned, stock-aware GBM): {full_ic:.4f}  (earlier untuned pooled GBM: 0.0177)")

print("building position matrix + scoring ...")
POS_gbm = np.zeros((nInst, nt))
first_t = min(t for (_, t) in gbm_sig)
for j in range(1, nInst):
    for t in range(first_t, T - 1):
        if (j, t) not in gbm_sig: continue
        k = t + 1
        cur = Praw[j, k]; lim = int(dlr[j] / cur)
        POS_gbm[j, k] = np.clip(np.sign(gbm_sig[(j, t)]) * (dlr[j] / cur), -lim, lim)

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))
FULL = (first_t + 2, nt)


def report(nm, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW); wf = window(POS, *FULL)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days if E - NUMTEST >= first_t]
    print(f"{nm:<20}FULL={wf['score']:>8.1f}  OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={np.mean(scs):>8.1f}  rfloor={min(scs):>8.1f}")


report("tuned stock-aware GBM", POS_gbm)

idio_shipped = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = Praw[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(Praw[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_shipped[:, k] = np.clip(p, -lim, lim).astype(int)
report("shipped SAFE idio", idio_shipped)
