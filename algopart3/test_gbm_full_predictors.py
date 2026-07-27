"""Full-information version: give the GBM the SAME 51-dimensional predictor set SAFE.py's ridge
uses (every instrument's today's return, not just the single identified best leader), plus stock
identity so it can learn which of those 51 columns matters for each specific target. This is the
honest, complete test of whether a tree model can match/beat the ridge given equivalent information.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import SAFE

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


print("building the FULL 51-predictor panel (every instrument's today's return, per row) ...")
VOL_W = 20
own_vol = np.full((nInst, T), np.nan)
for j in range(nInst):
    for t in range(VOL_W, T):
        own_vol[j, t] = r[j, t - VOL_W:t].std()
mom5 = np.full((nInst, T), np.nan); mom20 = np.full((nInst, T), np.nan)
for j in range(nInst):
    mom5[j, 5:] = np.array([r[j, t - 5:t].sum() for t in range(5, T)])
    mom20[j, 20:] = np.array([r[j, t - 20:t].sum() for t in range(20, T)])
cross_rank = np.full((nInst, T), np.nan)
for t in range(T):
    cross_rank[1:, t] = pd.Series(r[1:, t]).rank(pct=True).values

R_COLS = [f"r{i}" for i in range(nInst)]
rT = r.T  # (T, nInst) -- row t = all instruments' return on day t

extra_rows = []
for j in range(1, nInst):
    for t in range(200, T - 1):
        extra_rows.append([j, t, own_vol[j, t], mom5[j, t], mom20[j, t], cross_rank[j, t], r[j, t + 1]])
extra = pd.DataFrame(extra_rows, columns=["stock", "t", "own_vol", "mom5", "mom20", "xs_rank", "target"])
rdf = pd.DataFrame(rT, columns=R_COLS)
rdf["t"] = np.arange(T)
panel = extra.merge(rdf, on="t", how="left").dropna()
panel["stock"] = panel["stock"].astype("category")
print(f"panel shape: {panel.shape}  (columns: {panel.shape[1]})")
FEATS = ["stock", "own_vol", "mom5", "mom20", "xs_rank"] + R_COLS

print("\nhyperparameter search (causal validation on days 500-650), centered on the earlier winner ...")
train_hp = panel[panel["t"] < 500]
val_hp = panel[(panel["t"] >= 500) & (panel["t"] < 650)]
best = None
for max_iter in (100, 200, 300):
    for max_depth in (3, 4, 6):
        for lr in (0.01, 0.02, 0.05):
            for l2 in (1.0, 5.0, 15.0):
                m = HistGradientBoostingRegressor(max_iter=max_iter, max_depth=max_depth,
                                                   learning_rate=lr, l2_regularization=l2,
                                                   categorical_features=["stock"], random_state=0)
                m.fit(train_hp[FEATS], train_hp["target"])
                pred = m.predict(val_hp[FEATS])
                ic = np.corrcoef(pred, val_hp["target"])[0, 1]
                if best is None or ic > best[0]:
                    best = (ic, max_iter, max_depth, lr, l2)
print(f"best: IC={best[0]:.4f}  max_iter={best[1]} max_depth={best[2]} lr={best[3]} l2={best[4]}")
MAX_ITER, MAX_DEPTH, LR, L2 = best[1], best[2], best[3], best[4]

CHECKPOINTS = list(range(300, T - 1, 100))
gbm_sig = {}
imp_accum = None
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

targets_lookup = panel.set_index(["stock", "t"])["target"]
preds = np.array([gbm_sig[k] for k in gbm_sig])
actuals = np.array([targets_lookup.loc[k] for k in gbm_sig])
print(f"\nfull walk-forward OOS IC (full 51-predictor GBM): {np.corrcoef(preds, actuals)[0,1]:.4f}  "
      f"(leader-only tuned version: 0.0366)")

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
    print(f"{nm:<24}FULL={wf['score']:>8.1f}  OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={np.mean(scs):>8.1f}  rfloor={min(scs):>8.1f}")


report("full-predictor GBM", POS_gbm)

idio_shipped = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = Praw[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(Praw[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_shipped[:, k] = np.clip(p, -lim, lim).astype(int)
report("shipped SAFE idio", idio_shipped)
