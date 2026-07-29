"""
test_batch100_J95_J96.py

J95: quick IC pre-check for a Support Vector Regression (sklearn SVR) on the same feature panel used
by the existing GBM tests (test_gbm_panel_v2.py's "fair" panel, including the causal leader_ret
feature), against v10's ridge.
J96: same pre-check for a bagged-tree Random Forest (sklearn RandomForestRegressor), distinct from the
already-tried HistGradientBoostingRegressor/GradientBoostingRegressor.

Reuses test_gbm_panel_v2.py's exact panel construction (own_ret, own_vol, algo_ret, xs_rank,
own_mom10, xs_mean, leader_ret -> next-day own return) and walk-forward checkpoint evaluation
(train on all data before cp, test on [cp, cp+100), roll cp forward), so the comparison against the
"GBM 0.0089 / Ridge 0.0099" baseline numbers already in this repo is apples-to-apples.
"""
import numpy as np, pandas as pd, time
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
P = P.values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
T = r.shape[1]
VOL_W = 20; MOM_W = 10


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


print("building the pooled panel (identical to test_gbm_panel_v2.py) ...", flush=True)
own_vol = np.full((nInst, T), np.nan)
for j in range(nInst):
    for t in range(VOL_W, T):
        own_vol[j, t] = r[j, t - VOL_W:t].std()
own_mom = np.full((nInst, T), np.nan)
for j in range(nInst):
    own_mom[j, MOM_W:] = np.array([r[j, t - MOM_W:t].sum() for t in range(MOM_W, T)])
xs_mean_others = np.array([r[1:, t].mean() for t in range(T)])
cross_rank = np.full((nInst, T), np.nan)
for t in range(T):
    cross_rank[1:, t] = pd.Series(r[1:, t]).rank(pct=True).values

rows = []
for j in range(1, nInst):
    for t in range(200, T - 1):
        leader = leader_for_day(t)
        lret = r[leader[j], t] if j in leader else 0.0
        rows.append([j, t, r[j, t], own_vol[j, t], r[0, t], cross_rank[j, t], own_mom[j, t],
                      xs_mean_others[t], lret, r[j, t + 1]])
panel = pd.DataFrame(rows, columns=["stock", "t", "own_ret", "own_vol", "algo_ret", "xs_rank",
                                     "own_mom10", "xs_mean", "leader_ret", "target"]).dropna()
print(f"panel shape: {panel.shape}")
FEATS = ["own_ret", "own_vol", "algo_ret", "xs_rank", "own_mom10", "xs_mean", "leader_ret"]

CHECKPOINTS = list(range(300, T - 1, 100))


def walk_forward(model_fn, name, max_train=None, seed=0):
    """max_train: if set, randomly subsample the training rows at each checkpoint to this many rows
    (cost control for kernel methods that scale poorly to tens of thousands of rows -- this is a
    quick IC pre-check, not a full production fit, per the house convention for this batch)."""
    preds = []; lin_preds = []; actual = []
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for cp in CHECKPOINTS:
        train = panel[panel["t"] < cp]
        test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
        if len(test) == 0: continue
        tr = train
        if max_train is not None and len(train) > max_train:
            tr = train.sample(n=max_train, random_state=rng)
        model = model_fn().fit(tr[FEATS], tr["target"])
        lin = Ridge(alpha=1.0).fit(train[FEATS], train["target"])
        preds.append(model.predict(test[FEATS])); lin_preds.append(lin.predict(test[FEATS]))
        actual.append(test["target"].values)
        print(f"    checkpoint {cp}: train n={len(tr)} (of {len(train)}), test n={len(test)}  "
              f"[{time.time()-t0:.0f}s elapsed]", flush=True)
    preds = np.concatenate(preds); lin_preds = np.concatenate(lin_preds); actual = np.concatenate(actual)
    ic = np.corrcoef(preds, actual)[0, 1]
    lic = np.corrcoef(lin_preds, actual)[0, 1]
    print(f"\n{name} OOS IC: {ic:.4f}   (Ridge-on-same-panel OOS IC: {lic:.4f})   [{time.time()-t0:.0f}s]")
    return ic, lic


print("\n=== J95: SVR (RBF kernel, C=1.0, standardized features) ===", flush=True)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def svr_model():
    return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.001))


svr_ic, svr_lin_ic = walk_forward(svr_model, "SVR", max_train=4000)

print("\n=== J96: Random Forest (bagged trees, n_estimators=200, max_depth=5) ===", flush=True)


def rf_model():
    return RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=20,
                                  random_state=0, n_jobs=2)


rf_ic, rf_lin_ic = walk_forward(rf_model, "RandomForest")

print("\n=== summary (context: earlier no-leader_ret GBM/Ridge baseline on this panel's ancestor was "
      "GBM 0.0089 / Ridge 0.0099, per test_gbm_panel_v2.py's own docstring/prints) ===")
print(f"  SVR OOS IC:           {svr_ic:.4f}   vs Ridge (same panel): {svr_lin_ic:.4f}")
print(f"  RandomForest OOS IC:  {rf_ic:.4f}   vs Ridge (same panel): {rf_lin_ic:.4f}")
