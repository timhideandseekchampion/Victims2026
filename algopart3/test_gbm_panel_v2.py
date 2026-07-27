"""Extend the GBM panel test: add each stock's OWN best-leader's return as a feature (causally
re-estimated, same expanding-window checkpoint approach used all night), so the model has access to
the SAME specific pairwise information SAFE.py's ridge already exploits. This is the fair test of
"does a non-linear model find MORE than the linear ridge from the same information" -- the earlier
version only had generic/aggregate features and was missing the one feature we know carries signal.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
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


print("building the pooled panel with a causal leader-return feature ...")
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
gbm_preds = []; lin_preds = []; actual = []
importances_over_time = []
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=0).fit(train[FEATS], train["target"])
    lin = Ridge(alpha=1.0).fit(train[FEATS], train["target"])
    gbm_preds.append(gbm.predict(test[FEATS])); lin_preds.append(lin.predict(test[FEATS]))
    actual.append(test["target"].values)
    importances_over_time.append(gbm.feature_importances_)

gbm_preds = np.concatenate(gbm_preds); lin_preds = np.concatenate(lin_preds); actual = np.concatenate(actual)
print(f"\nGBM (with leader_ret feature)  OOS IC: {np.corrcoef(gbm_preds, actual)[0,1]:.4f}")
print(f"Ridge (with leader_ret feature) OOS IC: {np.corrcoef(lin_preds, actual)[0,1]:.4f}")
print(f"(earlier, WITHOUT leader_ret:  GBM 0.0089, Ridge 0.0099)")

avg_imp = np.mean(importances_over_time, axis=0)
print("\nGBM feature importances (with leader_ret added):")
for f, imp in sorted(zip(FEATS, avg_imp), key=lambda x: -x[1]):
    print(f"  {f:<12} {imp:.3f}")

# does a leader_ret x own_vol INTERACTION show up as valuable (the convexity-like effect)?
combo = pd.concat([pd.Series(gbm_preds, name="gbm"), pd.Series(lin_preds, name="lin"),
                    pd.Series(actual, name="y")], axis=1)
print(f"\ncorr(gbm_pred, lin_pred): {combo['gbm'].corr(combo['lin']):.3f}  (how similar are the two models' predictions?)")
