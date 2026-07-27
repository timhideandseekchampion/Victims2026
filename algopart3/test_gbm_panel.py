"""Completely different model class: instead of a linear ridge cross-sectional regression, pool ALL
49 stocks' daily observations into one panel (~870 days x 49 names ~ 42,000 rows -- solves the
"not enough data per name" problem that broke GARCH-M and the causal pairwise tests) and fit a
gradient-boosted tree model. Tree ensembles can capture non-linearities/interactions a linear model
can't -- this directly tests the "maybe the gap to top teams is a non-linear model class" hypothesis
raised earlier. Feature importances are the "parameters" to inspect. Walk-forward causal validation
throughout (train on an expanding window, predict strictly out-of-sample, roll forward).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)   # (51, T) T=nt-1
T = r.shape[1]

VOL_W = 20; MOM_W = 10

print("building the pooled feature panel (all 49 stocks stacked) ...")
own_vol = np.full((nInst, T), np.nan)
for j in range(nInst):
    for t in range(VOL_W, T):
        own_vol[j, t] = r[j, t - VOL_W:t].std()
own_mom = np.full((nInst, T), np.nan)
for j in range(nInst):
    own_mom[j, MOM_W:] = np.array([r[j, t - MOM_W:t].sum() for t in range(MOM_W, T)])
xs_mean_others = np.full(T, np.nan)
for t in range(T):
    xs_mean_others[t] = r[1:, t].mean()
cross_rank = np.full((nInst, T), np.nan)
for t in range(T):
    ranks = pd.Series(r[1:, t]).rank(pct=True).values
    cross_rank[1:, t] = ranks

rows = []
for j in range(1, nInst):
    for t in range(50, T - 1):
        rows.append([j, t, r[j, t], own_vol[j, t], r[0, t], cross_rank[j, t], own_mom[j, t],
                      xs_mean_others[t], r[j, t + 1]])
panel = pd.DataFrame(rows, columns=["stock", "t", "own_ret", "own_vol", "algo_ret", "xs_rank",
                                     "own_mom10", "xs_mean", "target"])
panel = panel.dropna()
print(f"panel shape: {panel.shape}")

FEATS = ["own_ret", "own_vol", "algo_ret", "xs_rank", "own_mom10", "xs_mean"]

print("\nwalk-forward: train on expanding window, predict the next 100-day block, roll forward ...")
CHECKPOINTS = list(range(300, T - 1, 100))
oos_preds = []; oos_actual = []; oos_t = []
importances_over_time = []
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    model = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05,
                                       subsample=0.8, random_state=0)
    model.fit(train[FEATS], train["target"])
    pred = model.predict(test[FEATS])
    oos_preds.append(pred); oos_actual.append(test["target"].values); oos_t.append(test["t"].values)
    importances_over_time.append(model.feature_importances_)

oos_preds = np.concatenate(oos_preds); oos_actual = np.concatenate(oos_actual)
ic = np.corrcoef(oos_preds, oos_actual)[0, 1]
print(f"\nGBM pooled-panel OUT-OF-SAMPLE IC (walk-forward, causal): {ic:.4f}")

# compare to a LINEAR (ridge-like) benchmark on the SAME feature set / same walk-forward splits
from sklearn.linear_model import Ridge
lin_preds = []
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    lm = Ridge(alpha=1.0).fit(train[FEATS], train["target"])
    lin_preds.append(lm.predict(test[FEATS]))
lin_preds = np.concatenate(lin_preds)
ic_lin = np.corrcoef(lin_preds, oos_actual)[0, 1]
print(f"Ridge (same features, same splits) OUT-OF-SAMPLE IC: {ic_lin:.4f}")

avg_imp = np.mean(importances_over_time, axis=0)
print("\nGBM feature importances (averaged across walk-forward refits):")
for f, imp in sorted(zip(FEATS, avg_imp), key=lambda x: -x[1]):
    print(f"  {f:<12} {imp:.3f}")
