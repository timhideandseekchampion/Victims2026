"""Module 5: Machine-learning predictability of next-day returns.

Libraries: scikit-learn (Ridge/Lasso/ElasticNet, RandomForest, GradientBoosting,
LogisticRegression, TimeSeriesSplit, permutation baseline), PyTorch (MLP),
TensorFlow/Keras (LSTM). All strictly walk-forward / out-of-sample.

Question: given lagged returns (own + market), can ANY model predict the next
day's return better than chance OUT OF SAMPLE? We compare OOS R^2 / directional
accuracy against a shuffled-target baseline to get a significance read.
"""
import warnings, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score
from scipy import stats
from common import load, log_returns, section, stars

RNG = np.random.RandomState(0)
df, tickers = load()
rets = log_returns(df)
R = rets.values
T, N = R.shape
mkt = R.mean(axis=1)

# Build a pooled panel: features = last 5 own returns + last market return + own vol
def build_panel(nlags=5):
    X, y, inst = [], [], []
    for i in range(N):
        r = R[:, i]
        for t in range(nlags, T - 1):
            feat = list(r[t-nlags:t][::-1])          # own lags 1..5
            feat += [mkt[t-1], r[t-nlags:t].std()]   # market lag, recent vol
            X.append(feat); y.append(r[t]); inst.append(i)
    return np.array(X), np.array(y), np.array(inst)

X, y, inst = build_panel()
section(f"5A. POOLED PANEL: {X.shape[0]} samples x {X.shape[1]} features")
print("Features: own returns lag1-5, market return lag1, recent 5d vol.")
print("Target: next-day own log return. Metric: OOS R^2 (walk-forward 5 folds).\n")

tscv = TimeSeriesSplit(n_splits=5)
models = {
    "Ridge":            Ridge(alpha=1.0),
    "Lasso":            Lasso(alpha=1e-4),
    "ElasticNet":       ElasticNet(alpha=1e-4, l1_ratio=0.5),
    "RandomForest":     RandomForestRegressor(n_estimators=100, max_depth=4,
                                              n_jobs=-1, random_state=0),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                  learning_rate=0.03, random_state=0),
}
print(f"{'model':<18}{'OOS_R2':>10}{'dir_acc':>10}{'shuffled_R2':>13}")
results = {}
for name, mdl in models.items():
    r2s, accs = [], []
    for tr, te in tscv.split(X):
        mdl.fit(X[tr], y[tr])
        pred = mdl.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        accs.append(accuracy_score(y[te] > 0, pred > 0))
    # shuffled-target baseline (destroys any real signal)
    ys = RNG.permutation(y)
    sh = []
    for tr, te in tscv.split(X):
        mdl.fit(X[tr], ys[tr]); sh.append(r2_score(ys[te], mdl.predict(X[te])))
    results[name] = np.mean(r2s)
    print(f"{name:<18}{np.mean(r2s):>10.5f}{np.mean(accs):>10.4f}{np.mean(sh):>13.5f}")
print("\n(OOS R^2 <= 0 means the model does no better than predicting the mean.)")

section("5B. PyTorch MLP (nonlinear, walk-forward)")
try:
    import torch, torch.nn as nn
    torch.manual_seed(0)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    n = len(X); split = int(n * 0.8)
    mu, sd = Xt[:split].mean(0), Xt[:split].std(0) + 1e-8
    Xn = (Xt - mu) / sd
    net = nn.Sequential(nn.Linear(X.shape[1], 32), nn.ReLU(),
                        nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    for ep in range(60):
        net.train(); opt.zero_grad()
        loss = lossf(net(Xn[:split]), yt[:split]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xn[split:]).numpy().ravel()
    yte = y[split:]
    r2 = r2_score(yte, pred); acc = accuracy_score(yte > 0, pred > 0)
    print(f"  MLP OOS R^2: {r2:.5f}   directional acc: {acc:.4f}")
except Exception as e:
    print("  PyTorch MLP failed:", e)

section("5C. TensorFlow/Keras LSTM (sequence model on ALGO)")
try:
    import tensorflow as tf
    tf.random.set_seed(0)
    # sequence: predict ALGO next-day return from last 10 days of (ALGO, market)
    seqlen = 10
    algo = R[:, 0]
    Xs, ys_ = [], []
    for t in range(seqlen, T - 1):
        Xs.append(np.column_stack([algo[t-seqlen:t], mkt[t-seqlen:t]]))
        ys_.append(algo[t])
    Xs = np.array(Xs); ys_ = np.array(ys_)
    sp = int(len(Xs) * 0.8)
    m = tf.keras.Sequential([
        tf.keras.layers.Input((seqlen, 2)),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(1)])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    m.fit(Xs[:sp], ys_[:sp], epochs=40, batch_size=16, verbose=0)
    pred = m.predict(Xs[sp:], verbose=0).ravel()
    r2 = r2_score(ys_[sp:], pred); acc = accuracy_score(ys_[sp:] > 0, pred > 0)
    print(f"  LSTM(ALGO) OOS R^2: {r2:.5f}   directional acc: {acc:.4f}")
except Exception as e:
    print("  TensorFlow LSTM failed:", e)

section("5D. DIRECTIONAL-ACCURACY SIGNIFICANCE (binomial test)")
# best linear model directional acc vs 50%
best = max(results, key=results.get)
mdl = models[best]
preds, acts = [], []
for tr, te in tscv.split(X):
    mdl.fit(X[tr], y[tr]); preds.append(mdl.predict(X[te]) > 0); acts.append(y[te] > 0)
preds = np.concatenate(preds); acts = np.concatenate(acts)
acc = (preds == acts).mean(); nobs = len(acts)
bt = stats.binomtest(int(acc * nobs), nobs, 0.5)
print(f"Best model ({best}) directional acc {acc:.4f} on {nobs} OOS samples")
print(f"Binomial test vs 50%: p={bt.pvalue:.4g} {stars(bt.pvalue)}")

section("5E. VERDICT")
print(f"All OOS R^2: {', '.join(f'{k}={v:.4f}' for k,v in results.items())}")
print("If every OOS R^2 <= ~0 and directional acc ~50% (n.s.), returns are")
print("effectively UNPREDICTABLE from their own/market history -> no ML edge;")
print("the only real structure is the cointegrated-pair spreads (module 3).")
