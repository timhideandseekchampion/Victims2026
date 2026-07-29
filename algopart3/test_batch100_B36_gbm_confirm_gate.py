"""
test_batch100_B36_gbm_confirm_gate.py

B36: Re-test a GBM confirm-gate (trade only when v10's forecast and a leader-only
HistGradientBoostingRegressor agree in sign) against v10 ITSELF as the base, not an older version
(prior GBM-combination work -- test_combine_ridge_gbm.py -- gated the OLD SAFE ridge, before the
beta-adjusted target, boost, or rank-stability blend existed).

GBM feature panel reused verbatim from test_combine_ridge_gbm.py's "leader-only" panel (the one that
achieved standalone IC=0.0366): own_ret, own_ret_lag1, own_vol(20d), algo_ret, cross-sectional rank,
mom5, mom20, cross-sectional mean, causal leader return + its lag (leader identified by expanding-
window pairwise correlation, re-estimated every 50 days), ALGO vol z-score, stock identity (categorical).
Retrained at rolling 100-day checkpoints (walk-forward, no lookahead) exactly as before.

MECHANISM: trade sign(v10's actual final wz) only when the GBM's predicted sign for that (stock, day)
agrees; else flat (0), matching the original confirm-gate convention. Falls back to full v10 sign
when no GBM prediction is available for that (stock, day) (pre-panel-warmup days).
"""
import numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor
import pandas as pd
from batch100_shared import (
    nInst, nIdio, nt, P_, logp, r, dlr, days, algo_pos, WZ_FULL, base_wo, base_wn, base_scs,
    SANITY_OK, evaluate
)

print(f"\n=== B36 sanity check (shared precompute) reproduces v10: {'PASS' if SANITY_OK else 'FAIL'} ===")
print(f"  OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")

T = r.shape[1]  # nt-1


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("\nbuilding leader-only GBM panel (verbatim feature set from test_combine_ridge_gbm.py) ...")
t0 = time.time()
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
import SAFE_llvol as M
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
FEATS = ["stock", "own_ret", "own_ret_lag1", "own_vol", "algo_ret", "xs_rank", "mom5", "mom20",
         "xs_mean", "leader_ret", "leader_ret_lag1", "algo_volz"]
print(f"  panel built, shape={panel.shape} ({time.time()-t0:.0f}s)")

print("\ntraining GBM at rolling 100-day checkpoints (walk-forward) ...")
t0 = time.time()
CHECKPOINTS = list(range(300, T - 1, 100))
gbm_sig = {}
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    gbm = HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.02,
                                         l2_regularization=5.0, categorical_features=["stock"],
                                         random_state=0).fit(train[FEATS], train["target"])
    gp = gbm.predict(test[FEATS])
    for (stk, t), gv in zip(zip(test["stock"], test["t"]), gp):
        gbm_sig[(stk, t)] = gv
print(f"  GBM signal computed for {len(gbm_sig)} (stock, day) pairs ({time.time()-t0:.0f}s)")

# diagnostics: standalone GBM IC + agreement rate with v10's actual wz
pairs = [(j, t) for (j, t) in gbm_sig if t < nt]
gbm_vals = np.array([gbm_sig[(j, t)] for (j, t) in pairs])
actual = np.array([r[j, t + 1] for (j, t) in pairs])
print(f"\nstandalone GBM IC (pred[t] vs actual next-day return): {np.corrcoef(gbm_vals, actual)[0,1]:.4f}")
v10_vals = np.array([WZ_FULL[j - 1, t] for (j, t) in pairs])
agree = (np.sign(v10_vals) == np.sign(gbm_vals))
print(f"sign agreement rate (v10 wz vs GBM): {agree.mean()*100:.1f}%  ({agree.sum()}/{len(agree)})")

first_t = min(t for (_, t) in gbm_sig)


def build_pos_confirm_gate():
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_FULL[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        sgn = np.sign(wz).copy()
        for jj in range(nIdio):
            j = jj + 1
            if (j, t) in gbm_sig:
                gsign = np.sign(gbm_sig[(j, t)])
                if gsign != sgn[jj]:
                    sgn[jj] = 0.0
        POS[1:, t] = np.clip(sgn * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== B36: confirm-gate (trade v10's sign only when GBM agrees; else flat) ===")
Pz = build_pos_confirm_gate()
res = evaluate("GBM confirm-gate", Pz)
print(f"\n{'PASS' if res['passed'] else 'FAIL'}: confirm-gate vs v10 on OLD+NEW+rmean jointly.")
