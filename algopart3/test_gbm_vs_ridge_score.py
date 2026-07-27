"""Convert the GBM and Ridge OOS predictions from the panel test into actual trading positions
(sign-based, full $10k/name, same convention as the rest of the idio book) and score both with the
exact eval-mirroring accounting used all night -- IC alone doesn't capture the Sharpe/consistency
that the score formula rewards.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
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
VOL_W = 20; MOM_W = 10


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


print("rebuilding the pooled panel (same as before) ...")
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
FEATS = ["own_ret", "own_vol", "algo_ret", "xs_rank", "own_mom10", "xs_mean", "leader_ret"]

CHECKPOINTS = list(range(300, T - 1, 100))
gbm_sig = {}; lin_sig = {}
for cp in CHECKPOINTS:
    train = panel[panel["t"] < cp]
    test = panel[(panel["t"] >= cp) & (panel["t"] < cp + 100)]
    if len(test) == 0: continue
    gbm = GradientBoostingRegressor(n_estimators=80, max_depth=3, learning_rate=0.05,
                                     subsample=0.8, random_state=0).fit(train[FEATS], train["target"])
    lin = Ridge(alpha=1.0).fit(train[FEATS], train["target"])
    gp = gbm.predict(test[FEATS]); lp = lin.predict(test[FEATS])
    for (stk, t), gv, lv in zip(zip(test["stock"], test["t"]), gp, lp):
        gbm_sig[(stk, t)] = gv; lin_sig[(stk, t)] = lv

print("building position matrices (idio book: GBM-signal vs Ridge-signal, sign-based full $10k) ...")
POS_gbm = np.zeros((nInst, nt)); POS_lin = np.zeros((nInst, nt))
first_t = min(t for (_, t) in gbm_sig)
for j in range(1, nInst):
    for t, k in [(t, t + 1) for t in range(first_t, T - 1)]:   # position set at day index k=t+1 (matches prcSoFar length t+2... align with Praw's day index)
        pass

# day alignment: r[:, t] = logp[:, t+1]-logp[:, t]; a position informed by r[j,t] etc. is set for "day k"
# where prcSoFar has k+1 price columns, i.e. k = t+1 (the position earns r[j, t+1]... careful: our
# panel's features at row (j,t) predict target=r[j,t+1], i.e. the signal is KNOWN as of day t+1 (since
# it uses r[j,t] etc which is realized over day t->t+1) and should be TRADED on day k=t+1, earning r[j,t+1].
for j in range(1, nInst):
    for t in range(first_t, T - 1):
        k = t + 1
        cur = Praw[j, k]; lim = int(dlr[j] / cur)
        if (j, t) in gbm_sig:
            POS_gbm[j, k] = np.clip(np.sign(gbm_sig[(j, t)]) * (dlr[j] / cur), -lim, lim)
        if (j, t) in lin_sig:
            POS_lin[j, k] = np.clip(np.sign(lin_sig[(j, t)]) * (dlr[j] / cur), -lim, lim)

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))
FULL = (first_t + 2, nt)


def report(nm, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW); wf = window(POS, *FULL)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days if E - NUMTEST >= first_t]
    print(f"{nm:<12}FULL={wf['score']:>8.1f}  OLD={wo['score']:>8.1f}  NEW={wn['score']:>8.1f}  "
          f"rmean={np.mean(scs):>8.1f}  rfloor={min(scs):>8.1f}")


print()
report("GBM", POS_gbm)
report("Ridge", POS_lin)

# and for reference: shipped SAFE.py's idio book over the SAME window (different feature/model, apples-to-apples-ish context only)
idio_shipped = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = Praw[:, k]; lim = (dlr / cur).astype(int)
    full = np.asarray(SAFE.getMyPosition(Praw[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_shipped[:, k] = np.clip(p, -lim, lim).astype(int)
report("shipped SAFE idio", idio_shipped)
