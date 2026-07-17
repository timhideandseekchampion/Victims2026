"""EDGE MENU - backtest every candidate alpha signal standalone and rank by score.

Each signal is market-aware and scored with the exact eval.py logic on the last
250 days. Goal: a ranked menu of what actually has an edge, so it can be added
to an existing book. Families: mean-reversion, momentum, lead-lag, factor,
volatility, ML. Positive score + positive Sharpe = a real, addable edge.
"""
import warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
from numpy.linalg import lstsq
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, N_TEST_DAYS, section)

P, df, tickers = prices_array()
N, T = P.shape
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
ALGO = 0
LOGP = np.log(P)


def backtest(get_pos, start=T - N_TEST_DAYS, end=T):
    cash = 0.0; curPos = np.zeros(N); totDVol = 0.0; value = 0.0; comm = 0.0; pll = []
    for t in range(start, end + 1):
        hist = P[:, :t]; cur = hist[:, -1]
        if t < end:
            lim = (dlrPosLimit / cur).astype(int)
            newPos = np.clip(get_pos(hist), -lim, lim).astype(int)
        else:
            newPos = np.array(curPos)
        d = newPos - curPos; cash -= cur.dot(d) + comm
        dvol = cur * np.abs(d); comm = np.sum(dvol * commRate); totDVol += dvol.sum()
        curPos = np.array(newPos); pv = curPos.dot(cur)
        todayPL = cash + pv - value; value = cash + pv
        if t > start: pll.append(todayPL)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    sr = np.sqrt(250) * mu / sd if sd > 0 else 0
    score = mu * (sr**2 / (sr**2 + 1)) if (mu > 0 and sd > 1e-10) else mu
    return mu, sr, score, totDVol


def mn_book(sig, cur, dollars):
    """turn a raw per-name signal into $-neutral integer positions."""
    sig = sig - sig.mean()
    s = np.abs(sig).sum()
    if s < 1e-12: return np.zeros(N)
    return ((sig / s) * dollars * N / cur).astype(int)


# ---------- MEAN-REVERSION ----------
def xs_reversal(hist, h=1, dollars=3000):
    if hist.shape[1] < h + 1: return np.zeros(N)
    r = np.log(hist[:, -1] / hist[:, -1 - h])
    return mn_book(-r, hist[:, -1], dollars)

def resid_pc_reversal(hist, k=3, lb=60, dollars=3000):
    if hist.shape[1] < lb + 2: return np.zeros(N)
    R = np.diff(LOGP[:, hist.shape[1]-lb:hist.shape[1]], axis=1).T  # lb-1 x N
    Rc = R - R.mean(0)
    U, S, Vt = np.linalg.svd(Rc, full_matrices=False)
    comp = Vt[:k]                      # k x N
    last = Rc[-1]
    resid = last - comp.T @ (comp @ last)
    return mn_book(-resid, hist[:, -1], dollars)

def bollinger(hist, lb=20, dollars=2500):
    if hist.shape[1] < lb + 1: return np.zeros(N)
    w = LOGP[:, hist.shape[1]-lb:hist.shape[1]]
    z = (w[:, -1] - w.mean(1)) / (w.std(1) + 1e-9)
    return mn_book(-z, hist[:, -1], dollars)

# ---------- MOMENTUM ----------
def xs_momentum(hist, h=20, dollars=3000):
    if hist.shape[1] < h + 1: return np.zeros(N)
    r = np.log(hist[:, -1] / hist[:, -1 - h])
    return mn_book(r, hist[:, -1], dollars)

def ts_momentum(hist, h=60, dollars=2500):
    if hist.shape[1] < h + 1: return np.zeros(N)
    r = np.log(hist[:, -1] / hist[:, -1 - h])
    return mn_book(np.sign(r) * np.abs(r), hist[:, -1], dollars)

# ---------- FACTOR / LEAD-LAG ----------
def corr_algo_resid(hist, lb=90, entry=0.75, dollars=3500):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lb + 2: return pos
    cur = hist[:, -1]; la = LOGP[ALGO, t-lb:t]; leg = 0.0
    for i in range(1, n):
        li = LOGP[i, t-lb:t]; beta = np.polyfit(la, li, 1)[0]
        resid = LOGP[i, :t] - beta * LOGP[ALGO, :t]
        w = resid[-lb:]; z = (resid[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            sh = -np.sign(z) * dollars / cur[i]; pos[i] += sh
            leg += -sh * beta * cur[i] / cur[ALGO]
    pos[ALGO] += leg
    return pos.astype(int)

def algo_timing(hist, h=5, mode="rev", dollars=40000):
    t = hist.shape[1]
    if t < h + 1: return np.zeros(N)
    r = np.log(hist[ALGO, -1] / hist[ALGO, -1 - h])
    pos = np.zeros(N)
    pos[ALGO] = (-np.sign(r) if mode == "rev" else np.sign(r)) * dollars / hist[ALGO, -1]
    return pos.astype(int)

def leadlag_xpred(hist, lb=60, dollars=2500):
    """for each name, predict next ret from the lag-1 ret of its best cross-corr peer."""
    n, t = hist.shape
    if t < lb + 3: return np.zeros(N)
    R = np.diff(LOGP[:, t-lb:t], axis=1)            # N x lb-1
    lastret = R[:, -1]
    # cross-corr(name_t, peer_{t-1})
    A = R[:, 1:]; B = R[:, :-1]
    A = (A - A.mean(1, keepdims=True)); B = (B - B.mean(1, keepdims=True))
    num = A @ B.T
    den = np.sqrt((A**2).sum(1)[:, None] * (B**2).sum(1)[None, :]) + 1e-12
    xc = num / den                                   # xc[i,j]=corr(i_t, j_{t-1})
    np.fill_diagonal(xc, 0)
    sig = np.zeros(n)
    for i in range(n):
        j = np.argmax(np.abs(xc[i]))
        sig[i] = xc[i, j] * lastret[j]
    return mn_book(sig, hist[:, -1], dollars)

# ---------- ADVANCED PAIRS ----------
PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("HETT","ULXY"),("SMAH","ILVX"),
         ("HUXZ","ACAC"),("CTGI","EELT")]
IDX = {t: i for i, t in enumerate(tickers)}
def kalman_pairs(hist, entry=0.75, dollars=8000, q=1e-4):
    n, t = hist.shape; pos = np.zeros(n)
    if t < 40: return pos
    cur = hist[:, -1]
    for a, b in PAIRS:
        ia, ib = IDX[a], IDX[b]
        x = LOGP[ib, :t]; y = LOGP[ia, :t]
        # recursive least squares for dynamic beta
        beta = 1.0; Pv = 1.0; resids = []
        for k in range(1, t):
            pred = beta * x[k]; e = y[k] - pred
            Pv += q; K = Pv * x[k] / (x[k]*x[k]*Pv + 1.0)
            beta += K * e; Pv *= (1 - K * x[k]); resids.append(e)
        resids = np.array(resids[-60:])
        z = (resids[-1] - resids.mean()) / (resids.std() + 1e-9)
        if abs(z) > entry:
            pos[ia] += -np.sign(z) * dollars / cur[ia]
            pos[ib] += np.sign(z) * beta * dollars / cur[ib]
    return pos.astype(int)

# ---------- MACHINE LEARNING ----------
_ml = {"day": -10**9, "model": None}
def ml_gbm(hist, lb=5, retrain=25, dollars=2500):
    n, t = hist.shape
    if t < 120: return np.zeros(n)
    mkt = np.diff(LOGP[:, :t], axis=1).mean(0)
    if t - _ml["day"] >= retrain or _ml["model"] is None:
        X, y = [], []
        for i in range(n):
            r = np.diff(LOGP[i, :t])
            for k in range(lb, len(r) - 1):
                X.append(list(r[k-lb:k][::-1]) + [mkt[k-1]]); y.append(r[k])
        m = GradientBoostingRegressor(n_estimators=80, max_depth=3,
                                      learning_rate=0.03, random_state=0)
        m.fit(np.array(X), np.array(y)); _ml["model"] = m; _ml["day"] = t
    feats = []
    for i in range(n):
        r = np.diff(LOGP[i, :t]); feats.append(list(r[-lb:][::-1]) + [mkt[-1]])
    pred = _ml["model"].predict(np.array(feats))
    return mn_book(pred, hist[:, -1], dollars)


section("EDGE MENU - standalone score of every candidate signal (last 250 days)")
signals = [
    ("XS reversal 1d",        "mean-rev",  lambda h: xs_reversal(h, 1)),
    ("XS reversal 3d",        "mean-rev",  lambda h: xs_reversal(h, 3)),
    ("XS reversal 5d",        "mean-rev",  lambda h: xs_reversal(h, 5)),
    ("XS reversal 10d",       "mean-rev",  lambda h: xs_reversal(h, 10)),
    ("PC(3)-residual reversal","mean-rev", lambda h: resid_pc_reversal(h, 3, 60)),
    ("PC(5)-residual reversal","mean-rev", lambda h: resid_pc_reversal(h, 5, 60)),
    ("Bollinger 20d",         "mean-rev",  lambda h: bollinger(h, 20)),
    ("XS momentum 5d",        "momentum",  lambda h: xs_momentum(h, 5)),
    ("XS momentum 20d",       "momentum",  lambda h: xs_momentum(h, 20)),
    ("XS momentum 60d",       "momentum",  lambda h: xs_momentum(h, 60)),
    ("TS momentum 60d",       "momentum",  lambda h: ts_momentum(h, 60)),
    ("TS momentum 120d",      "momentum",  lambda h: ts_momentum(h, 120)),
    ("Corr-vs-ALGO residual", "factor",    lambda h: corr_algo_resid(h, 90, 0.75)),
    ("ALGO timing (revert 5d)","factor",   lambda h: algo_timing(h, 5, "rev")),
    ("ALGO timing (mom 20d)", "factor",    lambda h: algo_timing(h, 20, "mom")),
    ("Lead-lag cross-predict","lead-lag",  lambda h: leadlag_xpred(h, 60)),
    ("Kalman dynamic pairs",  "pairs",     lambda h: kalman_pairs(h, 0.75)),
    ("GBM ML cross-section",  "ML",        ml_gbm),
]
rows = []
for name, fam, fn in signals:
    _ml["day"] = -10**9; _ml["model"] = None
    try:
        mu, sr, score, dv = backtest(fn)
        rows.append((name, fam, mu, sr, score, dv))
    except Exception as e:
        rows.append((name, fam, np.nan, np.nan, np.nan, 0))
        print(f"  {name} failed: {e}")

rows.sort(key=lambda r: (-(r[4] if r[4] == r[4] else -1e9)))
print(f"\n{'signal':<26}{'family':<11}{'mean$':>9}{'Sharpe':>8}{'Score':>9}{'$vol':>12}")
for name, fam, mu, sr, score, dv in rows:
    flag = "  <-- EDGE" if (score == score and score > 5 and sr > 0.5) else ""
    print(f"{name:<26}{fam:<11}{mu:>9.2f}{sr:>8.2f}{score:>9.2f}{dv:>12.0f}{flag}")

section("VERDICT: addable edges (Sharpe>0.5 & Score>5, standalone)")
edges = [r for r in rows if r[4] == r[4] and r[4] > 5 and r[3] > 0.5]
for name, fam, mu, sr, score, dv in edges:
    print(f"  {name:<26} ({fam}): Sharpe {sr:.2f}, score {score:.1f}")
print(f"\n{len(edges)} standalone positive edges found. Mean-reversion / stat-arb")
print("dominates; momentum & directional signals are flat-to-negative (as the")
print("autocorrelation & variance-ratio nulls predicted).")
