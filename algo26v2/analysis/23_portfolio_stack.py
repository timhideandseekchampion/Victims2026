"""Portfolio stack: combine the top INDEPENDENT reversion edges and find the
ceiling. Shows (a) PnL correlation between edges, (b) greedy incremental
stacking by marginal score, (c) the best combined book + strict OOS check.

All positions summed then clipped to eval.py limits; scored by eval.py logic.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from common import (prices_array, COMM_DEFAULT, COMM_INST0, POSLIM_DEFAULT,
                    POSLIM_INST0, N_TEST_DAYS, section)

P, df, tickers = prices_array()
N, T = P.shape
LOGP = np.log(P)
commRate = np.full(N, COMM_DEFAULT); commRate[0] = COMM_INST0
dlrPosLimit = np.full(N, POSLIM_DEFAULT); dlrPosLimit[0] = POSLIM_INST0
IDX = {t: i for i, t in enumerate(tickers)}
ALGO = 0
FDR_PAIRS = [("AENO","NWIG"),("EORC","NGTE"),("HETT","ULXY"),("SMAH","ILVX"),
             ("HUXZ","ACAC"),("CTGI","EELT")]

# Johansen triplet vectors (trained on days 0..250, from module 22)
_train = P[:, :T - N_TEST_DAYS]
TRIPLETS = []
for (a, b), c in zip(FDR_PAIRS, ["RRES","EELT","MHRM","GARI","FCSG","GARI"]):
    ia, ib, ic = IDX[a], IDX[b], IDX[c]
    try:
        vec = coint_johansen(_train[[ia, ib, ic], :].T, 0, 1).evec[:, 0]
        TRIPLETS.append((ia, ib, ic, vec))
    except Exception:
        pass


def mn_book(sig, cur, dollars):
    sig = sig - sig.mean(); s = np.abs(sig).sum()
    return ((sig / s) * dollars * N / cur).astype(float) if s > 1e-12 else np.zeros(N)

# ---------------- edges (return $-share target vectors) ----------------
_kal = {}
def e_kalman(hist, entry=1.0, exit_z=0.5, dollars=8000, q=1e-4):
    n, t = hist.shape; pos = np.zeros(n)
    if t < 40: return pos
    cur = hist[:, -1]
    for a, b in FDR_PAIRS:
        ia, ib = IDX[a], IDX[b]
        x = LOGP[ib, :t]; y = LOGP[ia, :t]; beta = 1.0; Pv = 1.0; res = []
        for k in range(1, t):
            e = y[k] - beta * x[k]; Pv += q; K = Pv * x[k] / (x[k]*x[k]*Pv + 1.0)
            beta += K * e; Pv *= (1 - K * x[k]); res.append(e)
        res = np.array(res[-60:]); z = (res[-1] - res.mean()) / (res.std() + 1e-9)
        state = _kal.get((a, b), 0)
        if state == 0 and abs(z) > entry: state = -int(np.sign(z))
        elif state != 0 and abs(z) < exit_z: state = 0
        _kal[(a, b)] = state
        if state:
            pos[ia] += state * dollars / cur[ia]
            pos[ib] += -state * beta * dollars / cur[ib]
    return pos

def e_algo_timing(hist, h=5, dollars=40000):
    t = hist.shape[1]
    if t < h + 1: return np.zeros(N)
    r = np.log(hist[ALGO, -1] / hist[ALGO, -1 - h]); pos = np.zeros(N)
    pos[ALGO] = -np.sign(r) * dollars / hist[ALGO, -1]
    return pos

def e_leadlag(hist, lb=60, dollars=2500):
    n, t = hist.shape
    if t < lb + 3: return np.zeros(N)
    R = np.diff(LOGP[:, t-lb:t], axis=1); lastret = R[:, -1]
    A = R[:, 1:] - R[:, 1:].mean(1, keepdims=True)
    B = R[:, :-1] - R[:, :-1].mean(1, keepdims=True)
    xc = (A @ B.T) / (np.sqrt((A**2).sum(1)[:, None] * (B**2).sum(1)[None, :]) + 1e-12)
    np.fill_diagonal(xc, 0)
    sig = np.array([xc[i, np.argmax(np.abs(xc[i]))] * lastret[np.argmax(np.abs(xc[i]))] for i in range(n)])
    return mn_book(sig, hist[:, -1], dollars)

def e_corr_algo(hist, lb=90, entry=0.9, dollars=3500):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lb + 2: return pos
    cur = hist[:, -1]; la = LOGP[ALGO, t-lb:t]; leg = 0.0
    for i in range(1, n):
        beta = np.polyfit(la, LOGP[i, t-lb:t], 1)[0]
        resid = LOGP[i, :t] - beta * LOGP[ALGO, :t]
        w = resid[-lb:]; z = (resid[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            sh = -np.sign(z) * dollars / cur[i]; pos[i] += sh; leg += -sh * beta * cur[i] / cur[ALGO]
    pos[ALGO] += leg
    return pos

def e_xs_reversal(hist, h=10, dollars=3000):
    if hist.shape[1] < h + 1: return np.zeros(N)
    return mn_book(-np.log(hist[:, -1] / hist[:, -1 - h]), hist[:, -1], dollars)

def e_basket(hist, lb=90, entry=0.75, dollars=5000):
    n, t = hist.shape; pos = np.zeros(n)
    if t < lb + 2: return pos
    cur = hist[:, -1]
    for ia, ib, ic, vec in TRIPLETS:
        Y = hist[[ia, ib, ic], :].T; spread = Y @ vec
        w = spread[-lb:]; z = (spread[-1] - w.mean()) / (w.std() + 1e-9)
        if abs(z) > entry:
            v = vec / np.abs(vec).max()
            for leg, idx in zip(v, (ia, ib, ic)):
                pos[idx] += -np.sign(z) * leg * dollars / cur[idx]
    return pos

EDGES = {
    "Kalman pairs":   e_kalman,
    "ALGO timing 5d": e_algo_timing,
    "Lead-lag":       e_leadlag,
    "Corr-vs-ALGO":   e_corr_algo,
    "XS reversal 10d":e_xs_reversal,
    "Triplet basket": e_basket,
}


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
    return dict(mean=mu, sharpe=sr, score=score, dvol=totDVol, pnl=pll)


def reset():
    _kal.clear()

def make_portfolio(names, weights=None):
    weights = weights or {n: 1.0 for n in names}
    def gp(hist):
        tot = np.zeros(N)
        for n in names:
            tot += weights[n] * EDGES[n](hist)
        return tot
    return gp


section("23A. STANDALONE SCORE + DAILY-PnL CORRELATION BETWEEN EDGES")
pnls = {}
print(f"{'edge':<18}{'Sharpe':>8}{'Score':>9}")
for name, fn in EDGES.items():
    reset(); r = backtest(fn)
    pnls[name] = r["pnl"]
    print(f"{name:<18}{r['sharpe']:>8.2f}{r['score']:>9.2f}")
names = list(EDGES)
M = np.corrcoef(np.array([pnls[n] for n in names]))
print("\nDaily-PnL correlation matrix (low = diversifying):")
print("            " + "".join(f"{n[:8]:>9}" for n in names))
for i, n in enumerate(names):
    print(f"{n[:11]:<12}" + "".join(f"{M[i,j]:>9.2f}" for j in range(len(names))))

section("23B. GREEDY INCREMENTAL STACK (add the edge that most raises score)")
chosen = []; remaining = list(EDGES)
print(f"{'step':>4}  {'added edge':<18}{'combined Sharpe':>16}{'combined Score':>16}")
while remaining:
    best = (None, -1e9, None)
    for cand in remaining:
        reset(); r = backtest(make_portfolio(chosen + [cand]))
        if r["score"] > best[1]:
            best = (cand, r["score"], r["sharpe"])
    chosen.append(best[0]); remaining.remove(best[0])
    print(f"{len(chosen):>4}  +{best[0]:<17}{best[2]:>16.2f}{best[1]:>16.2f}")
print(f"\nOrder of marginal value: {' > '.join(chosen)}")

section("23C. BEST COMBINED BOOK — tune sub-weights (coarse)")
core = ["Kalman pairs", "ALGO timing 5d", "Lead-lag", "Corr-vs-ALGO", "Triplet basket"]
best = (None, -1e9)
for wk in (1.0, 1.5):
    for wa in (1.0, 2.0):
        for wl in (0.5, 1.0):
            for wc in (0.5, 1.0):
                for wb in (0.5, 1.0):
                    w = {"Kalman pairs":wk,"ALGO timing 5d":wa,"Lead-lag":wl,
                         "Corr-vs-ALGO":wc,"Triplet basket":wb}
                    reset(); r = backtest(make_portfolio(core, w))
                    if r["score"] > best[1]: best = (w, r["score"], r["sharpe"], r["mean"])
print(f"BEST combined: Sharpe {best[2]:.2f}  score {best[1]:.2f}  mean ${best[3]:.2f}")
print(f"  weights: {best[0]}")

section("23D. STRICT OUT-OF-SAMPLE (tune weights on 250-375, test 375-500)")
mid = T - 250; q3 = T - 125
best_oos = (None, -1e9)
for wk in (1.0, 1.5):
    for wa in (1.0, 2.0):
        for wb in (0.5, 1.0):
            w = {"Kalman pairs":wk,"ALGO timing 5d":wa,"Lead-lag":1.0,
                 "Corr-vs-ALGO":1.0,"Triplet basket":wb}
            reset(); r = backtest(make_portfolio(core, w), mid, q3)
            if r["score"] > best_oos[1]: best_oos = (w, r["score"])
reset(); ris = backtest(make_portfolio(core, best_oos[0]), mid, q3)
reset(); roos = backtest(make_portfolio(core, best_oos[0]), q3, T)
print(f"Tuned 250-375: Sharpe {ris['sharpe']:.2f} score {ris['score']:.2f}")
print(f"Held-out 375-500: Sharpe {roos['sharpe']:.2f} score {roos['score']:.2f}  <- honest ceiling")

section("VERDICT")
print("The combined ceiling and the diversification (23A corr matrix) show how much")
print("stacking adds beyond the best single edge. Weakly-correlated edges (ALGO")
print("timing, lead-lag) add PnL at little Sharpe cost; the pair engines dominate risk.")
