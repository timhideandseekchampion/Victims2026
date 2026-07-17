"""Fine-grained IC maximization. Search HL x ALPHA x predictor-set x intercept to raise the
cross-sectional IC (the edge measure) above the ridge's 0.074. Score = IC-driven, so higher IC on
BOTH windows = higher expected score everywhere. Report configs ranked by the WORSE of the two
windows (robust), then backtest the winner to confirm score."""
import itertools
import numpy as np, pandas as pd

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]           # (51, nt-1)


def fit_forecast(d, hl, a, pred_set, use_int):
    """Forecast the 50 names' return for day d (predict RET[:,d-1]) using data through RET[:,d-2]."""
    # training design: predict RET[1:, tau+1] from predictors at tau, tau in [.., d-2]
    Yidx = np.arange(1, d-1)                               # target columns (<= d-2)
    if len(Yidx) < 40: return None
    Xall = RET[:, Yidx-1].T                                # predictors at tau  (n,51)
    Y = RET[1:, Yidx].T                                    # targets  (n,50)
    if pred_set == "full": Xp = Xall
    elif pred_set == "no_algo": Xp = Xall[:, 1:]           # drop ALGO from predictors
    xin_full = RET[:, d-2]
    xin = xin_full if pred_set == "full" else xin_full[1:]
    n, p = Xp.shape
    lam = 0.5 ** (1.0/hl); w = lam ** np.arange(n-1, -1, -1); sw = w.sum()
    mx = (w[:, None]*Xp).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw
    Xc = Xp - mx; Yc = Y - my
    XtWX = Xc.T@(w[:, None]*Xc); XtWY = Xc.T@(w[:, None]*Yc); eps = 1e-8*np.trace(XtWX)/p
    B = np.linalg.solve(XtWX + (eps+a)*np.eye(p), XtWY)
    pred = (0.0 if not use_int else my) + (xin - mx) @ B
    return pred - pred.mean()


def mean_ic(hl, a, pred_set, use_int, S, E, step=2):
    ics = []
    for d in range(S, E, step):
        f = fit_forecast(d, hl, a, pred_set, use_int)
        if f is None: continue
        fwd = RET[1:, d-1]
        if f.std() > 0 and fwd.std() > 0:
            ics.append(np.corrcoef(f, fwd)[0, 1])
    return np.mean(ics) if ics else 0.0


grid = dict(hl=[250, 500, 750, 1000, 1500, 3000], a=[0.03, 0.1, 0.3],
            pred=["full", "no_algo"], use_int=[True, False])
oldS, oldE = 250, 500
newS, newE = nt-250, nt
res = []
for hl, a, ps, ui in itertools.product(grid["hl"], grid["a"], grid["pred"], grid["use_int"]):
    ic_o = mean_ic(hl, a, ps, ui, oldS, oldE)
    ic_n = mean_ic(hl, a, ps, ui, newS, newE)
    res.append((min(ic_o, ic_n), ic_o, ic_n, dict(hl=hl, a=a, pred=ps, use_int=ui)))
res.sort(key=lambda x: -x[0])
print("baseline ridge (HL500,a0.1,full,intercept): reference IC ~0.068 / 0.074\n")
print(f"{'IC_old':>8} {'IC_new':>8} {'min':>8}  config")
for mn, io, inw, cfg in res[:15]:
    print(f"{io:8.4f} {inw:8.4f} {mn:8.4f}  {cfg}")
best = res[0][3]
print(f"\nBEST by worse-window IC: {best}  (IC_old {res[0][1]:.4f}, IC_new {res[0][2]:.4f})")
