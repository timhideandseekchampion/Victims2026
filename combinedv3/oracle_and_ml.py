"""Break the prior properly. (1) ORACLE: perfect next-day foresight sized to the caps -> the
PHYSICAL score ceiling on 500-750 (is 700 even in range?). (2) A PROPERLY-BUILT nonlinear ML
predictor (gradient boosting), walk-forward, as a REAL backtest (not a broken IC calc) -> does a
better predictor than the ridge exist on this window? Compare to ridge (505)."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]
S, E = nt - 250, nt


def score(pll):
    mu, sd = pll.mean(), pll.std()
    if mu <= 0 or sd < 1e-10: return mu, 0.0
    sr = np.sqrt(250)*mu/sd; return mu*sr**2/(sr**2+1), sr


def run(posfn):
    cash = 0; cp = np.zeros(nInst); val = 0; cm = 0; pll = []
    for t in range(S, E+1):
        p = prc[:, :t]; cur = p[:, -1]
        npos = np.clip(posfn(t, p, cur), -(dlr/cur).astype(int), (dlr/cur).astype(int)).astype(int) if t < E else cp.copy()
        d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
        pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
        if t > S: pll.append(pl)
    return score(np.array(pll))


# ---------- (1) ORACLE: perfect next-day sign, sized to cap ----------
def oracle_mn(t, p, cur):                                  # market-neutral perfect foresight
    pos = np.zeros(nInst)
    if t >= E: return pos
    fwd = RET[1:, t-1]                                     # the move this position earns (oracle cheats)
    w = fwd - fwd.mean()
    pos[1:] = np.sign(w) * (10000/cur[1:])
    return pos
def oracle_dir(t, p, cur):                                 # directional perfect foresight
    pos = np.zeros(nInst)
    if t >= E: return pos
    pos[1:] = np.sign(RET[1:, t-1]) * (10000/cur[1:])
    pos[0] = np.sign(RET[0, t-1]) * (100000/cur[0])
    return pos

sc_mn, sh_mn = run(oracle_mn); sc_dir, sh_dir = run(oracle_dir)
print("PHYSICAL CEILING (perfect foresight):")
print(f"  oracle market-neutral: score {sc_mn:.0f} (Sharpe {sh_mn:.1f})")
print(f"  oracle directional:    score {sc_dir:.0f} (Sharpe {sh_dir:.1f})")
print(f"  => 700-800 is {'WITHIN' if sc_mn>800 else 'NOT within'} physical range for a good predictor.\n")


# ---------- (2) proper nonlinear ML predictor, walk-forward real backtest ----------
_ml = {"t": -999, "model": None}
def ml_pos(t, p, cur, refit=10):
    pos = np.zeros(nInst)
    if t < 120: return pos
    if t - _ml["t"] >= refit:                              # refit every `refit` days on pooled history
        Xtr, ytr = [], []
        for tau in range(60, t-2):                         # causal: target RET[:,tau+1] must be <= t-2
            feat = RET[:, tau]
            for i in range(50):
                Xtr.append(np.concatenate([feat, [i]])); ytr.append(RET[i+1, tau+1])
        Xtr = np.array(Xtr); ytr = np.array(ytr)
        m = HistGradientBoostingRegressor(max_iter=80, max_depth=3, learning_rate=0.08,
                                          min_samples_leaf=200).fit(Xtr, ytr)
        _ml["model"] = m; _ml["t"] = t
    feat = RET[:, t-2]                                     # last OBSERVED return (causal)
    Xq = np.array([np.concatenate([feat, [i]]) for i in range(50)])
    pred = _ml["model"].predict(Xq)
    w = pred - pred.mean()
    keep = np.abs(w) >= 0.2*(np.std(w)+1e-12)
    pos[1:] = np.where(keep, np.sign(w)*(10000/cur[1:]), 0.0)
    ret = np.log(p[:, 1:])-np.log(p[:, :-1]); rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
    betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
    cap = 100000/cur[0]; pos[0] = float(np.clip(-((pos[1:]*cur[1:])@betas)/cur[0], -cap, cap))
    return pos

print("PROPER nonlinear ML (gradient boosting, walk-forward real backtest):")
sc_ml, sh_ml = run(ml_pos)
print(f"  GBM book on 500-750: score {sc_ml:.0f} (Sharpe {sh_ml:.1f})   vs ridge 505")
print(f"  => nonlinear ML {'BEATS' if sc_ml>530 else 'does NOT beat'} the ridge/ceiling.")
