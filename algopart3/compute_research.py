"""Gather the research-findings dataset for research.html: the idio forecast IC by window
(is the edge stable?), the ML overfitting result (GBM in-sample vs out-of-sample), the lead-lag
mechanism (cross-name vs own-name coefficients + lagged-market persistence per name), and the
signal scoreboard (mechanism + OOS verdicts). Exports research_data.json."""
import json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
P = pd.read_csv("prices.txt", sep=r"\s+", header=0); names = list(P.columns); P = P.values.T.astype(float)
nInst, nt = P.shape; logp = np.log(P); R = np.full((nInst, nt), np.nan); R[:, 1:] = logp[:, 1:] - logp[:, :-1]
HL = (250, 500, 1000, 2000); RIDGE_A = 0.1; REV_W = 10

def ewls(X, Y, hl, a):
    n, p = X.shape; lam = 0.5 ** (1 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw; Xc, Yc = X - mx, Y - my
    XtWX = Xc.T @ (w[:, None] * Xc); XtWY = Xc.T @ (w[:, None] * Yc); eps = 1e-8 * np.trace(XtWX) / p
    return np.linalg.solve(XtWX + (eps + a) * np.eye(p), XtWY), mx, my

WLL = np.full((nInst, nt), np.nan); RV = np.full((nInst, nt), np.nan)
for t in range(131, nt):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]; fs = []
    for hl in HL:
        B, mx, my = ewls(r[:, :-1].T, r[1:, 1:].T, hl, RIDGE_A); pred = my + (r[:, -1] - mx) @ B
        fi = pred - pred.mean(); fs.append(fi / (fi.std() + 1e-12))
    WLL[1:, t - 1] = np.mean(fs, 0)
    rr = lp[1:, -1] - lp[1:, -1 - REV_W]; rr = rr - rr.mean(); RV[1:, t - 1] = -rr / (rr.std() + 1e-12)
NR = np.full((nInst, nt), np.nan); NR[:, :-1] = logp[:, 1:] - logp[:, :-1]

def pooled_ic(F, s, e):
    xs, ys = [], []
    for d in range(s, min(e, nt - 1)):
        m = ~np.isnan(F[1:, d]) & ~np.isnan(NR[1:, d]); xs.append(F[1:, d][m]); ys.append(NR[1:, d][m])
    x, y = np.concatenate(xs), np.concatenate(ys); return round(float(np.corrcoef(x, y)[0, 1]), 3)

wins = {"full": (131, nt), "H1": (131, 500), "H2": (501, nt), "OLD": (500, 750), "NEW": (750, nt)}
idio_ic = {lab: {w: pooled_ic(F, *rng) for w, rng in wins.items()}
           for lab, F in [("leadlag", WLL), ("reversion", RV), ("blend", 0.7 * WLL + 0.3 * RV)]}

# ---- ML overfitting: features, train H1, test H2 ----
VOL = np.full((nInst, nt), np.nan)
for t in range(20, nt): VOL[:, t] = np.nanstd(R[:, t - 19:t + 1], axis=1)
rows = []
for d in range(140, nt - 1):
    for i in range(1, nInst):
        if np.isnan(WLL[i, d]) or np.isnan(NR[i, d]): continue
        rows.append((d, WLL[i, d], RV[i, d], R[i, d], logp[i, d] - logp[i, d - 5], logp[i, d] - logp[i, d - 10],
                     logp[i, d] - logp[i, d - 20], R[0, d], VOL[i, d], (WLL[1:, d] < WLL[i, d]).mean(), NR[i, d]))
A = np.array(rows); d = A[:, 0]; feats = A[:, 1:10]; y = A[:, 10]; tr = d <= 500; te = d > 500
def ic(p, yy): return round(float(np.corrcoef(p, yy)[0, 1]), 3)
ml = [{"model": "linear ridge (current)", "is": ic(feats[tr, 0], y[tr]), "oos": ic(feats[te, 0], y[te])}]
lr = Ridge(alpha=1.0).fit(feats[tr], y[tr])
ml.append({"model": "linear on 9 feats", "is": ic(lr.predict(feats[tr]), y[tr]), "oos": ic(lr.predict(feats[te]), y[te])})
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_depth=3, l2_regularization=1.0,
                                   early_stopping=True, validation_fraction=0.2, random_state=0).fit(feats[tr], y[tr])
ml.append({"model": "GBM (gradient boosting)", "is": ic(gb.predict(feats[tr]), y[tr]), "oos": ic(gb.predict(feats[te]), y[te])})

# ---- lead-lag mechanism ----
def scorr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else float("nan")
mkt = R[1:].mean(0); X = mkt[:-1]; Y = R[1:, 1:]; s = 480
h1 = [scorr(X[:s], Y[i, :s]) for i in range(Y.shape[0])]
h2 = [scorr(X[s:], Y[i, s:]) for i in range(Y.shape[0])]
lam = 0.5 ** (1 / 1000); r = R[:, 1:700]; Xr = r[:, :-1].T; Yr = r[1:, 1:].T; n = Xr.shape[0]; w = lam ** np.arange(n - 1, -1, -1)
mx = (w[:, None] * Xr).sum(0) / w.sum(); my = (w[:, None] * Yr).sum(0) / w.sum(); Xc, Yc = Xr - mx, Yr - my
XtWX = Xc.T @ (w[:, None] * Xc); B = np.linalg.solve(XtWX + (1e-8 * np.trace(XtWX) / 51 + 0.1) * np.eye(51), Xc.T @ (w[:, None] * Yc))
own = float(np.mean([abs(B[c + 1, c]) for c in range(49)]))
cross = float(np.mean([abs(B[rr2, c]) for c in range(49) for rr2 in range(51) if rr2 != c + 1]))
cross_share = round(100 * cross * 50 / (cross * 50 + own), 0)

scoreboard = [
    {"sig": "idio lead-lag", "mech": "cross-sectional info diffusion", "oos": "0.06 every window", "ok": 1},
    {"sig": "idio reversion (blend)", "mech": "orthogonal diversifier (corr -0.04)", "oos": "stable, small", "ok": 1},
    {"sig": "ALGO vol→return", "mech": "vol risk-premium (index-specific)", "oos": "p<0.001, fragile", "ok": 2},
    {"sig": "ALGO momentum / EMA", "mech": "none stable", "oos": "flips OOS", "ok": 0},
    {"sig": "ALGO breakout (EMA×z)", "mech": "none", "oos": "max-stat p=0.71", "ok": 0},
    {"sig": "support/resistance zones", "mech": "none", "oos": "flips H1↔H2", "ok": 0},
    {"sig": "ensemble of weak signals", "mech": "redundant, not orthogonal", "oos": "OLS overfits", "ok": 0},
    {"sig": "GBM on idio", "mech": "none (capacity)", "oos": "IS 0.24 → OOS 0.00", "ok": 0}]

out = {"names": names, "idio_ic": idio_ic, "ml": ml,
       "mech": {"h1": [round(x, 4) for x in h1], "h2": [round(x, 4) for x in h2],
                "lag_pos": int(sum(1 for a, b in zip(h1, h2) if not np.isnan(a) and (a + b) / 2 > 0)),
                "persist_corr": round(scorr(h1, h2), 2),
                "own": round(own, 3), "cross": round(cross, 3), "cross_share": cross_share},
       "scoreboard": scoreboard, "windows_note": "H1=days 131-500, H2=501-1000, OLD=501-750, NEW=751-1000"}
json.dump(out, open("research_data.json", "w"))
print("wrote research_data.json")
print("idio blend IC:", idio_ic["blend"])
print("ML:", [(m["model"], m["is"], m["oos"]) for m in ml])
print(f"mechanism: cross-name share {cross_share}%, lagged-market persist corr {out['mech']['persist_corr']}")
