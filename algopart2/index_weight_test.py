"""index_weight_test.py — does a WEIGHTED aggregation of the per-stock lead-lag signal predict
ALGO better than the equal-weighted vote (frac)?

ALGO is a weighted index of the 50 stocks, so stock i's lead-lag view should count in proportion to
  (its index weight w_i)  x  (its expected move ~ its volatility vol_i).

Step 0: recover index weights by regressing ALGO return on the 50 stock returns (report R^2 -> is
        ALGO really a linear combo?). Step 1: build candidate ALGO signals and measure IC on ALGO's
        next-day return, causally, per window:
  A frac      = mean(sign(wz))                      # current: equal-weighted vote
  B wvote     = sum w_i * sign(wz_i)                # index-weighted vote
  C wvolvote  = sum w_i * vol_i * sign(wz_i)        # index x vol weighted vote  (the proposal)
  D wraw      = sum w_i * pred_i                    # index-weighted RAW forecast (model magnitude)
  E common    = mean(pred_i)                        # equal-weight raw mean (reference; was -ve IC)
"""
import numpy as np, pandas as pd
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
logp = np.log(prc); r_all = logp[:, 1:] - logp[:, :-1]
ENS = [250, 500, 1000, 2000]; WREG = 250; VOLW = 60; RA_W = 1e-6

# ---- Step 0: is ALGO a weighted index of the 50 stocks? (global OLS, in-sample R^2) ----
R = r_all[1:].T; y = r_all[0]
w_glob, *_ = np.linalg.lstsq(R, y, rcond=None)
resid = y - R @ w_glob
r2 = 1 - resid.var() / y.var()
print(f"[premise] ALGO ~ 50 stocks  global R^2 = {r2:.4f}   weights: sum={w_glob.sum():.3f}  "
      f"min={w_glob.min():.3f}  max={w_glob.max():.3f}  #neg={(w_glob<0).sum()}")

def weights(t):
    """causal index weights: regress ALGO return on stock returns over trailing WREG days."""
    s = max(0, t-1-WREG)
    Rw = r_all[1:, s:t-1].T; yw = r_all[0, s:t-1]
    w, *_ = np.linalg.lstsq(Rw, yw, rcond=None)
    return w

def sigs(t, blend=0.3):
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]; n = X.shape[0]
    zs = []; raws = []
    for hl in ENS:
        lam = 0.5**(1/hl); w = lam**np.arange(n-1, -1, -1); sw = w.sum()
        mx = (w[:, None]*X).sum(0)/sw; my = (w[:, None]*Y).sum(0)/sw; Xc = X-mx; Yc = Y-my
        B = np.linalg.solve(Xc.T@(w[:, None]*Xc)+0.1*np.eye(nInst), Xc.T@(w[:, None]*Yc))
        f = my+(xin-mx)@B; raws.append(f); d = f-f.mean(); zs.append(d/(d.std()+1e-12))
    z = np.mean(zs, 0); pred = np.mean(raws, 0)                      # z-scored & raw per-stock forecast
    rr = logp[1:, t-1]-logp[1:, t-1-10]; rr = rr-rr.mean(); rv = -rr/(rr.std()+1e-12)
    wz = (1-blend)*z + blend*rv                                     # traded per-stock signal
    s = np.sign(wz)
    w = weights(t); w = w / (np.abs(w).sum() + 1e-12)              # normalise index weights
    vol = r_all[1:, t-1-VOLW:t-1].std(1)                          # trailing stock vol
    return {"A": np.mean(s), "B": float(w@s), "C": float((w*vol)@s),
            "D": float(w@pred), "E": float(pred.mean())}

WINDOWS = {"500-750": (500, 749), "400-500": (400, 499), "250-400": (250, 399)}
labels = {"A":"frac (equal vote)","B":"wvote (idx)","C":"wvolvote (idx*vol)","D":"wraw (idx*pred)","E":"common (eq raw)"}
print(f"\n{'variant':<22}" + "".join(f"{wl:>12}" for wl in WINDOWS))
rows = {k: {} for k in labels}
for wl, (S, E) in WINDOWS.items():
    acc = {k: [] for k in labels}; ys = []
    for t in range(S, E):
        v = sigs(t); [acc[k].append(v[k]) for k in labels]; ys.append(float(r_all[0, t]))
    ys = np.array(ys)
    for k in labels: rows[k][wl] = np.corrcoef(np.array(acc[k]), ys)[0, 1]
for k in labels:
    print(f"{labels[k]:<22}" + "".join(f"{rows[k][wl]:>+12.3f}" for wl in WINDOWS))
print("\n(IC on ALGO next-day return; higher +ve = better directional signal for the ALGO leg)")
