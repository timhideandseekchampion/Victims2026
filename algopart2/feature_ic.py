"""feature_ic.py — the RIGHT version of the detector idea: do OBSERVABLE features (computed from
prices/forecasts at decision time, no look-ahead) predict FUTURE lead-lag IC? Unlike lagged-IC
timing (which failed the persistence test), a price feature that forecasts IC weakness WOULD be a
usable detector. Test corr(feature_t, mean lead-lag IC over next W days)."""
import numpy as np
import stability as S
ridge_z = S.ridge_z; revz = S.revz; r_all = S.r_all; logp = S.logp; nDays = S.nDays; ENS = S.ENS
nInst = S.nInst

# realized daily lead-lag IC (same as detector.py)
IC = {}
for t in range(96, nDays):
    fll = np.mean([ridge_z(t, hl) for hl in ENS], 0)
    fwd = r_all[1:, t - 1]; fwd = fwd - fwd.mean()
    if fwd.std() > 1e-12:
        IC[t] = float(np.corrcoef(fll, fwd)[0, 1])

def ridge_raw_norm(t, hl, a=0.1):
    """dispersion (std) of the RAW (un-z-scored) lead-lag forecast = a 'confidence' proxy."""
    lp = logp[:, :t]; r = lp[:, 1:] - lp[:, :-1]
    X = r[:, :-1].T; Y = r[1:, 1:].T; xin = r[:, -1]
    n = X.shape[0]; lam = 0.5 ** (1 / hl); w = lam ** np.arange(n - 1, -1, -1); sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx; Yc = Y - my
    B = np.linalg.solve(Xc.T @ (w[:, None] * Xc) + a * np.eye(nInst), Xc.T @ (w[:, None] * Yc))
    f = my + (xin - mx) @ B
    return float((f - f.mean()).std())

def features(t):
    """all computed from returns realized through day t-1 (no look-ahead)."""
    r = r_all[:, :t - 1]                       # realized returns, last col = day t-1
    idx = r[0]; stk = r[1:]
    f = {}
    f["idxvol40"] = idx[-40:].std()
    f["stkvol40"] = stk[:, -40:].std(1).mean()
    f["xdisp10"] = stk[:, -10:].sum(1).std()                      # dispersion of 10-day stock returns
    # factor strength: avg |corr(stock, index)| over last 60d
    a = idx[-60:] - idx[-60:].mean()
    cs = []
    for i in range(stk.shape[0]):
        b = stk[i, -60:] - stk[i, -60:].mean()
        d = (a.std() * b.std())
        if d > 1e-12: cs.append(abs((a * b).mean() / d))
    f["factorcorr60"] = np.mean(cs)
    f["fnorm"] = ridge_raw_norm(t, 500)                           # forecast 'confidence'
    return f

W = 20
rows = {}
ts = [t for t in range(130, nDays - W) if all(s in IC for s in range(t, t + W))]
feat = {t: features(t) for t in ts}
fwdic = {t: np.mean([IC[s] for s in range(t, t + W)]) for t in ts}   # FUTURE mean IC
names = list(feat[ts[0]].keys())
print(f"predicting mean lead-lag IC over NEXT {W} days, n={len(ts)} decision days")
print(f"{'feature':<14}{'corr w/ future IC':>18}{'|t-stat|':>10}")
y = np.array([fwdic[t] for t in ts])
for nm in names:
    x = np.array([feat[t][nm] for t in ts])
    m = np.isfinite(x) & np.isfinite(y)
    c = np.corrcoef(x[m], y[m])[0, 1]
    tt = abs(c) * np.sqrt((m.sum() - 2) / max(1e-9, 1 - c * c))
    print(f"{nm:<14}{c:>18.3f}{tt:>10.1f}")
print("\n|corr| > ~0.2 with a stable sign = a usable detector signal; near 0 = no predictive content.")
print("NOTE: overlapping fwd windows inflate t-stats; treat |corr| as the honest effect size.")
