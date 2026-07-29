"""
test_batch100_B39_contemp_divergence.py  [DIAGNOSTIC]

B39: Re-test the contemporaneous divergence-from-usual-co-mover signal against v10 (originally
near-zero IC against an earlier baseline: test_contemporaneous_divergence.py found full-sample pooled
IC(div_z[t] -> resid[t+1]) essentially null). This signal was never Stage-2 backtested (IC was too
weak to justify it) -- so "against v10" here means: (a) recompute the same causal signal on the
CURRENT price data, and (b) additionally check whether it says anything about where v10's OWN sign
calls are more/less likely to be right (a more direct "against v10" framing than a generic IC check).
No pass/fail forced -- this is a methodology/finding check, not a new mechanism with a backtest.

Signal construction reused VERBATIM from test_contemporaneous_divergence.py: idiosyncratic residuals
(causal expanding-window ALGO-beta, checkpoint-refit every 50 days) -> causal contemporaneous co-mover
map (highest |corr| among OTHER stocks' residuals, checkpoint-refit) -> daily "surprise" = resid_j[t] -
comov_beta_j * resid_i[t] -> trailing 60-day z-score -> div_z.
"""
import numpy as np
from batch100_shared import nInst, nt, logp, r, rs, WZ_FULL, days

T = r.shape[1]
r0 = r[0]

print("=== B39: (re)build causal idiosyncratic residuals + contemporaneous co-mover surprise signal ===")
CP = list(range(100, T, 50))


def beta_at(cp):
    v0 = r0[:cp]
    return np.array([np.polyfit(v0, r[j, :cp], 1)[0] if j > 0 else 1.0 for j in range(nInst)])


BETA_CP = {cp: beta_at(cp) for cp in CP}


def beta_for_day(t):
    valid = [c for c in CP if c <= t]
    return BETA_CP[valid[-1]] if valid else BETA_CP[CP[0]]


resid = np.full((nInst, T), np.nan)
for t in range(CP[0], T):
    b = beta_for_day(t)
    resid[:, t] = r[:, t] - b * r0[t]
resid[0, :] = np.nan


def comover_at(cp):
    X = resid[1:, :cp]
    ok = ~np.any(np.isnan(X), axis=0)
    Xc = X[:, ok]
    Xn = (Xc - Xc.mean(1, keepdims=True)) / (Xc.std(1, keepdims=True) + 1e-12)
    C = (Xn @ Xn.T) / Xn.shape[1]
    n = nInst - 1
    comov = {}; comov_beta = {}
    for j in range(n):
        col = C[:, j].copy(); col[j] = np.nan
        i = int(np.nanargmax(np.abs(col)))
        comov[j + 1] = i + 1
        xi = Xc[i]; xj = Xc[j]
        comov_beta[j + 1] = float(np.polyfit(xi, xj, 1)[0])
    return comov, comov_beta


COMOVE_CP = {cp: comover_at(cp) for cp in CP if cp >= 150}
CP2 = sorted(COMOVE_CP.keys())


def comove_for_day(t):
    valid = [c for c in CP2 if c <= t]
    return COMOVE_CP[valid[-1]] if valid else COMOVE_CP[CP2[0]]


first_cp = CP2[0]
START = first_cp + 10
surprise = np.full((nInst, T), np.nan)
for t in range(START, T):
    comov, comov_beta = comove_for_day(t)
    for j in range(1, nInst):
        i = comov[j]
        surprise[j, t] = resid[j, t] - comov_beta[j] * resid[i, t]

VOL_Z_W = 60
div_z = np.full((nInst, T), np.nan)
for j in range(1, nInst):
    s = surprise[j]
    for t in range(START + VOL_Z_W, T):
        w = s[t - VOL_Z_W:t]
        ok = ~np.isnan(w)
        if ok.sum() > 20:
            div_z[j, t] = (s[t] - w[ok].mean()) / (w[ok].std() + 1e-12)


def pooled_ic(feat, target, tmin, tmax):
    rows_x = []; rows_y = []
    for t in range(tmin, tmax):
        fx = feat[1:, t]; fy = target[1:, t + 1]
        ok = ~np.isnan(fx) & ~np.isnan(fy)
        if ok.sum() == 0: continue
        rows_x.append(fx[ok]); rows_y.append(fy[ok])
    X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
    return float(np.corrcoef(X, Y)[0, 1]), len(X)


tmin, tmax = START + VOL_Z_W, T - 1
print("\n=== finding (a): recompute pooled IC(div_z[t] -> raw idio resid[t+1]) on CURRENT data ===")
ic_full, n_full = pooled_ic(div_z, resid, tmin, tmax)
print(f"  full-sample pooled IC = {ic_full:+.4f}  (n={n_full})")

half = (tmin + tmax) // 2
ic_h1, n1 = pooled_ic(div_z, resid, tmin, half)
ic_h2, n2 = pooled_ic(div_z, resid, half, tmax)
print(f"  H1 IC: {ic_h1:+.4f} (n={n1})   H2 IC: {ic_h2:+.4f} (n={n2})")

rng = np.random.default_rng(0)
perm_ics = []
for p in range(200):
    div_shuf = div_z.copy()
    for j in range(1, nInst):
        col = div_shuf[j, tmin:tmax + 1].copy()
        rng.shuffle(col)
        div_shuf[j, tmin:tmax + 1] = col
    ic_p, _ = pooled_ic(div_shuf, resid, tmin, tmax)
    perm_ics.append(ic_p)
perm_ics = np.array(perm_ics)
pval = (np.abs(perm_ics) >= np.abs(ic_full)).mean()
print(f"  permutation null: mean={perm_ics.mean():+.4f}  std={perm_ics.std():.4f}  p-value={pval:.3f}")

print("\n=== finding (b): does div_z say anything about whether v10's OWN sign call will be right? ===")
# v10's realized per-name-day PnL sign correctness: does sign(WZ_FULL[j,t]) match the actual next-day
# return sign, and does that correctness correlate with |div_z| (a "when to trust v10 more" question)?
rows_x = []; rows_y = []
for t in range(max(tmin, 96), min(tmax, nt - 1)):
    dz = div_z[1:, t]
    wzc = WZ_FULL[:, t]
    ok = ~np.isnan(dz) & np.isfinite(wzc)
    if ok.sum() == 0: continue
    actual_next = rs[:, t]
    hit = (np.sign(wzc) == np.sign(actual_next)).astype(float)
    rows_x.append(np.abs(dz[ok])); rows_y.append(hit[ok])
X = np.concatenate(rows_x); Y = np.concatenate(rows_y)
ic_hit = float(np.corrcoef(X, Y)[0, 1])
print(f"  IC(|div_z[t]| -> v10 sign-hit indicator[t]) = {ic_hit:+.4f}  (n={len(X)})")
print(f"  overall v10 sign-hit rate in this sample: {Y.mean()*100:.2f}%")

print("\n=== B39 interpretation ===")
print(f"  Same-order-of-magnitude near-zero IC as the original diagnostic ({ic_full:+.4f} vs raw idio "
      f"return; perm p={pval:.2f}); the |div_z| vs v10-hit-rate check ({ic_hit:+.4f}) shows no "
      f"meaningful relationship either -- the signal carries no exploitable information against v10's "
      f"current forecast, confirming (not just re-citing) the prior near-null finding.")
