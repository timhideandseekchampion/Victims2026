"""
test_batch100_B30_elasticnet.py

B30: Re-test Elastic Net against v10 using the exact current feature/target set (beta-adjusted Y).
Previously (`test_elasticnet.py`) rejected on a cheaper IC-only bar: best pooled IC (0.0527/0.0504)
never exceeded the ridge reference's own IC (0.0563) at any alpha/l1_ratio, using the OLD raw
next-day-return target and periodic checkpoints (no full traded-score harness was run). Re-checking
with the CURRENT beta-adjusted target Y = V10._beta_adjusted_target(...) -- a fair re-run rather than
assuming the old IC verdict still holds once the target itself changed.

APPROACH (screening-pass budget: ElasticNet requires one independent per-target fit x 50 targets x
4 half-lives, refit daily would be far too slow for coordinate descent inside a Python loop -- so,
matching this repo's own precedent for expensive per-day refits, PCA's `refit_freq` in
`test_q20_items01_04_ridge_variants.py`):
  STAGE 1 (cheap precheck, same IC-only bar as the original test, single representative half-life
  HL=500, periodic checkpoints every 30 days): sweep a small alpha x l1_ratio grid, compare best pooled
  IC against ridge's own IC on the SAME beta-adjusted target/checkpoints.
  STAGE 2 (only if Stage 1 looks non-trivially competitive): one full traded-score run of the best
  config from Stage 1, ensembled across all 4 half-lives, refit periodically (every 10 days, B/mx/my
  held fixed between refits -- the daily prediction still varies day to day via that day's own return
  vector) through the full v10 pipeline (BLEND reversion + boost + rank-stability + ALGO leg, all
  cached once, unaffected by the ridge estimator).

"Mechanism OFF" doesn't literally apply to a different model class -- the sanity check instead verifies
the shared REV/boost/rank-stability/ALGO cache reproduces v10 exactly when combined with the ORDINARY
daily ridge ensemble (same convention as the Kalman/RRR/Huber tests in this batch).
"""
import numpy as np, pandas as pd, time
from sklearn.linear_model import ElasticNet
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

# ==================================================================================================
# STAGE 1: cheap IC-only precheck, single half-life, periodic checkpoints (matches test_elasticnet.py)
# ==================================================================================================
HL_REP = 500
CHECKPOINTS = list(range(WARMUP, nt, 30))


def fit_enet_at(cp, alpha, l1_ratio):
    rr_ = r[:, :cp]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    n = X.shape[0]
    lam = 0.5 ** (1.0 / HL_REP)
    w = lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx
    B = np.zeros((nInst, nIdio))
    for j in range(nIdio):
        yc = Y[:, j] - my[j]
        m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, tol=1e-4)
        m.fit(Xc, yc, sample_weight=w)
        B[:, j] = m.coef_
    return B, mx, my


def fit_ridge_at(cp):
    rr_ = r[:, :cp]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    return V10._ewls_ridge(X, Y, HL_REP, RIDGE_A)


def pooled_ic(preds_by_day, actual_by_day):
    X = np.concatenate(preds_by_day); Y = np.concatenate(actual_by_day)
    ok = ~np.isnan(X) & ~np.isnan(Y)
    return float(np.corrcoef(X[ok], Y[ok])[0, 1])


print("=== STAGE 1: IC-only precheck, HL=500, beta-adjusted target, periodic checkpoints (every 30d) ===",
      flush=True)
t0 = time.time()
# "actual" for IC purposes is the raw next-day idio return (matches the original test's own convention
# -- what the forecast is ultimately trying to get the sign of right, whether or not the fit target
# was itself beta-adjusted)
ridge_preds = []; actuals = []
for i, cp in enumerate(CHECKPOINTS[:-1]):
    nxt = CHECKPOINTS[i + 1]
    Br, mxr, myr = fit_ridge_at(cp)
    for t in range(cp, min(nxt, nt - 1)):
        x = r[:, t]
        ridge_preds.append(myr + (x - mxr) @ Br)
        actuals.append(rs[:, t + 1])
ic_ridge = pooled_ic(ridge_preds, actuals)
print(f"  ridge(hl={HL_REP}, beta-adj target) IC on these days: {ic_ridge:.4f}  [{time.time()-t0:.0f}s]")

print("\n  sweeping ElasticNet alpha x l1_ratio ...")
print(f"  {'alpha':>10}{'l1_ratio':>10}{'IC':>9}{'avg_nonzero':>13}")
best_ic_cfg = None
t0 = time.time()
for alpha in (1e-5, 2e-5, 5e-5):
    for l1_ratio in (0.3, 0.5, 0.7):
        enet_preds = []; actuals2 = []; n_nonzero = 0; n_targets = 0
        for i, cp in enumerate(CHECKPOINTS[:-1]):
            nxt = CHECKPOINTS[i + 1]
            B, mx, my = fit_enet_at(cp, alpha, l1_ratio)
            n_nonzero += int((np.abs(B) > 1e-10).sum()); n_targets += nInst * nIdio
            for t in range(cp, min(nxt, nt - 1)):
                x = r[:, t]
                enet_preds.append(my + (x - mx) @ B); actuals2.append(rs[:, t + 1])
        ic = pooled_ic(enet_preds, actuals2)
        avg_nz = n_nonzero / n_targets
        print(f"  {alpha:>10}{l1_ratio:>10}{ic:>9.4f}{avg_nz:>13.3f}")
        if best_ic_cfg is None or ic > best_ic_cfg[0]:
            best_ic_cfg = (ic, alpha, l1_ratio)
print(f"  Stage 1 sweep done ({time.time()-t0:.0f}s)")
print(f"\n  best ElasticNet: IC={best_ic_cfg[0]:.4f} at alpha={best_ic_cfg[1]}, l1_ratio={best_ic_cfg[2]}")
print(f"  ridge reference: IC={ic_ridge:.4f}")
print(f"  ElasticNet best IC {'DOES' if best_ic_cfg[0] > ic_ridge else 'does NOT'} beat ridge's own IC "
      f"on the same beta-adjusted target.")

# ==================================================================================================
# STAGE 2: one full traded-score run of the best Stage-1 config, ensembled across all 4 half-lives,
# periodic refit (every 10 days) through the full v10 pipeline. Run regardless of the Stage-1 gate
# (cheap enough at this scale, and gives a real pass/fail number rather than an IC proxy).
# ==================================================================================================
print("\n=== precompute: BLEND reversion, pairwise boost, rank-stability signal, ALGO leg -- IDENTICAL "
      "regardless of the ridge estimator, cached once ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

RS_SIG = np.full((nIdio, nt), np.nan)
for t in days:
    rs_sig = V10._rank_stability_signal(logp[:, :t + 1])
    if rs_sig is not None:
        RS_SIG[:, t] = rs_sig
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def blend_final(wz, t):
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_pos_from_wz(wz_fn):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = blend_final(wz_fn(t), t)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def wz_ridge_ensemble(t):
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    return np.mean(fs, 0)


print("\n=== sanity check: plain ridge ensemble (mechanism 'off') must reproduce SAFE_llboost_v10 "
      "exactly -- validates the shared REV/boost/rank-stability/ALGO cache before testing ElasticNet ===")
POS_base = build_pos_from_wz(wz_ridge_ensemble)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")

REFIT_FREQ = 15
_, BEST_ALPHA, BEST_L1 = best_ic_cfg
print(f"\n=== building ElasticNet ensemble wz series (alpha={BEST_ALPHA}, l1_ratio={BEST_L1}, "
      f"all 4 half-lives, refit every {REFIT_FREQ} days) ===", flush=True)
t0 = time.time()
_enet_cache = {}


def fit_enet_hl_at(cp, hl):
    key = (cp, hl)
    if key in _enet_cache:
        return _enet_cache[key]
    rr_ = r[:, :cp]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    n = X.shape[0]
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(n - 1, -1, -1)
    sw = w.sum()
    mx = (w[:, None] * X).sum(0) / sw; my = (w[:, None] * Y).sum(0) / sw
    Xc = X - mx
    B = np.zeros((nInst, nIdio))
    for j in range(nIdio):
        yc = Y[:, j] - my[j]
        m = ElasticNet(alpha=BEST_ALPHA, l1_ratio=BEST_L1, max_iter=5000, tol=1e-4)
        m.fit(Xc, yc, sample_weight=w)
        B[:, j] = m.coef_
    _enet_cache[key] = (B, mx, my)
    return B, mx, my


ENET_WZ = {}
last_refit = -10**9
cur_fit = None
for t in days:
    if t - last_refit >= REFIT_FREQ or cur_fit is None:
        cur_fit = [fit_enet_hl_at(t, hl) for hl in HALF_LIVES]
        last_refit = t
    xq = r[:, t]
    fs = []
    for (B, mx, my) in cur_fit:
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    ENET_WZ[t] = np.mean(fs, 0)
print(f"  done ({time.time()-t0:.0f}s, {len(_enet_cache)} distinct (checkpoint, half-life) fits)",
      flush=True)


def evaluate(nm, verbose=True):
    Pz = build_pos_from_wz(lambda t: ENET_WZ[t]); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<40}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


print(f"\n=== STAGE 2 RESULT: ElasticNet(alpha={BEST_ALPHA}, l1_ratio={BEST_L1}), full pipeline ===")
result = evaluate(f"enet a={BEST_ALPHA} l1={BEST_L1} refit={REFIT_FREQ}d")
print(f"\n{'PASS' if result['passed'] else 'FAIL'}: ElasticNet {'beats' if result['passed'] else 'does not beat'} "
      f"v10 on OLD+NEW+rmean jointly.")
