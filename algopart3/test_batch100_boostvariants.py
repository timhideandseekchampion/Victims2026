"""
test_batch100_boostvariants.py

Batch-100 ideas C49-C54: all six modify ONLY the pairwise-boost mechanism (_pairwise_boost),
leaving the idio ridge ensemble, BLEND reversion, rank-stability blend, and ALGO leg untouched.
Shares one expensive precompute (WZ_PRE = ridge ensemble w/ beta-adjusted target + BLEND reversion,
RS_SIG = raw rank-stability signal, algo_pos) across all six variants, exactly like
test_v19cand_boost_ncandidates.py's caching pattern -- only the boost array itself is recomputed
per variant/config.

C49: continuous boost-magnitude scaling by how far |corr| clears the Bonferroni threshold (not
     binary pass/fail) -- multiply lead_boost by clip(|corr|/thr, 1.0, CAP).
C50: restrict FOLLOWER eligibility too -- only the bottom-N (lowest vol) names may receive a boost;
     higher-vol names are forced to zero boost regardless of leader significance.
C51: time-decayed (EW) correlation for leader detection, replacing the flat equal-weighted corrmat.
C52: does a leader's OWN lag-2 return add incremental value on top of its already-used lag-1 return,
     for pairs where a significant lag-1 relationship already exists -- boost = lead_boost[-1] +
     W2*lead_boost[-2].
C53: leader selection via partial correlation, controlling for ALGO's contemporaneous same-day
     return (removed from both leader and follower before correlating).
C54: restrict the candidate pool by trailing REALIZED boost profitability (a panel-wide trailing IC
     of each candidate's own lead_boost against the AVERAGE next-day return of the other idio names)
     instead of pure vol-rank; candidates failing this are dropped before vol-ranking the rest.
"""
import numpy as np, pandas as pd, time
from scipy import stats
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
BOOST_ALPHA, BOOST_P, BOOST_SCALE_W, BOOST_IC_L = V10.BOOST_ALPHA, V10.BOOST_P, V10.BOOST_SCALE_W, V10.BOOST_IC_L
BOOST_N_CANDIDATES = V10.BOOST_N_CANDIDATES
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

print("=== precompute: ridge WZ (beta-adjusted target) + BLEND reversion + ALGO leg + raw "
      "rank-stability signal -- unaffected by any boost-mechanism variant tested here ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

print(f"  REV done ({time.time()-t0:.0f}s)", flush=True)

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  algo_pos done ({time.time()-t0:.0f}s)", flush=True)

WZ_PRE = np.full((nIdio, nt), np.nan)
for ii, t in enumerate(days):
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
    wz = np.mean(fs, 0)
    WZ_PRE[:, t] = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if ii % 200 == 0:
        print(f"    WZ_PRE day {t}/{nt} ({time.time()-t0:.0f}s)", flush=True)
print(f"  WZ_PRE done ({time.time()-t0:.0f}s)", flush=True)

RS_SIG = np.full((nIdio, nt), np.nan)
for t in days:
    if t < max(RS_SHORT_W, RS_LONG_W) + 5:
        continue
    short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
    long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        continue
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    RS_SIG[:, t] = np.where(disagree, -sz, 0.0)
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def sig_threshold(n_samples, n_candidates=BOOST_N_CANDIDATES):
    if n_samples < 10:
        return 1.0
    alpha_adj = BOOST_ALPHA / n_candidates
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


def build_pos_from_boost(BOOST):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s = RS_SIG[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, BOOST, base_wo=None, base_wn=None, base_scs=None, verbose=True):
    Pz = build_pos_from_boost(BOOST); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    if base_wo is None:
        passed = None
    else:
        passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum()) if base_scs is not None else None
    if verbose:
        tag = "  <== PASS" if passed else ("" if passed is None else "")
        print(f"  {nm:<32}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
              f"rfloor={scs.min():7.1f}  n_worse={nworse}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


# ============================================================================
# baseline boost (verbatim V10._pairwise_boost) -- sanity check
# ============================================================================
print("\n=== sanity check: baseline boost (V10._pairwise_boost verbatim) must reproduce v10 ===")
BOOST_BASE = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST_BASE[:, k] = V10._pairwise_boost(rs[:, :k])
base = evaluate("baseline (verbatim v10)", BOOST_BASE, verbose=False)
print(f"  baseline: OLD={base['wo']:.1f}  NEW={base['wn']:.1f}  rmean={base['rm']:.1f}  "
      f"rfloor={base['rf']:.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base['wo'] - 871.0) < 0.5 and abs(base['wn'] - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
base_wo, base_wn, base_scs = base['wo'], base['wn'], scs_curve(build_pos_from_boost(BOOST_BASE))


def report(nm, BOOST):
    return evaluate(nm, BOOST, base_wo, base_wn, base_scs)


# ============================================================================
# C49: continuous boost-magnitude scaling by |corr| strength vs threshold
# ============================================================================
def boost_c49(rs_k, cap):
    n, T = rs_k.shape
    boost = np.zeros(n)
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        factor = float(np.clip(abs(col[ci]) / thr, 1.0, cap))
        boost[j] = lead_boost[-1] * factor
    return boost


print("\n=== C49: continuous boost-magnitude scaling by |corr|/thr, capped at CAP "
      "(single-config screening choice: CAP=2.0, given compute budget) ===")
c49_results = []
for cap in [2.0]:
    t0 = time.time()
    B = np.zeros((nIdio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        B[:, k] = boost_c49(rs[:, :k], cap)
    c49_results.append(report(f"C49 cap={cap}", B))
    print(f"    [{time.time()-t0:.0f}s]", flush=True)


# ============================================================================
# C50: restrict FOLLOWER eligibility to bottom-N lowest-vol names
# ============================================================================
def boost_c50(rs_k, follower_n):
    n, T = rs_k.shape
    boost = np.zeros(n)
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    eligible_followers = set(np.argsort(vol_causal)[:follower_n].tolist())
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    for j in range(n):
        if j not in eligible_followers:
            continue
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


print("\n=== C50: restrict FOLLOWER eligibility to bottom-N lowest-vol names (of 50 idio) "
      "(single-config screening choice: N=20, bottom 40%, given compute budget) ===")
c50_results = []
for fn in [20]:
    t0 = time.time()
    B = np.zeros((nIdio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        B[:, k] = boost_c50(rs[:, :k], fn)
    c50_results.append(report(f"C50 follower_n={fn}", B))
    print(f"    [{time.time()-t0:.0f}s]", flush=True)


# ============================================================================
# C51: time-decayed (EW) correlation for leader detection
# ============================================================================
def corrmat_ew(X, Y, hl):
    n, T = X.shape
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(T - 1, -1, -1); sw = w.sum()
    mx = (X * w[None, :]).sum(1) / sw; my = (Y * w[None, :]).sum(1) / sw
    Xc = X - mx[:, None]; Yc = Y - my[:, None]
    vx = (w[None, :] * Xc * Xc).sum(1) / sw; vy = (w[None, :] * Yc * Yc).sum(1) / sw
    Xs = Xc / (np.sqrt(vx)[:, None] + 1e-12); Ys = Yc / (np.sqrt(vy)[:, None] + 1e-12)
    return (Xs * w[None, :]) @ Ys.T / sw


def boost_c51(rs_k, hl):
    n, T = rs_k.shape
    boost = np.zeros(n)
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = corrmat_ew(Xi, Yj, hl)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


print("\n=== C51: EW (time-decayed) correlation for leader detection, half-life HL "
      "(single-config screening choice: HL=500, given compute budget) ===")
c51_results = []
for hl in [500]:
    t0 = time.time()
    B = np.zeros((nIdio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        B[:, k] = boost_c51(rs[:, :k], hl)
    c51_results.append(report(f"C51 HL={hl}", B))
    print(f"    [{time.time()-t0:.0f}s]", flush=True)


# ============================================================================
# C52: leader's OWN lag-2 return incremental to lag-1, for already-significant pairs
# ============================================================================
def boost_c52(rs_k, w2):
    n, T = rs_k.shape
    boost = np.zeros(n)
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1] + w2 * lead_boost[-2]
    return boost


print("\n=== C52: add leader's own lag-2 return (lead_boost[-2]) w/ weight W2 on top of lag-1 "
      "(single-config screening choice: W2=0.5, given compute budget) ===")
c52_results = []
for w2 in [0.5]:
    t0 = time.time()
    B = np.zeros((nIdio, nt))
    for k in range(BOOST_MIN_DAY, nt):
        B[:, k] = boost_c52(rs[:, :k], w2)
    c52_results.append(report(f"C52 w2={w2}", B))
    print(f"    [{time.time()-t0:.0f}s]", flush=True)


# ============================================================================
# C53: leader selection via partial correlation controlling for ALGO's contemporaneous return
# ============================================================================
def partial_corrmat(X, Y, zx, zy):
    """X:(n,T) candidate leader returns at t; Y:(m,T) follower returns at t+1; zx:(T,) ALGO ret at
    t (aligned w/ X); zy:(T,) ALGO ret at t+1 (aligned w/ Y). Residualize each row of X on zx and
    each row of Y on zy (simple OLS on one control var), then corr the residuals."""
    def resid(A, z):
        zc = z - z.mean(); vz = (zc * zc).sum() + 1e-12
        Ac = A - A.mean(1, keepdims=True)
        b = (Ac @ zc) / vz
        return Ac - b[:, None] * zc[None, :]
    Xr = resid(X, zx); Yr = resid(Y, zy)
    Xs = Xr / (Xr.std(1, keepdims=True) + 1e-12); Ys = Yr / (Yr.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


def boost_c53(rs_k, algo_r_k):
    n, T = rs_k.shape
    boost = np.zeros(n)
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    thr = sig_threshold(n_samples)  # approx: partial-corr df loses 1 more dof, ignored for screening
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    zx = algo_r_k[:-1]  # ALGO ret at t, aligned with Xi_full/Xi (t = 0..T-2)
    zy = algo_r_k[1:]   # ALGO ret at t+1, aligned with Yj (t+1 = 1..T-1)
    C = partial_corrmat(Xi, Yj, zx, zy)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


print("\n=== C53: leader selection via partial correlation, controlling for ALGO contemporaneous ret ===")
algo_r = r[0]
t0 = time.time()
B = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    B[:, k] = boost_c53(rs[:, :k], algo_r[:k])
c53_result = report("C53 partial-corr", B)
print(f"    [{time.time()-t0:.0f}s]")


# ============================================================================
# C54: restrict candidate pool by trailing REALIZED panel-wide leadership profitability
# ============================================================================
def boost_c54(rs_k, trail_l):
    n, T = rs_k.shape
    boost = np.zeros(n)
    Xi_full = rs_k[:, :-1]; Yj = rs_k[:, 1:]
    n_samples = Xi_full.shape[1]
    vol_causal = np.nanstd(Xi_full, axis=1)
    # trailing panel-wide leadership profitability: for each name i, build its own lead_boost
    # series over the trailing window and correlate vs the AVERAGE next-day return of the OTHER
    # idio names (a proxy for "if i had been used as a leader broadly, would it have paid off").
    a0 = max(0, T - 1 - trail_l)
    mean_other = np.zeros(n)  # placeholder, filled per-i below (excl. self)
    panel_next = rs_k[:, 1:]  # (n, T-1), day t+1 return, t=0..T-2
    panel_mean_all = np.nanmean(panel_next, axis=0)  # (T-1,) mean across ALL names incl self
    eligible = np.zeros(n, dtype=bool)
    for i in range(n):
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lb = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        xs = lb[a0:T - 1]
        # exclude i's own contribution from the panel mean to avoid self-correlation inflation
        ys = (panel_mean_all[a0:] * n - rs_k[i, 1 + a0:T]) / (n - 1)
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            eligible[i] = False
            continue
        picv = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        eligible[i] = picv > 0
    pool = np.where(eligible)[0]
    if len(pool) == 0:
        return boost
    thr = sig_threshold(n_samples, n_candidates=max(len(pool), 1))
    ranked = pool[np.argsort(-vol_causal[pool])][:BOOST_N_CANDIDATES]
    cand_idx = ranked
    Xi = Xi_full[cand_idx]
    C = corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_k[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rs_k[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


print("\n=== C54: restrict candidate pool by trailing panel-wide leadership profitability "
      "(trail window = BOOST_IC_L) then vol-rank the survivors ===")
t0 = time.time()
B = np.zeros((nIdio, nt))
n_eligible_hist = []
for k in range(BOOST_MIN_DAY, nt):
    B[:, k] = boost_c54(rs[:, :k], BOOST_IC_L)
c54_result = report("C54 trailing-profit pool", B)
print(f"    [{time.time()-t0:.0f}s]")


# ============================================================================
# summary
# ============================================================================
print("\n" + "=" * 90)
print("SUMMARY (all vs v10 baseline OLD=%.1f NEW=%.1f rmean=%.1f)" % (base_wo, base_wn, base_scs.mean()))
print("=" * 90)
for c in c49_results + c50_results + c51_results + c52_results + [c53_result, c54_result]:
    tag = "PASS" if c["passed"] else "fail"
    print(f"  [{tag}] {c['name']:<32} OLD={c['wo']:7.1f} NEW={c['wn']:7.1f} rmean={c['rm']:7.1f} "
          f"rfloor={c['rf']:7.1f} n_worse={c['nworse']}/61")
