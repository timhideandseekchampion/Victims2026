"""New angle: does the significance-gated pairwise boost work better if leader selection and boost
sizing are based on ALGO-BETA RESIDUALS (idiosyncratic returns, with each stock's market exposure
to ALGO removed) instead of RAW returns? Currently SAFE_llboost.py's corrmat/argmax runs on raw
returns, which could partly be picking up shared ALGO co-movement rather than genuine idiosyncratic
lead-lag -- residualizing first should isolate the latter. The TARGET (what we're actually trying
to predict/trade) stays the raw next-day return throughout, since that's what a position earns PnL
on; only the LEADER SELECTION and boost magnitude are computed from residuals.

Stage 1: compare the boost signal's own predictive IC (same alignment validated in
test_h1_reciprocal_pairs.py: feature from day k-1 predicts rs[j,k]) between the raw-return-based
population (current shipped) and the residual-based population, with permutation + H1/H2 for each.
"""
import numpy as np, pandas as pd, time
from scipy import stats

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
logp = np.log(P)
r = np.diff(logp, axis=1)
r0 = r[0]
rs = r[1:]  # raw idio returns, (49, T)
n, T = rs.shape

BOOST_MIN_DAY = 500
ALPHA = 0.05
N_CANDIDATES = 49
BOOST_P = 2.0
BOOST_SCALE_W = 1000


def sig_threshold(n_samples):
    if n_samples < 10: return 1.0
    alpha_adj = ALPHA / N_CANDIDATES
    tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
    return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))


def corrmat(X, Y):
    Xc = X - X.mean(1, keepdims=True); Yc = Y - Y.mean(1, keepdims=True)
    Xs = Xc / (Xc.std(1, keepdims=True) + 1e-12); Ys = Yc / (Yc.std(1, keepdims=True) + 1e-12)
    return (Xs @ Ys.T) / X.shape[1]


print("=== building causal, checkpoint-refit ALGO-beta residuals ===")
BETA_CP = list(range(100, T, 50))


def beta_at(cp):
    v0 = r0[:cp]
    return np.array([np.polyfit(v0, rs[j, :cp], 1)[0] for j in range(n)])


BETA_AT = {cp: beta_at(cp) for cp in BETA_CP}


def beta_for_day(t):
    valid = [c for c in BETA_CP if c <= t]
    return BETA_AT[valid[-1]] if valid else BETA_AT[BETA_CP[0]]


resid = np.full((n, T), np.nan)
for t in range(BETA_CP[0], T):
    b = beta_for_day(t)
    resid[:, t] = rs[:, t] - b * r0[t]
print(f"done. mean|beta| at final checkpoint: {np.abs(BETA_AT[BETA_CP[-1]]).mean():.3f}")


def compute_boost_population(source_mat, label):
    """source_mat: (49, T) array used for BOTH leader-selection (corrmat) and boost magnitude.
    Target is always the raw rs[j, k] (matching production's validated alignment)."""
    rows = []
    t0 = time.time()
    for k in range(BOOST_MIN_DAY, min(nt, T)):
        Tn = k
        Xi = source_mat[:, :Tn - 1]; Yj = source_mat[:, 1:Tn]
        ok_cols = ~np.any(np.isnan(Xi), axis=0) & ~np.any(np.isnan(Yj), axis=0)
        if ok_cols.sum() < 60:
            continue
        Xi_c = Xi[:, ok_cols]; Yj_c = Yj[:, ok_cols]
        n_samples = Xi_c.shape[1]
        thr = sig_threshold(n_samples)
        C = corrmat(Xi_c, Yj_c)
        for j in range(n):
            col = C[:, j].copy(); col[j] = np.nan
            i = int(np.nanargmax(np.abs(col)))
            if abs(col[i]) <= thr:
                continue
            lead = source_mat[i, :Tn]
            ok_lead = ~np.isnan(lead)
            if ok_lead[-1000:].sum() < 100:
                continue
            valid_lead = lead[max(0, Tn - 1 - BOOST_SCALE_W):Tn - 1]
            valid_lead = valid_lead[~np.isnan(valid_lead)]
            if len(valid_lead) < 60:
                continue
            scale = np.nanstd(valid_lead) + 1e-12
            lv = lead[-1]
            if np.isnan(lv):
                continue
            boost_val = float(np.sign(lv) * (np.abs(lv) / scale) ** BOOST_P)
            rows.append((k, j, boost_val))
    print(f"  {label}: {len(rows)} significant-pair-day instances ({time.time()-t0:.0f}s)")
    return rows


print("\n=== computing RAW-return-based population (current shipped mechanism) ===")
rows_raw = compute_boost_population(rs, "raw")

print("=== computing RESIDUAL-based population (new angle) ===")
rows_resid = compute_boost_population(resid, "residual")


def pooled_ic_perm(rows, label, n_perm=300):
    if len(rows) < 30:
        print(f"{label}: too few, skipping"); return
    ks = np.array([x[0] for x in rows])
    bv = np.array([x[2] for x in rows])
    target = np.array([rs[x[1], x[0]] for x in rows])
    ic = float(np.corrcoef(bv, target)[0, 1])
    med_k = np.median(ks)
    m1 = ks < med_k; m2 = ~m1
    ic1 = float(np.corrcoef(bv[m1], target[m1])[0, 1]) if m1.sum() > 20 else float('nan')
    ic2 = float(np.corrcoef(bv[m2], target[m2])[0, 1]) if m2.sum() > 20 else float('nan')
    rng = np.random.default_rng(0)
    perm_ics = np.empty(n_perm)
    for p in range(n_perm):
        perm_ics[p] = np.corrcoef(bv, rng.permutation(target))[0, 1]
    pval = float((np.abs(perm_ics) >= abs(ic)).mean())
    print(f"{label}: IC={ic:+.4f} (n={len(rows)})  H1={ic1:+.4f}  H2={ic2:+.4f}  perm p={pval:.3f}")


print("\n=== Stage 1 comparison ===")
pooled_ic_perm(rows_raw, "raw-return-based (current shipped)")
pooled_ic_perm(rows_resid, "residual-based (new)")
