"""
test_algo_ic_regime_drivers.py

Follow-up to the formal stationarity check: ALGO's trailing vol->next-return IC is confirmed
non-stationary (KPSS rejects stationarity, ADF can't reject a unit root, 22% of days negative, 13
zero-crossings). Before proposing new signals or adaptive machinery, find out WHAT ACTUALLY DRIVES
the sign changes -- is this a real time-varying regime with an identifiable conditioning variable
(which could be traded on LEADINGLY, ahead of the reactive trailing-IC estimate that `_side` uses
today), or is it just estimation noise around a weak/secularly-strengthening true effect (in which
case there's nothing to "figure out", `_side`'s reactive gate is already close to the best available
response, and the repo's own README framing -- "GARCH-in-mean risk premium, IC +0.02->+0.11->+0.14,
strengthening every sub-period" -- is the more accurate story)?

Four independent checks, all using only information available causally up to each day (no leakage),
plus one deliberately NON-causal full-sample GARCH-M fit used only to characterize the DGP, not to
trade:

  1. TIME TREND: is the negative/flip pattern concentrated early (consistent with "weak effect that
     strengthens over the file") or spread uniformly across the whole 1000 days (consistent with a
     genuine recurring regime)?
  2. VOL-LEVEL CONDITIONING: does the instantaneous vol->return relationship (volz[t] * ret1[t], the
     day-by-day contribution to the trailing IC) depend on the LEVEL of realized vol at the time --
     i.e. is this a "risk premium that only shows up when vol is already elevated" pattern (a
     recognized asymmetry in the vol risk-premium literature), rather than a constant-sign effect
     contaminated by noise?
  3. TREND-REGIME CONDITIONING: does it depend on whether the index itself is trending up, trending
     down, or flat over the same window (since a trend can mechanically correlate with vol clustering
     in this one-factor DGP)?
  4. FULL-SAMPLE GARCH(1,1)-in-Mean FIT: directly tests the README's own conjecture. If the risk-
     premium coefficient (lambda on conditional variance in the mean equation) is significant and its
     SIGN is what the trailing IC has been chasing, that's a real, structural, single-parameter
     explanation for the whole "non-stationarity" -- and predicts the effect should be present (with
     the same sign) throughout, just estimated noisily by a short trailing window.
"""
import numpy as np, pandas as pd
from scipy import stats as sstats
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
logp = np.log(P_)
r = np.diff(logp, axis=1)
lpA = logp[0]
retA = r[0]


def ew_weights(m, hl):
    return (0.5 ** (1.0 / hl)) ** np.arange(m - 1, -1, -1)


def wcorr(x, y, w, min_n=60):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n: return np.nan
    x, y, w = x[ok], y[ok], w[ok]; sw = w.sum()
    mx = (w * x).sum() / sw; my = (w * y).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw; vy = (w * (y - my) ** 2).sum() / sw
    if vx < 1e-24 or vy < 1e-24: return np.nan
    return float((w * (x - mx) * (y - my)).sum() / sw / np.sqrt(vx * vy))


# ==================================================================================================
# shared build: volz[t] (causal, matches V7._algo_vol_shares exactly) and ret1[t] = next-day return
# ==================================================================================================
print("=== building causal volz series (identical construction to V7._algo_vol_shares) ===")
T = len(lpA)
rr = np.diff(lpA)
vol = np.full(T, np.nan); vol[V7.VOL_WIN:] = V7._roll_std(rr, V7.VOL_WIN)
volz = np.full(T, np.nan)
for tnow in range(V7.VOL_WIN + V7.VOL_Z, T):
    lo = max(V7.VOL_WIN + V7.VOL_Z, tnow - V7.IC_LOOKBACK)
    wv = vol[lo:tnow]     # NOTE: matches _algo_vol_shares' per-day call, which recomputes the whole
    # array fresh each call; here volz[tnow] uses vol[tnow-VOL_Z:tnow] exactly as the real code does
for s in range(V7.VOL_WIN + V7.VOL_Z, T):
    wv = vol[s - V7.VOL_Z:s]
    volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]   # ret1[t] = return realized AFTER day t

contrib = volz * ret1     # day t's instantaneous contribution to the vol->next-return relationship
valid = np.isfinite(contrib)
print(f"  {valid.sum()} valid days")

# trailing IC series (for reference / cross-check against the earlier diagnostic)
IC_FAST = V7.IC_FAST
algo_ic = np.full(T, np.nan)
for tnow in range(200, T):
    a = max(0, tnow - IC_FAST)
    algo_ic[tnow] = wcorr(volz[a:tnow], ret1[a:tnow], np.ones(tnow - a))

# ==================================================================================================
print("\n" + "=" * 96)
print("1) TIME TREND: is the effect concentrated early, or a recurring regime throughout?")
print("=" * 96)
QUARTS = [(0, 250), (250, 500), (500, 750), (750, 1000)]
for lo, hi in QUARTS:
    c = contrib[lo:hi][valid[lo:hi]]
    ic = np.array([x for x in algo_ic[lo:hi] if np.isfinite(x)])
    print(f"  days {lo:4d}-{hi:4d}: mean(volz*ret1)={c.mean():+.6f}  sd={c.std():.6f}  "
          f"frac(contrib>0)={100*(c>0).mean():.1f}%   trailing-IC mean over period={ic.mean() if len(ic) else float('nan'):+.4f}")

# split flip days (algo_ic crosses zero) by decade to see if flips cluster early or are spread out
signs = np.sign(algo_ic[np.isfinite(algo_ic)])
days_ic = np.where(np.isfinite(algo_ic))[0]
crossings = days_ic[1:][np.diff(signs) != 0]
print(f"\n  {len(crossings)} zero-crossings total; by quartile of the file:")
for lo, hi in QUARTS:
    n = int(((crossings >= lo) & (crossings < hi)).sum())
    print(f"    days {lo:4d}-{hi:4d}: {n} crossings")
neg_days = days_ic[algo_ic[days_ic] < 0]
print(f"\n  {len(neg_days)} negative-IC days total; by quartile:")
for lo, hi in QUARTS:
    n = int(((neg_days >= lo) & (neg_days < hi)).sum())
    tot = int(((days_ic >= lo) & (days_ic < hi)).sum())
    print(f"    days {lo:4d}-{hi:4d}: {n}/{tot} negative ({100*n/max(tot,1):.0f}%)")

# ==================================================================================================
print("\n" + "=" * 96)
print("2) VOL-LEVEL CONDITIONING: does the sign of volz*ret1 depend on the LEVEL of realized vol?")
print("=" * 96)
vol_level = vol.copy()   # raw realized 20d vol, not its z-score
med = np.nanmedian(vol_level[valid])
hi_vol = valid & (vol_level > med)
lo_vol = valid & (vol_level <= med)
print(f"  median realized 20d vol = {med:.5f}")
for nm, mask in (("HIGH-vol days (above median)", hi_vol), ("LOW-vol days (at/below median)", lo_vol)):
    c = contrib[mask]
    r_ic = wcorr(volz[mask], ret1[mask], np.ones(mask.sum()))
    print(f"  {nm:<32} n={mask.sum():4d}  mean(volz*ret1)={c.mean():+.6f}  "
          f"frac>0={100*(c>0).mean():.1f}%  static IC={r_ic:+.4f}")

# tertile split for a finer look
terts = np.nanpercentile(vol_level[valid], [33.3, 66.7])
for lo_t, hi_t, nm in ((None, terts[0], "LOW tertile"), (terts[0], terts[1], "MID tertile"),
                        (terts[1], None, "HIGH tertile")):
    m = valid.copy()
    if lo_t is not None: m &= vol_level > lo_t
    if hi_t is not None: m &= vol_level <= hi_t
    r_ic = wcorr(volz[m], ret1[m], np.ones(m.sum()))
    print(f"    {nm:<12} n={m.sum():4d}  static IC={r_ic:+.4f}  frac(contrib>0)={100*(contrib[m]>0).mean():.1f}%")

# ==================================================================================================
print("\n" + "=" * 96)
print("3) TREND-REGIME CONDITIONING: does it depend on the index's own trailing trend direction?")
print("=" * 96)
TREND_W = 60
trend = np.full(T, np.nan)
trend[TREND_W:] = lpA[TREND_W:] - lpA[:-TREND_W]     # trailing log-return over the last 60 days
up = valid & np.isfinite(trend) & (trend > 0)
down = valid & np.isfinite(trend) & (trend <= 0)
for nm, mask in (("UPTREND (60d trailing return > 0)", up), ("DOWNTREND (60d trailing return <= 0)", down)):
    r_ic = wcorr(volz[mask], ret1[mask], np.ones(mask.sum()))
    print(f"  {nm:<38} n={mask.sum():4d}  static IC={r_ic:+.4f}  frac(contrib>0)={100*(contrib[mask]>0).mean():.1f}%")

# cross-tab: vol level x trend direction (4 cells) -- looking for an interaction
print("\n  cross-tab (vol level x trend direction):")
for vnm, vmask in (("HIGH-vol", hi_vol), ("LOW-vol", lo_vol)):
    for tnm, tmask in (("UP", up), ("DOWN", down)):
        m = vmask & tmask
        if m.sum() < 30: continue
        r_ic = wcorr(volz[m], ret1[m], np.ones(m.sum()))
        print(f"    {vnm} x {tnm:<5} n={m.sum():4d}  static IC={r_ic:+.4f}")

# ==================================================================================================
print("\n" + "=" * 96)
print("4) FULL-SAMPLE GARCH(1,1)-in-Mean fit (non-causal, DGP characterization only -- not traded)")
print("=" * 96)
try:
    from arch import arch_model
    ret_pct = retA * 100.0   # arch expects returns in percent for numerical stability
    am = arch_model(ret_pct, mean="ARX", lags=0, vol="GARCH", p=1, q=1, dist="normal")
    # arch's ARX mean with an exogenous conditional-variance regressor requires a 2-step approach:
    # fit plain GARCH(1,1) first to get conditional variance, then regress return on lagged cond.var.
    res0 = am.fit(disp="off")
    cvar = res0.conditional_volatility ** 2
    print(res0.summary().tables[1])
    print("\n  GARCH(1,1) fit done. Now testing the IN-MEAN hypothesis directly:")
    print("  regress ret[t] on cvar[t-1] (lagged conditional variance -> risk-premium test)")
    x = cvar[:-1].values; y = ret_pct[1:].values
    X = np.column_stack([np.ones_like(x), x])
    beta, res_, rank_, sv_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y); k = 2
    sigma2 = (resid @ resid) / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    tstat = beta / se
    pval = 2 * (1 - sstats.t.cdf(np.abs(tstat), df=n - k))
    print(f"  lambda (risk-premium coefficient on lagged cond. variance): {beta[1]:+.4f}  "
          f"t={tstat[1]:+.3f}  p={pval[1]:.4g}")
    print(f"  {'SIGNIFICANT positive risk premium -- structural, single-parameter explanation' if (pval[1] < 0.05 and beta[1] > 0) else 'not a clean significant positive risk premium at full-sample level'}")

    # rolling GARCH-M lambda over sub-periods, to see if the coefficient itself is stable or drifting
    print("\n  lambda estimated separately per quartile (is the 'true' coefficient itself stable?):")
    for lo, hi in QUARTS:
        if hi > len(x): hi = len(x)
        xs, ys = x[lo:hi], y[lo:hi]
        if len(xs) < 60: continue
        Xq = np.column_stack([np.ones_like(xs), xs])
        bq, _, _, _ = np.linalg.lstsq(Xq, ys, rcond=None)
        rq = ys - Xq @ bq
        seq = np.sqrt(((rq @ rq) / (len(ys) - 2)) * np.linalg.inv(Xq.T @ Xq)[1, 1])
        print(f"    days {lo:4d}-{hi:4d}: lambda={bq[1]:+.4f}  t={bq[1]/seq:+.3f}")
except Exception as e:
    print(f"  GARCH-M fit failed: {e}")

print("\n" + "=" * 96)
print("VERDICT to be written up from the numbers above.")
print("=" * 96)
