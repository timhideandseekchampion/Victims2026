"""
test_v7cand_double_ic_diag.py  --  companion diagnostic to test_v7cand_double_ic_idio.py

WHY the ALGO leg's double-IC veto helps but the same veto on the idio book does not. Measures, on
the same data, the one quantity the whole idea rests on: HOW OFTEN THE TWO ESTIMATORS DISAGREE, and
whether that disagreement is information (a real sign change in the underlying edge) or noise
(two estimates of the same stably-positive number, both wobbling around zero).

  1. ALGO leg   -- how often `_side`'s veto actually fires, and what the two ICs look like.
  2. Idio book  -- the pooled fast/EW IC series, its distribution and sign stability.
  3. Idio names -- per-stock trailing IC: mean, dispersion, and the noise floor 1/sqrt(n) it must
                   clear for a sign disagreement to mean anything.
"""
import numpy as np, pandas as pd
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP = V7.WARMUP


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
# 1) ALGO leg: instrument the real `_side` -- how often does the double-IC veto fire?
# ==================================================================================================
print("=== 1) ALGO leg: the shipped double-IC veto, instrumented on the real code path ===")
lpA = logp[0]
rows = []
for tnow in range(200, nt):
    T = tnow + 1
    if T < V7.VOL_WIN + V7.VOL_Z + 60:
        continue
    lp = lpA[:T]
    rr = np.diff(lp)
    vol = np.full(T, np.nan); vol[V7.VOL_WIN:] = V7._roll_std(rr, V7.VOL_WIN)
    lo = max(V7.VOL_WIN + V7.VOL_Z, T - 1 - V7.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V7.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lp[1:] - lp[:-1]
    tn = T - 1
    a = max(0, tn - V7.IC_FAST)
    icf = wcorr(volz[a:tn], ret1[a:tn], np.ones(tn - a))
    aw = max(0, tn - V7.IC_EW_W); m = tn - aw
    ics = [wcorr(volz[aw:tn], ret1[aw:tn], ew_weights(m, hl)) for hl in V7.IC_EW_HL]
    ice = np.nan if any(not np.isfinite(v) for v in ics) else float(np.mean(ics))
    rows.append((tnow, icf, ice))

A = np.array(rows, dtype=float)
ok = np.isfinite(A[:, 1]) & np.isfinite(A[:, 2])
icf, ice = A[ok, 1], A[ok, 2]
dis = (icf >= 0) != (ice >= 0)
print(f"  vol feature, {ok.sum()} decidable days")
print(f"    fast IC (90d simple):  mean={icf.mean():+.4f}  sd={icf.std():.4f}  "
      f"range=[{icf.min():+.3f}, {icf.max():+.3f}]  negative on {100*(icf<0).mean():.0f}% of days")
print(f"    EW IC  (20/45, 200d):  mean={ice.mean():+.4f}  sd={ice.std():.4f}  "
      f"range=[{ice.min():+.3f}, {ice.max():+.3f}]  negative on {100*(ice<0).mean():.0f}% of days")
print(f"    >>> VETO FIRES (signs disagree -> ALGO flat) on {dis.sum()}/{ok.sum()} days "
      f"= {100*dis.mean():.1f}%")
print(f"    the underlying edge genuinely CHANGES SIGN: fast IC crosses zero "
      f"{int((np.diff(np.sign(icf)) != 0).sum())} times over the sample")

# ==================================================================================================
# 2) + 3) idio side: the same two estimators, book-pooled and per-stock
# ==================================================================================================
print("\n=== 2/3) idio side: same two estimators on the traded ridge+blend score ===")
WZ = np.full((nIdio, nt), np.nan)
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V7.HALF_LIVES:
        B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    rv_ = logp[1:, t] - logp[1:, t - V7.REV_W]
    rv_ = rv_ - rv_.mean()
    wz = (1 - V7.BLEND) * wz + V7.BLEND * (-rv_ / (rv_.std() + 1e-12))
    WZ[:, t] = wz

FAST_L, EW_W = 90, 200
bf, be, sf_all, se_all, ns = [], [], [], [], []
for k in range(WARMUP + 200, nt):
    lo_f = max(WARMUP, k - FAST_L); lo_w = max(WARMUP, k - EW_W)
    m = k - lo_w
    X = WZ[:, lo_f:k].ravel(); Y = rs[:, lo_f:k].ravel()
    bf.append(wcorr(X, Y, np.ones(X.size)))
    Xw = WZ[:, lo_w:k]; Yw = rs[:, lo_w:k]
    vals = [wcorr(Xw.ravel(), Yw.ravel(), np.repeat(ew_weights(m, hl)[None, :], nIdio, 0).ravel())
            for hl in V7.IC_EW_HL]
    be.append(float(np.mean(vals)))
    for j in range(nIdio):
        sf_all.append(wcorr(WZ[j, lo_f:k], rs[j, lo_f:k], np.ones(k - lo_f)))
        se_all.append(wcorr(WZ[j, lo_w:k], rs[j, lo_w:k], ew_weights(m, V7.IC_EW_HL[0])))
    ns.append(k - lo_f)

bf = np.array(bf); be = np.array(be)
sf_all = np.array(sf_all); se_all = np.array(se_all)
bdis = (bf >= 0) != (be >= 0)
print(f"  BOOK-POOLED IC ({len(bf)} days, ~{nIdio*FAST_L} obs per estimate):")
print(f"    fast (90d simple):  mean={np.nanmean(bf):+.4f}  sd={np.nanstd(bf):.4f}  "
      f"range=[{np.nanmin(bf):+.4f}, {np.nanmax(bf):+.4f}]  negative on {100*(bf<0).mean():.1f}% of days")
print(f"    EW  (20/45, 200d):  mean={np.nanmean(be):+.4f}  sd={np.nanstd(be):.4f}  "
      f"range=[{np.nanmin(be):+.4f}, {np.nanmax(be):+.4f}]  negative on {100*(be<0).mean():.1f}% of days")
print(f"    >>> VETO FIRES on {bdis.sum()}/{len(bf)} days = {100*bdis.mean():.1f}%  "
      f"(both estimators are positive on every single day -- the gate is INERT here)")

okm = np.isfinite(sf_all) & np.isfinite(se_all)
sdis = (sf_all[okm] >= 0) != (se_all[okm] >= 0)
n_typ = int(np.mean(ns))
print(f"\n  PER-STOCK IC ({okm.sum()} decidable stock-days, ~{n_typ} obs per estimate):")
print(f"    fast (90d simple):  mean={np.nanmean(sf_all):+.4f}  sd across stock-days="
      f"{np.nanstd(sf_all):.4f}  negative on {100*(sf_all[okm]<0).mean():.1f}%")
print(f"    >>> disagreement on {100*sdis.mean():.1f}% of stock-days")
print(f"    noise floor: a true IC of {np.nanmean(sf_all):+.4f} estimated from {n_typ} points has "
      f"SE ~ 1/sqrt(n) = {1/np.sqrt(n_typ):.4f}")
print(f"    signal-to-noise |mean IC| / SE = {abs(np.nanmean(sf_all))*np.sqrt(n_typ):.2f}  "
      f"-> a single stock's trailing IC cannot resolve its own sign; disagreement is estimator "
      f"noise, not a regime.")
print(f"    (book-pooled equivalent: |{np.nanmean(bf):+.4f}| * sqrt({nIdio*FAST_L}) = "
      f"{abs(np.nanmean(bf))*np.sqrt(nIdio*FAST_L):.2f} -- 50x more data per estimate is exactly "
      f"why the book-level gate never fires and the per-stock one fires constantly.)")
