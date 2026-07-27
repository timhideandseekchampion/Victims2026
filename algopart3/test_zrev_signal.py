"""Ad-hoc test: does a z-scored mean-reversion signal on ALGO's own return work as a
signal, using the exact same causal-IC / persistence / surrogate methodology that
compute_signals.py used to validate the vol-level signal in SAFE_llvol.py?

Signal definition (Bollinger-style): cumulative log return over REV_L days, z-scored
against its own trailing REV_Z-day distribution, direction FIXED at -1 (fade the
z-score -> reversion). This is the same shape as the vol-level test (z-score a
rolling feature) but with the direction hard-coded to reversion instead of switched.

Sweeps (REV_L, REV_Z) pairs and reports, for ALGO and cross-sectionally for all 51
names: full/OLD/NEW IC, H1/H2 persistence corr, and a circular-shift surrogate p-value.
"""
import numpy as np, pandas as pd, math

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape

def zrev_ret(series, REV_L, REV_Z):
    lp = np.log(series)
    r = np.diff(lp)                                    # r[i] = day i->i+1 return
    T = len(lp)
    cum = np.full(T, np.nan)                            # cum[t] = sum of r[t-REV_L:t]  (return ending at t, causal)
    for t in range(REV_L, T):
        cum[t] = r[t - REV_L:t].sum()
    z = np.full(T, np.nan)
    for t in range(REV_L + REV_Z, T):
        w = cum[t - REV_Z:t]
        z[t] = (cum[t] - w.mean()) / (w.std() + 1e-12)
    sig = -z                                             # reversion: fade the z-score
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lp[1:] - lp[:-1]
    return sig, ret1

def ic(x_full, y_full, s, e):
    idx = np.arange(0, len(x_full) - 1)
    m = (idx >= s) & (idx <= e) & ~np.isnan(x_full[:len(x_full) - 1])
    x = x_full[:len(x_full) - 1][m]; y = y_full[:len(x_full) - 1][m]
    if len(x) < 40 or x.std() < 1e-12: return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def shift_p(x_full, y_full, s, e, N=2000):
    idx = np.arange(0, len(x_full) - 1)
    m = (idx >= s) & (idx <= e) & ~np.isnan(x_full[:len(x_full) - 1])
    x = x_full[:len(x_full) - 1][m]; y = y_full[:len(x_full) - 1][m]
    n = len(x)
    if n < 40 or x.std() < 1e-12: return float("nan")
    obs = np.corrcoef(x, y)[0, 1]
    rng = np.random.RandomState(1); null = np.empty(N)
    for i in range(N):
        sh = rng.randint(20, n - 20); null[i] = np.corrcoef(x, np.roll(y, sh))[0, 1]
    return round(float(np.mean(np.abs(null) >= abs(obs))), 4)

print("=== ALGO (instrument 0) z-score reversion sweep ===")
print(f"{'REV_L':>6} {'REV_Z':>6} {'IC_full':>9} {'IC_OLD':>8} {'IC_NEW':>8} {'H1':>8} {'H2':>8} {'p_full':>8} {'p_new':>8}")
best = None
for REV_L in (5, 10, 20, 30):
    for REV_Z in (40, 60, 90, 120):
        if REV_L + REV_Z + 60 > nt: continue
        sig, ret1 = zrev_ret(P[0], REV_L, REV_Z)
        icf = ic(sig, ret1, 1, nt)
        icold = ic(sig, ret1, 501, 750)
        icnew = ic(sig, ret1, 751, nt)
        h1 = ic(sig, ret1, 1, 500)
        h2 = ic(sig, ret1, 501, nt)
        pf = shift_p(sig, ret1, 1, nt)
        pn = shift_p(sig, ret1, 751, nt)
        print(f"{REV_L:>6} {REV_Z:>6} {icf:>9.4f} {icold:>8.4f} {icnew:>8.4f} {h1:>8.4f} {h2:>8.4f} {pf:>8.4f} {pn:>8.4f}")
        if best is None or (icf if not math.isnan(icf) else -9) > best[0]:
            best = (icf, REV_L, REV_Z)

print(f"\nbest full-sample |IC| combo: REV_L={best[1]} REV_Z={best[2]} IC={best[0]:.4f}")

print("\n=== cross-sectional check: same signal (best combo) on all 51 names ===")
REV_L, REV_Z = best[1], best[2]
icf_all = []; icn_all = []; ih1 = []; ih2 = []
for i in range(nInst):
    sig, ret1 = zrev_ret(P[i], REV_L, REV_Z)
    icf_all.append(ic(sig, ret1, 1, nt))
    icn_all.append(ic(sig, ret1, 751, nt))
    ih1.append(ic(sig, ret1, 1, 500))
    ih2.append(ic(sig, ret1, 501, nt))
icf_all = np.array(icf_all); icn_all = np.array(icn_all)
ih1 = np.array(ih1); ih2 = np.array(ih2)
ok = ~np.isnan(icf_all)
print(f"full-sample IC: mean {icf_all[ok].mean():.4f}, median {np.median(icf_all[ok]):.4f}, "
      f"{int((icf_all[ok] > 0).sum())}/{ok.sum()} positive, "
      f"t={icf_all[ok].mean() / (icf_all[ok].std(ddof=1) / math.sqrt(ok.sum())):.2f}")
okh = ~np.isnan(ih1) & ~np.isnan(ih2)
print(f"H1 vs H2 persistence corr across names: {np.corrcoef(ih1[okh], ih2[okh])[0,1]:.3f}")
algo_pctile = round(100 * float((icf_all[ok] < icf_all[0]).mean()))
print(f"ALGO full IC {icf_all[0]:.4f} = {algo_pctile}th percentile of {ok.sum()} names")
print(f"ALGO new-window IC {icn_all[0]:.4f}")
