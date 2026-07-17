#!/usr/bin/env python
"""alpha_lab.py — a clean-slate research harness for prices.txt.

Purpose: replace the old (circular) validation with HONEST measurement.
This file is NOT submitted; it only informs the book in teamName.py.

Step 1 (this file, `measure`): characterise what structure actually PERSISTS
across a strict train/test split of prices.txt, so we only build on edges that
survive out-of-sample. `train = days[0:250]`, `proxy = days[250:500]` — the proxy
is exactly the window eval.py scores.

Later steps (`validate`) add the non-circular robustness tests.

Usage:
    python alpha_lab.py measure      # Step 1 table
"""
import sys
import numpy as np

PRICES_FILE = "./prices.txt"
SPLIT = 250          # train = [0:SPLIT], proxy = [SPLIT:]
IDIO = slice(1, None)  # the 50 idiosyncratic names (exclude ALGO / inst 0)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_prices(fn=PRICES_FILE):
    """Return prices as (nInst, nDays) float array (row = instrument), matching
    eval.py's orientation."""
    arr = np.loadtxt(fn, skiprows=1)      # (nDays, nInst); skip ticker header row
    return arr.T                          # (nInst, nDays)


# ---------------------------------------------------------------------------
# building blocks (lifted math — pure, stateless)
# ---------------------------------------------------------------------------
def daily_returns(prc):
    """Simple daily returns, shape (nInst, nDays-1). col t = prc[:,t+1]/prc[:,t]-1."""
    return prc[:, 1:] / prc[:, :-1] - 1.0


def xs_demean(mat, rows=IDIO):
    """Cross-sectionally demean the selected rows on each day (column). Returns a
    copy with those rows made ~market-neutral; other rows untouched."""
    out = mat.copy()
    sub = out[rows]
    out[rows] = sub - np.nanmean(sub, axis=0, keepdims=True)
    return out


def zscore_col(prc_slice, w):
    """(price - rolling mean)/rolling std over the last w cols, per row. prc_slice
    is (nInst, >=w). Returns length-nInst vector as of the last column."""
    window = min(w, prc_slice.shape[1])
    recent = prc_slice[:, -window:]
    mu = recent.mean(axis=1)
    sd = recent.std(axis=1)
    return np.divide(prc_slice[:, -1] - mu, sd, out=np.zeros_like(mu), where=sd > 1e-9)


def nan_corr(a, b):
    """Pearson corr over finite pairs; 0 if degenerate."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return 0.0
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------
def reversion_ic(prc, w, h, day_lo, day_hi):
    """Mean cross-sectional IC of the contrarian z-reversion signal.

    On each day t in [day_lo, day_hi): signal = -zscore(prc[:, :t+1], w) on the 50
    idio names (cross-sectionally demeaned); forward target = the name's h-day
    return from t, also cross-sectionally demeaned (the book is market-neutral, so
    residual return is what matters). IC_t = corr over names of (signal, fwd).
    Returns (mean_ic, ic_information_ratio)."""
    ics = []
    for t in range(day_lo, day_hi):
        if t < w + 1 or t + h >= prc.shape[1]:
            continue
        sig = -zscore_col(prc[:, :t + 1], w)[IDIO]
        sig = sig - np.nanmean(sig)
        fwd = prc[IDIO, t + h] / prc[IDIO, t] - 1.0
        fwd = fwd - np.nanmean(fwd)
        ics.append(nan_corr(sig, fwd))
    ics = np.array(ics)
    if len(ics) == 0:
        return 0.0, 0.0
    ir = ics.mean() / ics.std() * np.sqrt(len(ics)) if ics.std() > 1e-12 else 0.0
    return float(ics.mean()), float(ir)


def horizon_autocorr(rets, h, rows=IDIO):
    """Mean per-name lag-1 autocorrelation of non-overlapping h-day returns of the
    (already cross-sectionally demeaned) return series. Negative = reversion."""
    r = rets[rows]
    # build non-overlapping h-day compounded returns
    T = r.shape[1]
    nblk = T // h
    if nblk < 3:
        return 0.0
    blk = np.stack([np.prod(1 + r[:, i * h:(i + 1) * h], axis=1) - 1 for i in range(nblk)], axis=1)
    acs = []
    for i in range(blk.shape[0]):
        x = blk[i, :-1]
        y = blk[i, 1:]
        acs.append(nan_corr(x, y))
    return float(np.mean(acs))


def factor_structure(prc, day_lo, day_hi):
    """Confirm one-factor market, ALGO == index. Returns dict of diagnostics on the
    window [day_lo, day_hi]."""
    rets = daily_returns(prc[:, day_lo:day_hi + 1])   # (nInst, W)
    algo = rets[0]
    ew = rets[IDIO].mean(axis=0)                       # equal-weight avg of 50 names
    corr_algo_ew = nan_corr(algo, ew)
    # PCA on the 50 names' returns (standardised) → variance shares
    X = rets[IDIO]
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = np.cov(Xc)
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]
    shares = eig / eig.sum()
    # mean R^2 of each name regressed on ALGO
    r2s = [nan_corr(rets[i], algo) ** 2 for i in range(1, prc.shape[0])]
    return {
        "corr(ALGO, EWavg)": corr_algo_ew,
        "PC1 share": float(shares[0]),
        "PC2 share": float(shares[1]),
        "PC3 share": float(shares[2]),
        "mean R^2 to ALGO": float(np.mean(r2s)),
    }


def per_name_reversion(prc, day_lo, day_hi):
    """Distribution of per-name lag-1 autocorr of demeaned daily returns. Tells us
    if reversion is broad (safe to trade all 50) or concentrated in a few names."""
    rets = xs_demean(daily_returns(prc), rows=IDIO)[:, day_lo:day_hi]
    acs = []
    for i in range(1, prc.shape[0]):
        x = rets[i, :-1]
        y = rets[i, 1:]
        acs.append(nan_corr(x, y))
    acs = np.array(acs)
    return {
        "mean lag-1 AC": float(acs.mean()),
        "median": float(np.median(acs)),
        "frac reverting (<0)": float((acs < 0).mean()),
        "min": float(acs.min()),
        "max": float(acs.max()),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def measure():
    prc = load_prices()
    n, T = prc.shape
    print(f"Loaded {n} instruments x {T} days. Split: train=[0:{SPLIT}] proxy=[{SPLIT}:{T}] "
          f"(proxy == the window eval.py scores)\n")

    print("=" * 72)
    print("1. CROSS-SECTIONAL REVERSION IC  (signal = -zscore(w); fwd = h-day resid ret)")
    print("   IC>0 means contrarian reversion pays. Want SIGN + magnitude to survive.")
    print("=" * 72)
    header = f"{'window':>7} {'horizon':>8} | {'train IC':>9} {'IR':>6} | {'proxy IC':>9} {'IR':>6} | survives?"
    print(header)
    print("-" * len(header))
    for w in (3, 5, 10, 20):
        for h in (1, 3, 5, 10):
            tr_ic, tr_ir = reversion_ic(prc, w, h, w + 1, SPLIT)
            px_ic, px_ir = reversion_ic(prc, w, h, SPLIT, T)
            survives = "YES" if (tr_ic > 0 and px_ic > 0) else ("sign-flip!" if tr_ic * px_ic < 0 else "weak")
            print(f"{w:>7} {h:>8} | {tr_ic:>+9.4f} {tr_ir:>6.1f} | {px_ic:>+9.4f} {px_ir:>6.1f} | {survives}")

    print("\n" + "=" * 72)
    print("2. HORIZON AUTOCORRELATION of demeaned returns  (negative = reversion)")
    print("=" * 72)
    rets_tr = xs_demean(daily_returns(prc[:, :SPLIT]))
    rets_px = xs_demean(daily_returns(prc[:, SPLIT:]))
    print(f"{'h-day':>6} | {'train AC':>9} | {'proxy AC':>9}")
    print("-" * 30)
    for h in (1, 3, 5, 10):
        print(f"{h:>6} | {horizon_autocorr(rets_tr, h):>+9.4f} | {horizon_autocorr(rets_px, h):>+9.4f}")

    print("\n" + "=" * 72)
    print("3. FACTOR STRUCTURE  (is it one-factor, ALGO == the index?)")
    print("=" * 72)
    fs_tr = factor_structure(prc, 0, SPLIT)
    fs_px = factor_structure(prc, SPLIT, T - 1)
    print(f"{'metric':>20} | {'train':>9} | {'proxy':>9}")
    print("-" * 46)
    for k in fs_tr:
        print(f"{k:>20} | {fs_tr[k]:>9.3f} | {fs_px[k]:>9.3f}")

    print("\n" + "=" * 72)
    print("4. PER-NAME REVERSION breadth  (is reversion broad or concentrated?)")
    print("=" * 72)
    pn_tr = per_name_reversion(prc, 0, SPLIT)
    pn_px = per_name_reversion(prc, SPLIT, T - 1)
    print(f"{'metric':>20} | {'train':>9} | {'proxy':>9}")
    print("-" * 46)
    for k in pn_tr:
        print(f"{k:>20} | {pn_tr[k]:>9.3f} | {pn_px[k]:>9.3f}")

    print("\nDone. Build the book only on rows marked 'YES' above (sign + magnitude persist).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if cmd == "measure":
        measure()
    else:
        print(f"unknown command: {cmd}")
