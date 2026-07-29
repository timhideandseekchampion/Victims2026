"""
test_batch100_D63_haarwave.py

D63: wavelet-decomposed log-price series as ridge INPUT instead of raw 1-day log-returns. pywt is not
installed in this environment, so this hand-rolls a standard stationary ("a trous") Haar-style
multi-resolution decomposition, computed causally per instrument:

    S0 = logp
    S1[t] = (S0[t] + S0[t-1]) / 2                (running pairwise mean, gap 1)
    D1[t] = S0[t] - S1[t]                         (detail/scale 1  ~ half the 1-day return)
    S2[t] = (S1[t] + S1[t-2]) / 2                 (gap doubles to 2)
    D2[t] = S1[t] - S2[t]                         (detail/scale 2  ~ short trend wiggle)
    S3[t] = (S2[t] + S2[t-4]) / 2                 (gap doubles to 4)
    D3[t] = S2[t] - S3[t]                         (detail/scale 3  ~ medium trend wiggle)

D1/D2/D3 are the 3 detail (wavelet) coefficients at increasing scales; all causal (only look at logp
up to and including t). Ridge features become the CONCATENATION of D1, D2, D3 across all nInst
instruments (p: 51 -> 153) instead of the single raw-return column -- a genuine multi-scale
decomposition, not just a smoothed/relabelled return.

NEW MODEL CLASS (per repo policy: gets a single-config precheck, not an exhaustive grid) -- one config
tested: all 3 detail levels concatenated, fed through the SAME ridge ensemble machinery (V10._ewls_ridge,
unchanged) and combined with BLEND/boost/rank-stability/ALGO leg exactly as V10 does, via the shared
batch100_d6x_shared.py precompute.
"""
import numpy as np, time
import SAFE_llboost_v10 as V10
import batch100_d6x_shared as SH

logp, r, days, nIdio, nt = SH.logp, SH.r, SH.days, SH.nIdio, SH.nt
HALF_LIVES, RIDGE_A = SH.HALF_LIVES, SH.RIDGE_A

print(f"\nSANITY_CHECK_PASSED (shared baseline) = {SH.SANITY_OK}")

print("\n=== precompute: causal a-trous Haar-style detail coefficients D1,D2,D3 (per instrument) ===")
t0 = time.time()
S0 = logp
S1 = np.full_like(S0, np.nan); S1[:, 1:] = (S0[:, 1:] + S0[:, :-1]) / 2.0
D1 = np.full_like(S0, np.nan); D1[:, 1:] = S0[:, 1:] - S1[:, 1:]

S2 = np.full_like(S0, np.nan); S2[:, 2:] = (S1[:, 2:] + S1[:, :-2]) / 2.0
D2 = np.full_like(S0, np.nan); D2[:, 2:] = S1[:, 2:] - S2[:, 2:]

S3 = np.full_like(S0, np.nan); S3[:, 4:] = (S2[:, 4:] + S2[:, :-4]) / 2.0
D3 = np.full_like(S0, np.nan); D3[:, 4:] = S2[:, 4:] - S3[:, 4:]
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

# align to r's indexing: r[:, k] = logp[:, k+1] - logp[:, k], i.e. r's column k corresponds to logp
# column k+1. D1/D2/D3 columns are indexed on logp's time axis (0..nt-1); use columns [1:] to match r.
D1r = D1[:, 1:]; D2r = D2[:, 1:]; D3r = D3[:, 1:]     # each (nInst, nt-1), same shape as r
D1r = np.where(np.isfinite(D1r), D1r, r)
D2r = np.where(np.isfinite(D2r), D2r, r)
D3r = np.where(np.isfinite(D3r), D3r, r)
FEATMAT = np.concatenate([D1r, D2r, D3r], axis=0)     # (nInst*3, nt-1), precomputed ONCE (vectorized)


print("\n=== sanity check: raw-return ridge features (mechanism OFF), re-derived via this script's own "
      "build_wz_ridge, must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE0 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    Y = V10._beta_adjusted_target(rr_)
    X = rr_[:, :-1].T
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    WZ_BASE0[:, t] = np.mean(fs, 0)
c0 = SH.evaluate("raw-return (=v10)", WZ_BASE0)
print(f"  [{time.time()-t0:.0f}s]")
SANITY_OK = abs(c0["wo"] - 871.0) < 0.5 and abs(c0["wn"] - 912.6) < 0.5 and SH.SANITY_OK
print("  OK -- matches v10 to within rounding." if SANITY_OK else
      "  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")


print("\n=== CANDIDATE: wavelet (D1+D2+D3 concatenated) ridge features, single config ===")
t0 = time.time()
WZ_WAVE = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    Y = V10._beta_adjusted_target(rr_)
    Xw = FEATMAT[:, :t - 1].T                                          # (n_train, 153)
    xq = FEATMAT[:, t - 1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(Xw, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    WZ_WAVE[:, t] = np.mean(fs, 0)
c = SH.evaluate("wavelet D1+D2+D3", WZ_WAVE, SH.base_wo, SH.base_wn, SH.base_scs)
print(f"  [{time.time()-t0:.0f}s]")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
