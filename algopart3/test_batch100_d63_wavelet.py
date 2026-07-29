"""
test_batch100_d63_wavelet.py

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

D1/D2/D3 are the 3 detail (wavelet) coefficients at increasing scales; all are causal (only look at
logp up to and including t). Ridge features become the CONCATENATION of D1, D2, D3 across all nInst
instruments (p: 51 -> 153) instead of the single raw-return column -- a genuine multi-scale
decomposition, not just a smoothed/relabelled return.

NEW MODEL CLASS (per repo policy: wavelet decomposition gets a single-config precheck, not an
exhaustive grid) -- one config tested: all 3 detail levels concatenated, fed through the SAME ridge
ensemble machinery (V10._ewls_ridge, unchanged) and combined with BLEND/boost/rank-stability/ALGO leg
exactly as V10 does.
"""
import numpy as np, pandas as pd, time
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
RS_WEIGHT, RS_SHORT_W, RS_LONG_W = V10.RS_WEIGHT, V10.RS_SHORT_W, V10.RS_LONG_W


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

print("=== precompute: REV blend leg, pairwise boost, rank-stability signal, ALGO leg -- all "
      "UNCHANGED / reused verbatim from V10 (independent of the ridge-input mechanism under test) ===",
      flush=True)
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


def combine_wz(wz_ridge, t):
    wz = (1 - V10.BLEND) * wz_ridge + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    s = RS_SIG[:, t]
    if np.isfinite(s).all():
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    return wz


def build_pos(WZ_RIDGE):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = combine_wz(WZ_RIDGE[:, t], t)
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


def evaluate(nm, WZ_RIDGE, base_wo=None, base_wn=None, base_scs=None, verbose=True):
    Pz = build_pos(WZ_RIDGE); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = None
    if base_wo is not None:
        passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = None if base_scs is None else int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ("  <== fail" if passed is False else "")
        extra = f"  n_worse={nworse}/{len(scs)}" if nworse is not None else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}"
              f"{extra}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


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


print("\n=== sanity check: raw-return features (mechanism OFF) must reproduce SAFE_llboost_v10 ===")
t0 = time.time()
WZ_BASE = np.full((nIdio, nt), np.nan)
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
    WZ_BASE[:, t] = np.mean(fs, 0)
POS_base = build_pos(WZ_BASE)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)  [{time.time()-t0:.0f}s]")
if not (abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
    SANITY_OK = False
else:
    print("  OK -- matches v10 to within rounding.")
    SANITY_OK = True


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
c = evaluate("wavelet D1+D2+D3", WZ_WAVE, base_wo, base_wn, base_scs)
print(f"  [{time.time()-t0:.0f}s]")

print(f"\nSANITY_CHECK_PASSED={SANITY_OK}")
