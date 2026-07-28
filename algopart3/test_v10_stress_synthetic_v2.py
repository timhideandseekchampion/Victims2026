"""
test_v10_stress_synthetic_v2.py

Follow-up requested after test_v10_stress_synthetic.py: build the "differently-enriched generator"
that was explicitly flagged as not yet built, to give v10's rank-stability mechanism a fair synthetic
test (the earlier refined generator only fixed v9's target -- residual common-mode correlation --
and left v10 untested by construction, not refuted).

WHAT WAS CHECKED FIRST, HONESTLY, BEFORE BUILDING ANYTHING: does the real data show genuine per-stock
OWN-RETURN autocorrelation at short vs medium lags (the naive way to imagine "trend + short-term
countermove" structure)? Measured on ridge residuals (post lag-1 VAR removal) at lags 1-30: every
value is <0.013 in magnitude, no pattern -- essentially zero. There is NO honest, non-arbitrary
magnitude to calibrate a raw-autocorrelation-based enrichment against, so that route was abandoned
rather than faked.

WHAT v10 ACTUALLY MEASURES is different: the CROSS-SECTIONAL z-scored short/long return divergence
(not raw own-return autocorrelation). Checked THAT specific quantity instead: pooled IC of v10's exact
vote construction (short8/long22) against next-day return, full real sample = +0.0147 (n=14,487) --
small, but real and directly measurable, unlike the null raw-autocorrelation check.

THE ENRICHMENT, stated plainly so the limitation is clear: rather than reverse-engineer an AR process
that organically produces this cross-sectional pattern (a much bigger, more speculative modeling
exercise), this injects a small return component proportional to the SAME causally-computable vote
signal, calibrated so the resulting pooled IC matches the REAL measured value (+0.0147) -- not tuned
to whatever would make v10 win. This is a more direct, and more honestly-limited, test than an
organic-dynamics generator would be: it asks "if a signal of EXACTLY the real-world-measured strength
existed, does v10's specific construction detect and profit from it the way v9's construction did for
the residual-correlation case?" -- not "does this arise naturally from richer price dynamics."
Both limitations are stated, not hidden.

Combined with the already-validated residual common-mode fix (rho=0.202, matching test_pc2_probe.py),
this generator now honestly contains BOTH structural features v9 and v10 individually target.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost as SHIPPED
import SAFE_llboost_v9 as V9
import SAFE_llboost_v10 as V10
import SAFE as SAFEMOD
from scipy import stats as sstats

P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_real.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp_real = np.log(P_real)
r_real = np.diff(logp_real, axis=1)
r0_real = r_real[0]


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, PR, S, E):
    n_inst = POS.shape[0]
    curPos = np.zeros(n_inst); comm_vec = np.zeros(n_inst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = PR[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


def full_score(mod, prcHist, numTestDays=250):
    n_inst_, nt_ = prcHist.shape
    startDay = nt_ - numTestDays
    POS = np.zeros((n_inst_, nt_))
    for k in range(startDay - 1, nt_ - 1):
        POS[:, k] = mod.getMyPosition(prcHist[:, :k + 1])
    return window(POS, prcHist, startDay, nt_)


print("=== 1. calibrate (identical to test_v10_stress_synthetic.py) ===")
VOL_WIN, VOL_Z = 20, 60


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol_real = np.full(len(r0_real), np.nan); vol_real[VOL_WIN - 1:] = roll_std(r0_real, VOL_WIN)
lv = np.log(vol_real[~np.isnan(vol_real)] + 1e-9)
phi_v = float(np.clip(np.polyfit(lv[:-1], lv[1:], 1)[0], 0.80, 0.995))
resid_v = lv[1:] - phi_v * lv[:-1] - (1 - phi_v) * lv.mean()
sig_innov = resid_v.std(); mean_log_vol = lv.mean()

volz_real = np.full(len(r0_real), np.nan)
for s in range(VOL_WIN + VOL_Z, len(r0_real)):
    wv = vol_real[s - VOL_Z:s]
    volz_real[s] = (vol_real[s] - wv.mean()) / (wv.std() + 1e-12)
ok = ~np.isnan(volz_real[:-1])
lam_vm = np.polyfit(volz_real[:-1][ok], r0_real[1:][ok], 1)[0]
mu_m = r0_real.mean()

X_full = r_real[:, :-1].T; Y_full = r_real[1:, 1:].T
B_true, mx_true, my_true = SAFEMOD._ewls_ridge(X_full, Y_full, hl=1000, a=SAFEMOD.RIDGE_A)
resid_true = Y_full - (my_true + (X_full - mx_true) @ B_true)
resid_std_true = resid_true.std(axis=0)
RHO = 0.202
print(f"  phi_v={phi_v:.4f}  RHO(common-mode)={RHO}")

nIdio = nInst - 1
SHORT_W, LONG_W = 8, 22
TARGET_VOTE_IC = 0.0147   # measured directly on the real data above -- not tuned


def vote_from_path(logp_stocks, t):
    """Same construction as V10._rank_stability_signal, applied to a (nIdio, >=t+1) log-price array."""
    if t < max(SHORT_W, LONG_W) + 5:
        return np.zeros(nIdio)
    short_ret = logp_stocks[:, t] - logp_stocks[:, t - SHORT_W]
    long_ret = logp_stocks[:, t] - logp_stocks[:, t - LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        return np.zeros(nIdio)
    sz = sz / sstd; lz = lz / lstd
    disagree = np.sign(lz) != np.sign(sz)
    return np.where(disagree, -sz, 0.0)


def simulate_enriched(ndays, seed, inject_k):
    """Common-mode residual (rho=0.202) generator, PLUS a small return component proportional to the
    causally-computed vote signal, injected with coefficient inject_k (calibrated separately, see
    calibration loop below, to hit TARGET_VOTE_IC)."""
    rng = np.random.default_rng(seed)
    logm = np.zeros(ndays); lv_path = np.zeros(ndays); lv_path[0] = mean_log_vol
    algo_ret = np.zeros(ndays)
    for t in range(1, ndays):
        lv_path[t] = phi_v * lv_path[t - 1] + (1 - phi_v) * mean_log_vol + rng.normal(0, sig_innov)
        vol_t = np.exp(lv_path[t])
        vol_hist = np.exp(lv_path[max(0, t - VOL_Z):t])
        volz_t = (vol_t - vol_hist.mean()) / (vol_hist.std() + 1e-9) if len(vol_hist) > 5 else 0.0
        drift = mu_m + lam_vm * volz_t
        algo_ret[t] = drift + rng.normal(0, vol_t)
        logm[t] = logm[t - 1] + algo_ret[t]

    stock_ret = np.zeros((nIdio, ndays))
    logp_stocks = np.zeros((nIdio, ndays))
    logp_stocks[:, 0] = np.log(P_real[1:, 0])
    burn = max(60, LONG_W + 10)
    sqrt_rho = np.sqrt(RHO); sqrt_1mrho = np.sqrt(1 - RHO)
    for t in range(1, ndays):
        common = rng.normal(0, 1); idio = rng.normal(0, 1, nIdio)
        noise_t = (sqrt_rho * common + sqrt_1mrho * idio) * resid_std_true
        if t <= burn:
            stock_ret[:, t] = noise_t
        else:
            x_full = np.concatenate([[algo_ret[t - 1]], stock_ret[:, t - 1]])
            pred = my_true + (x_full - mx_true) @ B_true
            v = vote_from_path(logp_stocks, t - 1)   # causal: uses log-prices only through t-1
            stock_ret[:, t] = pred + inject_k * v * resid_std_true + noise_t
        logp_stocks[:, t] = logp_stocks[:, t - 1] + stock_ret[:, t]

    out = np.zeros((nInst, ndays))
    out[0, :] = P_real[0, 0] * np.exp(logm)
    out[1:, :] = np.exp(logp_stocks) * (P_real[1:, [0]] / np.exp(logp_stocks[:, [0]]))
    return out


print("\n=== 2. calibrate inject_k to hit the REAL measured pooled vote-IC (+0.0147) ===")


def measure_vote_ic(ndays, n_seeds, inject_k):
    ics = []
    for s in range(n_seeds):
        panel = simulate_enriched(ndays, 70000 + s, inject_k)
        lp = np.log(panel[1:])
        votes, nxt = [], []
        for t in range(LONG_W + 5, ndays - 1):
            v = vote_from_path(lp, t)
            votes.append(v); nxt.append(panel[1:, t + 1] / panel[1:, t] - 1)
        V = np.array(votes); N = np.array(nxt)
        active = V != 0
        if active.sum() > 100:
            ics.append(np.corrcoef(V[active], N[active])[0, 1])
    return float(np.mean(ics))


for k_try in (0.0, 0.02, 0.05, 0.08):
    ic_ = measure_vote_ic(600, 3, k_try)
    print(f"  inject_k={k_try:<5} -> synthetic pooled vote IC = {ic_:+.4f}")

# linear interpolation to hit the target (cheap, since the relationship is close to linear for small k)
k_lo, ic_lo = 0.02, measure_vote_ic(600, 4, 0.02)
k_hi, ic_hi = 0.08, measure_vote_ic(600, 4, 0.08)
inject_k = k_lo + (TARGET_VOTE_IC - ic_lo) * (k_hi - k_lo) / (ic_hi - ic_lo)
inject_k = float(np.clip(inject_k, 0.0, 0.2))
print(f"  interpolated inject_k = {inject_k:.4f} (targeting IC={TARGET_VOTE_IC})")
ic_check = measure_vote_ic(800, 6, inject_k)
print(f"  verification at inject_k={inject_k:.4f}: synthetic pooled vote IC = {ic_check:+.4f} "
      f"(target {TARGET_VOTE_IC})")

print("\n=== 3. sanity: v10 on real prices.txt via this script's harness must match 912.64 ===")
t0 = time.time()
real_check = full_score(V10, P_real, 250)
print(f"  v10 real score: {real_check:.2f}  (official: 912.64)  [{time.time()-t0:.0f}s]")

print("\n=== 4. MAIN RUN: paired base/v9/v10 on N draws from the fully-enriched generator ===")
N_DRAWS = 25
rows = []
t0 = time.time()
for s in range(N_DRAWS):
    panel = simulate_enriched(1000, 30000 + s, inject_k)
    sc_base = full_score(SHIPPED, panel, NUMTEST)
    sc_v9 = full_score(V9, panel, NUMTEST)
    sc_v10 = full_score(V10, panel, NUMTEST)
    rows.append((sc_base, sc_v9, sc_v10))
    print(f"  [{s+1}/{N_DRAWS}] base={sc_base:7.1f}  v9={sc_v9:7.1f}  v10={sc_v10:7.1f}  "
          f"(v9-base)={sc_v9-sc_base:+7.1f}  (v10-v9)={sc_v10-sc_v9:+7.1f}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

rows = np.array(rows)
base_s, v9_s, v10_s = rows[:, 0], rows[:, 1], rows[:, 2]
d_v9_base = v9_s - base_s
d_v10_v9 = v10_s - v9_s
print(f"\ntotal time: {time.time()-t0:.0f}s")
print("\n" + "=" * 96)
print(f"FULLY-ENRICHED GENERATOR (rho={RHO} common-mode + inject_k={inject_k:.4f} vote-IC-matched)")
print("=" * 96)
t9, p9 = sstats.ttest_1samp(d_v9_base, 0.0)
t10, p10 = sstats.ttest_1samp(d_v10_v9, 0.0)
print(f"v9-baseline: mean={d_v9_base.mean():+.1f}  std={d_v9_base.std():.1f}  "
      f"win_rate={100*(d_v9_base>0).mean():.0f}%  t={t9:+.2f}  p={p9:.4f}")
print(f"v10-v9:      mean={d_v10_v9.mean():+.1f}  std={d_v10_v9.std():.1f}  "
      f"win_rate={100*(d_v10_v9>0).mean():.0f}%  t={t10:+.2f}  p={p10:.4f}")
print(f"\nreal-data reference: v9-base={912.64-real_check+real_check:.1f} (n/a, see prior script) -- "
      f"use test_v10_stress_synthetic.py's real deltas (+64.7, +19.3) for comparison.")
