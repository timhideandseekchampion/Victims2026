"""Synthetic stress test for the SHIPPED strategy: fit the one-factor market model + calibrated
idiosyncratic OU + calibrated ALGO vol-continuation process to OUR actual prices.txt, then generate
MANY independent synthetic 1000-day panels and run the ACTUAL SAFE_llvol.py on each one, collecting
the score distribution. This is the cross-draw robustness check nothing else tonight has done --
everything else was validated on this one file; this asks "how would the shipped code do on a
FRESH draw from a matching generator, many times over."
"""
import numpy as np, pandas as pd
import SAFE_llvol as SHIPPED

P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_real.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp_real = np.log(P_real)
r_real = np.diff(logp_real, axis=1)
r0_real = r_real[0]

print("=== 1. fit the one-factor market model to real data ===")
beta = np.array([np.polyfit(r0_real, r_real[k], 1)[0] for k in range(nInst)])
beta = np.clip(beta, 0.0, 3.0)
resid_real = r_real - np.outer(beta, r0_real)
sigma_e = resid_real.std(axis=1)
print(f"mean beta (excl ALGO): {beta[1:].mean():.3f}   mean sigma_e (excl ALGO): {sigma_e[1:].mean():.4f}")

print("\n=== 2. fit ALGO's own vol-continuation process ===")
VOL_WIN, VOL_Z = 20, 60


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol_real = np.full(len(r0_real), np.nan); vol_real[VOL_WIN - 1:] = roll_std(r0_real, VOL_WIN)
log_vol_real = np.log(vol_real[~np.isnan(vol_real)] + 1e-9)
# AR(1) fit on log-vol (mean-reverting stochastic vol)
lv = log_vol_real
phi_v = np.polyfit(lv[:-1], lv[1:], 1)[0]
phi_v = float(np.clip(phi_v, 0.80, 0.995))
resid_v = lv[1:] - phi_v * lv[:-1] - (1 - phi_v) * lv.mean()
sig_innov = resid_v.std()
mean_log_vol = lv.mean()
print(f"log-vol AR(1): phi={phi_v:.4f}  mean_log_vol={mean_log_vol:.4f}  innov_std={sig_innov:.4f}")

# vol-in-mean coefficient: regress ALGO's return on its OWN trailing volz (same construction as SAFE_llvol)
volz_real = np.full(len(r0_real), np.nan)
for s in range(VOL_WIN + VOL_Z, len(r0_real)):
    wv = vol_real[s - VOL_Z:s]
    volz_real[s] = (vol_real[s] - wv.mean()) / (wv.std() + 1e-12)
ok = ~np.isnan(volz_real[:-1])
lam_vm = np.polyfit(volz_real[:-1][ok], r0_real[1:][ok], 1)[0]
mu_m = r0_real.mean()
print(f"vol-in-mean coefficient (lambda): {lam_vm:.6f}   mu_m: {mu_m:.6f}")

print("\n=== 3. calibrate idiosyncratic OU speed (kappa) to reproduce the observed idio-leg score ===")


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
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


def simulate(ndays, kappa, seed):
    rng = np.random.default_rng(seed)
    logm = np.zeros(ndays); lv_path = np.zeros(ndays); lv_path[0] = mean_log_vol
    for t in range(1, ndays):
        lv_path[t] = phi_v * lv_path[t - 1] + (1 - phi_v) * mean_log_vol + rng.normal(0, sig_innov)
        vol_t = np.exp(lv_path[t])
        vol_hist = np.exp(lv_path[max(0, t - VOL_Z):t])
        volz_t = (vol_t - vol_hist.mean()) / (vol_hist.std() + 1e-9) if len(vol_hist) > 5 else 0.0
        drift = mu_m + lam_vm * volz_t
        logm[t] = logm[t - 1] + drift + rng.normal(0, vol_t)
    S_ = np.zeros(nInst - 1)
    out = np.zeros((nInst, ndays))
    P0 = P_real[:, 0]
    for t in range(ndays):
        S_ = S_ - kappa * (S_ - S_.mean()) + rng.normal(0, 1, nInst - 1) * sigma_e[1:]
        out[0, t] = P0[0] * np.exp(logm[t])
        out[1:, t] = P0[1:] * np.exp(beta[1:] * logm[t] + S_)
    return out


import SAFE as SAFEMOD


def idio_only_score(panel, kappa_tag=""):
    n_inst2, ndays = panel.shape
    POS = np.zeros((n_inst2, ndays))
    for k in range(SAFEMOD.WARMUP, ndays):
        cur = panel[:, k]; lim = (dlr / cur).astype(int)
        full = np.asarray(SAFEMOD.getMyPosition(panel[:, :k + 1])); p = full.copy(); p[0] = 0
        POS[:, k] = np.clip(p, -lim, lim).astype(int)
    S_, E_ = ndays - NUMTEST, ndays
    return window(POS, panel, S_, E_)


obs_idio_score = 585.0  # established many times tonight (idio-only, both windows ~585)
best_k = None
for kappa in (0.005, 0.01, 0.015, 0.02, 0.03, 0.05):
    scs = [idio_only_score(simulate(500, kappa, 4000 + s)) for s in range(8)]
    m = float(np.mean(scs))
    print(f"  kappa={kappa}: synth idio-only score {m:.1f}")
    if best_k is None or abs(m - obs_idio_score) < best_k[1]:
        best_k = (kappa, abs(m - obs_idio_score))
kappa = best_k[0]
print(f"-> chosen kappa={kappa}")

print("\n=== 4. FIXED generator: use SAFE.py's own fitted ridge coefficients as the assumed-true ===")
print("    idiosyncratic propagation matrix (parametric bootstrap), not symmetric cross-sectional OU ===")

X_full = r_real[:, :-1].T; Y_full = r_real[1:, 1:].T
B_true, mx_true, my_true = SAFEMOD._ewls_ridge(X_full, Y_full, hl=1000, a=SAFEMOD.RIDGE_A)
resid_true = Y_full - (my_true + (X_full - mx_true) @ B_true)
resid_std_true = resid_true.std(axis=0)
print(f"fitted propagation matrix B_true: {B_true.shape}  mean|coef|={np.abs(B_true).mean():.4f}  "
      f"residual std (avg): {resid_std_true.mean():.4f}")


def simulate_varlike(ndays, seed):
    """ALGO: same calibrated vol-continuation process. Stocks: driven by the FITTED B_true matrix
    (today's full 51-vector of returns -> tomorrow's 50 stock returns), i.e. assume the fitted model
    IS the true generator, plus fresh residual noise each day (parametric bootstrap)."""
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

    stock_ret = np.zeros((nInst - 1, ndays))
    burn = 60
    for t in range(1, ndays):
        if t <= burn:
            stock_ret[:, t] = rng.normal(0, resid_std_true)
            continue
        x_full = np.concatenate([[algo_ret[t - 1]], stock_ret[:, t - 1]])
        pred = my_true + (x_full - mx_true) @ B_true
        stock_ret[:, t] = pred + rng.normal(0, resid_std_true)

    out = np.zeros((nInst, ndays))
    out[0, :] = P_real[0, 0] * np.exp(logm)
    out[1:, :] = P_real[1:, [0]] * np.exp(np.cumsum(stock_ret, axis=1))
    return out


scs = [idio_only_score(simulate_varlike(500, 5000 + s)) for s in range(8)]
print(f"synth idio-only score (parametric-bootstrap generator, 8 draws): mean={np.mean(scs):.1f}  "
      f"vals={[round(s,1) for s in scs]}")
print(f"observed real idio-only score (reference): ~585")

print("\n=== 5. full stress test: generate many 1000-day synthetic panels, run the ACTUAL shipped ===")
print("    SAFE_llvol.getMyPosition walk-forward on each, score with the SAME window() convention used")
print("    all night (already cross-checked against eval_llvol.py's cash-accounting -- identical PnL) ===")


def full_shipped_score(prcHist, numTestDays=250):
    n_inst, nt_ = prcHist.shape
    startDay = nt_ - numTestDays
    POS = np.zeros((n_inst, nt_))
    for k in range(startDay - 1, nt_ - 1):
        POS[:, k] = SHIPPED.getMyPosition(prcHist[:, :k + 1])
    return window(POS, prcHist, startDay, nt_)



print("\nsanity check: full_shipped_score on REAL prices.txt must reproduce the known eval_llvol.py score (761.78)")
import time
t0 = time.time()
real_check = full_shipped_score(P_real, numTestDays=250)
print(f"full_shipped_score(real prices.txt) = {real_check:.2f}   (took {time.time()-t0:.1f}s)")

print("\n=== 6. MAIN RUN: N synthetic 1000-day panels, full shipped strategy, walk-forward, official score ===")
N_DRAWS = 60
synth_scores = []
t0 = time.time()
for s in range(N_DRAWS):
    panel = simulate_varlike(1000, 9000 + s)
    sc = full_shipped_score(panel, numTestDays=250)
    synth_scores.append(sc)
    if (s + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  [{s+1}/{N_DRAWS}] running mean={np.mean(synth_scores):.1f}  "
              f"elapsed={elapsed:.0f}s  est_total={elapsed/(s+1)*N_DRAWS:.0f}s")

synth_scores = np.array(synth_scores)
print(f"\ntotal time: {time.time()-t0:.0f}s")
print("\n=== SYNTHETIC SCORE DISTRIBUTION (SAFE_llvol.py, N={} independent redraws) ===".format(N_DRAWS))
print(f"mean:   {synth_scores.mean():.1f}")
print(f"median: {np.median(synth_scores):.1f}")
print(f"std:    {synth_scores.std():.1f}")
print(f"min:    {synth_scores.min():.1f}   max: {synth_scores.max():.1f}")
for p in (5, 10, 25, 75, 90, 95):
    print(f"p{p}:    {np.percentile(synth_scores, p):.1f}")
print(f"% of draws with score > 0:   {(synth_scores > 0).mean()*100:.0f}%")
print(f"% of draws with score > 500: {(synth_scores > 500).mean()*100:.0f}%")
print(f"% of draws with score > 761 (our real-file score): {(synth_scores > 761).mean()*100:.0f}%")
print(f"% of draws with score > 900 (reported top-team range): {(synth_scores > 900).mean()*100:.0f}%")
print(f"our real prices.txt score (761.78) sits at percentile: "
      f"{(synth_scores < 761.78).mean()*100:.0f} of the synthetic distribution")
