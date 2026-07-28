"""
test_v10_stress_synthetic.py

Parametric-bootstrap significance test for v10, extending stress_test_synthetic.py's methodology
(re-implemented fresh here rather than imported, since importing that file would trigger its own
expensive 60-draw main run as a side effect).

QUESTION: v9's beta-demean and v10's rank-stability were both found via walk-forward checks on the
SAME 1000 historical days used throughout this whole investigation. Even a walk-forward split (train
on OLD, test on NEW) can't rule out that the improvement exploits some quirk specific to this ONE
realized price path, since both windows come from the same single historical draw. This asks a
different question: does the improvement generalize to FRESH synthetic draws from a generator that
does NOT specifically encode beta-demean or rank-stability (it's fit from a plain single-half-life
ridge + one-factor ALGO vol-continuation process -- the same calibration `stress_test_synthetic.py`
already validated reproduces the real file's score almost exactly)? If v10 beats v9 on MOST fresh
draws (not just this one), that's evidence the mechanism exploits a genuine, generalizable statistical
regularity (one-factor market + VAR-like propagation + stochastic vol) rather than this path's noise.

PAIRED design: for each synthetic draw, run v9 and v10 (identical generator draw, identical starting
prices) and take the difference. This controls for per-draw noise far better than comparing separate
score distributions -- the question is "does v10 beat v9 on the SAME draw", not "are the two
distributions different in aggregate."
"""
import numpy as np, pandas as pd, time
import SAFE_llboost as SHIPPED
import SAFE_llboost_v9 as V9
import SAFE_llboost_v10 as V10
import SAFE as SAFEMOD

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


print("=== 1. calibrate the generator (identical methodology to stress_test_synthetic.py) ===")
VOL_WIN, VOL_Z = 20, 60


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol_real = np.full(len(r0_real), np.nan); vol_real[VOL_WIN - 1:] = roll_std(r0_real, VOL_WIN)
log_vol_real = np.log(vol_real[~np.isnan(vol_real)] + 1e-9)
lv = log_vol_real
phi_v = float(np.clip(np.polyfit(lv[:-1], lv[1:], 1)[0], 0.80, 0.995))
resid_v = lv[1:] - phi_v * lv[:-1] - (1 - phi_v) * lv.mean()
sig_innov = resid_v.std()
mean_log_vol = lv.mean()

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
print(f"  phi_v={phi_v:.4f}  mean_log_vol={mean_log_vol:.4f}  lam_vm={lam_vm:.6f}  "
      f"mean|B_true|={np.abs(B_true).mean():.4f}")


def simulate_varlike(ndays, seed):
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


print("\n=== 2. sanity check: v10 on the REAL prices.txt via this script's own harness must match 912.64 ===")


def full_score(mod, prcHist, numTestDays=250):
    n_inst_, nt_ = prcHist.shape
    startDay = nt_ - numTestDays
    POS = np.zeros((n_inst_, nt_))
    for k in range(startDay - 1, nt_ - 1):
        POS[:, k] = mod.getMyPosition(prcHist[:, :k + 1])
    return window(POS, prcHist, startDay, nt_)


t0 = time.time()
real_check_v10 = full_score(V10, P_real, 250)
print(f"  v10 on real prices.txt: {real_check_v10:.2f}  (official eval: 912.64)  [{time.time()-t0:.0f}s]")

print("\n=== 3. MAIN RUN: paired v9 vs v10 (and baseline vs v9) on N fresh synthetic 1000-day draws ===")
N_DRAWS = 25
rows = []
t0 = time.time()
for s in range(N_DRAWS):
    panel = simulate_varlike(1000, 9000 + s)
    sc_base = full_score(SHIPPED, panel, NUMTEST)
    sc_v9 = full_score(V9, panel, NUMTEST)
    sc_v10 = full_score(V10, panel, NUMTEST)
    rows.append((sc_base, sc_v9, sc_v10))
    elapsed = time.time() - t0
    print(f"  [{s+1}/{N_DRAWS}] base={sc_base:7.1f}  v9={sc_v9:7.1f}  v10={sc_v10:7.1f}  "
          f"(v10-v9)={sc_v10-sc_v9:+7.1f}  elapsed={elapsed:.0f}s  "
          f"est_total={elapsed/(s+1)*N_DRAWS:.0f}s", flush=True)

rows = np.array(rows)
base_s, v9_s, v10_s = rows[:, 0], rows[:, 1], rows[:, 2]
d_v10_v9 = v10_s - v9_s
d_v9_base = v9_s - base_s

print(f"\ntotal time: {time.time()-t0:.0f}s")
print("\n" + "=" * 96)
print(f"SYNTHETIC SCORE DISTRIBUTION (N={N_DRAWS} independent fresh draws)")
print("=" * 96)
for nm, arr in (("baseline", base_s), ("v9", v9_s), ("v10", v10_s)):
    print(f"  {nm:<10} mean={arr.mean():7.1f}  median={np.median(arr):7.1f}  std={arr.std():6.1f}  "
          f"min={arr.min():7.1f}  max={arr.max():7.1f}")

print("\n--- PAIRED: v10 - v9 on the SAME draw (the key question) ---")
print(f"  mean diff:   {d_v10_v9.mean():+.1f}")
print(f"  median diff: {np.median(d_v10_v9):+.1f}")
print(f"  std diff:    {d_v10_v9.std():.1f}")
print(f"  v10 beats v9 on {int((d_v10_v9 > 0).sum())}/{N_DRAWS} draws "
      f"({100*(d_v10_v9>0).mean():.0f}%)")
# paired t-test (simple, no scipy dependency needed beyond what's already used elsewhere in this repo)
from scipy import stats as sstats
tstat, pval = sstats.ttest_1samp(d_v10_v9, 0.0)
print(f"  paired t-test (H0: mean diff = 0): t={tstat:+.2f}  p={pval:.4f}")

print("\n--- PAIRED: v9 - baseline on the SAME draw (context: does v9's own improvement generalize?) ---")
print(f"  mean diff: {d_v9_base.mean():+.1f}  v9 beats baseline on "
      f"{int((d_v9_base > 0).sum())}/{N_DRAWS} draws ({100*(d_v9_base>0).mean():.0f}%)")
tstat2, pval2 = sstats.ttest_1samp(d_v9_base, 0.0)
print(f"  paired t-test: t={tstat2:+.2f}  p={pval2:.4f}")

print("\n--- for reference: real prices.txt deltas (v10-v9, v9-base) vs this synthetic distribution ---")
real_v9 = full_score(V9, P_real, 250)
real_base = full_score(SHIPPED, P_real, 250)
print(f"  real: v10-v9 = {real_check_v10 - real_v9:+.1f}  (synthetic mean {d_v10_v9.mean():+.1f}, "
      f"std {d_v10_v9.std():.1f})")
print(f"  real: v9-base = {real_v9 - real_base:+.1f}  (synthetic mean {d_v9_base.mean():+.1f}, "
      f"std {d_v9_base.std():.1f})")
