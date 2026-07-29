"""
changepoint_synthetic.py -- shared generator for the controlled lead-lag change-point experiment
(test_v11_changepoint.py). NOT a test_*.py file itself (matching batch100_shared.py's convention).

Builds a synthetic continuation with a KNOWN, EXPLICIT 20-pair lead-lag structure (rho=0.25) on top
of the real market's calibrated beta/idio-noise/ALGO-vol process, then breaks the structure at day
nt_pre+1 two ways:
  - "reverse": same 20 (leader, follower) pairs, coefficient sign flipped.
  - "rotate":  same 20 followers, each reassigned a genuinely NEW random leader.

This directly tests a user-raised concern: this book's ridge ensemble and pairwise boost re-learn
purely from price history, with no pooled detector -- how fast do they actually adapt to a real
structural break, and what does that cost? See algopart3/README.md's change-point section and
SAFE_llboost_v11.py's docstring for the results.
"""
import numpy as np, pandas as pd

ALGODIR = "/home/hideandseekchampion/SIG2026/ArbitrageVictims2026/Victims2026/algopart3"
nInst = 51
nStock = 50
RHO = 0.25
N_PAIRS = 20

P_real = pd.read_csv(f"{ALGODIR}/prices.txt", sep=r"\s+", header=0).values.T.astype(float)
logp_real = np.log(P_real)
r_real = np.diff(logp_real, axis=1)
beta = np.clip(np.array([np.polyfit(r_real[0], r_real[k], 1)[0] for k in range(nInst)]), 0.0, 3.0)
sigma_e = (r_real - np.outer(beta, r_real[0])).std(axis=1)

VOL_WIN, VOL_Z = 20, 60


def roll_std(x, w):
    c1 = np.concatenate(([0.0], np.cumsum(x))); c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s = c1[w:] - c1[:-w]; s2 = c2[w:] - c2[:-w]
    m = s / w; v = np.maximum(s2 / w - m * m, 0.0)
    return np.sqrt(v)


vol_real = np.full(r_real.shape[1], np.nan)
vol_real[VOL_WIN - 1:] = roll_std(r_real[0], VOL_WIN)
lv = np.log(vol_real[~np.isnan(vol_real)] + 1e-9)
phi_v = float(np.clip(np.polyfit(lv[:-1], lv[1:], 1)[0], 0.80, 0.995))
resid_v = lv[1:] - phi_v * lv[:-1] - (1 - phi_v) * lv.mean()
sig_innov = resid_v.std(); mean_log_vol = lv.mean()
volz_real = np.full(r_real.shape[1], np.nan)
for s in range(VOL_WIN + VOL_Z, r_real.shape[1]):
    wv = vol_real[s - VOL_Z:s]
    volz_real[s] = (vol_real[s] - wv.mean()) / (wv.std() + 1e-12)
ok = ~np.isnan(volz_real[:-1])
lam_vm = float(np.polyfit(volz_real[:-1][ok], r_real[0][1:][ok], 1)[0])
mu_m = float(r_real[0].mean())

rng0 = np.random.default_rng(42)
followers = rng0.choice(nStock, size=N_PAIRS, replace=False)
leaders_old = np.array([rng0.choice([i for i in range(nStock) if i != j]) for j in followers])


def build_W(sign=1.0, rotate=False, seed=0):
    W = np.zeros((nStock, nStock))
    if rotate:
        rng2 = np.random.default_rng(seed)
        new_leaders = np.array([rng2.choice([i for i in range(nStock) if i != j]) for j in followers])
        for j, i in zip(followers, new_leaders):
            W[j, i] = RHO
        return W, new_leaders
    for j, i in zip(followers, leaders_old):
        W[j, i] = sign * RHO
    return W, leaders_old


W_old, _ = build_W(sign=1.0)


def simulate(nt_pre, nt_post, mode, seed):
    """Returns (out, idio, algo_ret, W_new, leaders_new). `out` is the (51, nt_pre+nt_post) price
    panel; `idio` is the noise-free idiosyncratic-component ground truth (for the oracle)."""
    rng = np.random.default_rng(seed)
    nt_total = nt_pre + nt_post
    if mode == "reverse":
        W_new, leaders_new = build_W(sign=-1.0)
    elif mode == "rotate":
        W_new, leaders_new = build_W(rotate=True, seed=seed + 777)
    else:
        raise ValueError(mode)

    logm = np.zeros(nt_total); lv_path = np.zeros(nt_total); lv_path[0] = mean_log_vol
    algo_ret = np.zeros(nt_total)
    for t in range(1, nt_total):
        lv_path[t] = phi_v * lv_path[t - 1] + (1 - phi_v) * mean_log_vol + rng.normal(0, sig_innov)
        vol_t = np.exp(lv_path[t]); vol_hist = np.exp(lv_path[max(0, t - VOL_Z):t])
        volz_t = (vol_t - vol_hist.mean()) / (vol_hist.std() + 1e-9) if len(vol_hist) > 5 else 0.0
        drift = mu_m + lam_vm * volz_t
        algo_ret[t] = drift + rng.normal(0, vol_t)
        logm[t] = logm[t - 1] + algo_ret[t]

    idio = np.zeros((nStock, nt_total))
    for t in range(1, nt_total):
        W_t = W_old if t <= nt_pre else W_new
        idio[:, t] = W_t @ idio[:, t - 1] + rng.normal(0, 1, nStock) * sigma_e[1:]

    stock_ret = beta[1:][:, None] * algo_ret[None, :] + idio
    out = np.zeros((nInst, nt_total))
    out[0, :] = P_real[0, 0] * np.exp(logm)
    out[1:, :] = P_real[1:, 0][:, None] * np.exp(np.cumsum(stock_ret, axis=1))
    return out, idio, algo_ret, W_new, leaders_new
