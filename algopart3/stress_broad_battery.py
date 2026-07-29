"""
stress_broad_battery.py -- a SECOND, genuinely different stress battery, built specifically to check
for confirmation bias in the first one (stress_next500.py: bootstrap/momentum/flip/noise). Four NEW
scenario types, none of which are parameter variations of what was already tested favorably for v15:

  index_trend    a sustained INDEX-level bull/bear drift (instrument 0), idio names keep their real
                 historical betas to the index -- stresses the ALGO leg's own vol-timing/momentum
                 signal and the net-$ skew gate, NOT the idio cross-section. Nothing tested so far
                 touched the index's own drift.
  vol_spike      real historical cross-sectional covariance structure, but IID day-to-day (like
                 bootstrap) with volatility scaled 2.5x -- tests pure vol-regime robustness,
                 independent of any directional signal, a different axis than momentum/flip/noise.
  fast_whipsaw   the SAME flip generator already used, but period=10 instead of 25 -- more
                 adversarial than what was tested, to see if the pattern gets worse, better, or
                 different at a different oscillation frequency.
  cluster_shift  a random 10-stock subset suddenly shares a common factor not present in real
                 history (correlation ~0.6 within the cluster) -- tests whether the lead-lag boost
                 misattributes new cluster co-movement as genuine leader-follower structure, a
                 structural stress no prior scenario covered.

3 seeds per scenario (not 1), so this reports a distribution, not a single point estimate. Candidates
tested: v10, v12, v15, v16 -- v17/v18 deliberately excluded, NOT to hide a result but because they
were already shown strictly dominated by v16 on every prior axis (reverse/rotate/momentum/flip/noise
all equal-or-worse, never better, across two separate test cycles); re-including them here would
just spend compute re-confirming that rather than genuinely testing new ground.

Uses the same cache-snapshot-and-restore trick as stress_next500.py for efficiency, and the SAME
scoring convention (score(), max_drawdown()) so results are directly comparable across both batteries.

Run: python3 stress_broad_battery.py <candidate_module_name>
"""
import sys, copy
import numpy as np, pandas as pd
import importlib

CANDIDATE = sys.argv[1] if len(sys.argv) > 1 else "SAFE_llboost_v15"
MOD = importlib.import_module(CANDIDATE)

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
CACHE_ATTRS = ("_SIG", "_FB", "_RET", "_XC", "_PN", "_ICD")
SEEDS = [201, 202, 203]


def reset_module_state(mod):
    for name in CACHE_ATTRS:
        if hasattr(mod, name):
            getattr(mod, name).clear()
    if hasattr(mod, "_PREV_ALGO_SHARES"):
        mod._PREV_ALGO_SHARES = 0
    if hasattr(mod, "_PREV_T"):
        mod._PREV_T = -1
    if hasattr(mod, "_DLR"):
        mod._DLR = None


def snapshot(mod):
    snap = {}
    for name in CACHE_ATTRS:
        if hasattr(mod, name):
            snap[name] = copy.deepcopy(getattr(mod, name))
    snap["_PREV_ALGO_SHARES"] = getattr(mod, "_PREV_ALGO_SHARES", 0)
    snap["_PREV_T"] = getattr(mod, "_PREV_T", -1)
    snap["_DLR"] = copy.deepcopy(getattr(mod, "_DLR", None))
    return snap


def restore(mod, snap):
    for name in CACHE_ATTRS:
        if name in snap:
            d = getattr(mod, name)
            d.clear(); d.update(copy.deepcopy(snap[name]))
    if hasattr(mod, "_PREV_ALGO_SHARES"):
        mod._PREV_ALGO_SHARES = snap["_PREV_ALGO_SHARES"]
    if hasattr(mod, "_PREV_T"):
        mod._PREV_T = snap["_PREV_T"]
    if hasattr(mod, "_DLR"):
        mod._DLR = snap["_DLR"]


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def daily_pnl_idio(POS, full, S, E):
    nInst = full.shape[0]
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = full[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos[1:] * (cur[1:] - prevCur[1:])).sum()) - float(comm_vec[1:].sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    return np.array(tot)


def max_drawdown(pnl):
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def make_index_trend(P_real, T_ext, seed):
    rng = np.random.default_rng(seed)
    logp0 = np.log(P_real)
    r = np.diff(logp0, axis=1)
    nInst_local = P_real.shape[0]
    beta = np.clip(np.array([np.polyfit(r[0], r[k], 1)[0] for k in range(nInst_local)]), 0.0, 3.0)
    sigma_e = (r - np.outer(beta, r[0])).std(axis=1)
    idx_vol = r[0].std()
    drift = 0.3 * idx_vol * (1 if seed % 2 == 0 else -1)  # alternate bull/bear by seed for variety
    logp = logp0.copy()
    cur = logp[:, -1].copy()
    for _ in range(T_ext):
        idx_ret = drift + rng.normal(0, idx_vol)
        new = cur + beta * idx_ret + rng.normal(0, sigma_e)
        new[0] = cur[0] + idx_ret
        cur = new
        logp = np.concatenate([logp, cur[:, None]], axis=1)
    full = np.exp(logp)
    full[:, :P_real.shape[1]] = P_real
    return full


def make_vol_spike(P_real, T_ext, seed, vol_mult=2.5):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real)
    r = np.diff(logp, axis=1)
    cov = np.cov(r)
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(cov.shape[0]))
    z = rng.standard_normal((T_ext, cov.shape[0]))
    r_ext = (z @ L.T).T * vol_mult
    logp_ext = np.concatenate([logp, logp[:, -1:] + np.cumsum(r_ext, axis=1)], axis=1)
    full = np.exp(logp_ext)
    full[:, :P_real.shape[1]] = P_real
    return full


def make_fast_whipsaw(P_real, T_ext, seed, mom=0.6, period=10, K=5):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy()
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        sgn = 1.0 if (step // period) % 2 == 0 else -1.0
        drift = sgn * mom * (tc / (tc.std() + 1e-9)) * vol
        drift -= drift.mean()
        noise = rng.normal(0, vol, names.shape[0]); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :P_real.shape[1]] = P_real
    return full


def make_cluster_shift(P_real, T_ext, seed, cluster_size=10, cluster_corr=0.6):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    nInst_local = P_real.shape[0]
    r = np.diff(logp, axis=1)
    idio_vol = r[1:].std(axis=1)
    idx_vol = r[0].std()
    cluster = rng.choice(np.arange(1, nInst_local), size=cluster_size, replace=False)
    common_vol = float(idio_vol[cluster - 1].mean())
    cur = logp[:, -1].copy()
    for _ in range(T_ext):
        common = rng.normal(0, common_vol)
        drift = np.zeros(nInst_local)
        for k in range(1, nInst_local):
            if k in cluster:
                drift[k] = cluster_corr * common + np.sqrt(max(0.0, 1 - cluster_corr ** 2)) * rng.normal(0, idio_vol[k - 1])
            else:
                drift[k] = rng.normal(0, idio_vol[k - 1])
        drift[0] = rng.normal(0, idx_vol)
        cur = cur + drift
        logp = np.concatenate([logp, cur[:, None]], axis=1)
    full = np.exp(logp)
    full[:, :P_real.shape[1]] = P_real
    return full


if __name__ == "__main__":
    P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nDays = P_real.shape
    T_EXT = 150

    print(f"=== {CANDIDATE}: building real-data prefix cache (day {MOD.WARMUP}-{nDays-1}) ===")
    reset_module_state(MOD)
    for t in range(MOD.WARMUP, nDays):
        MOD.getMyPosition(P_real[:, :t + 1])
    snap = snapshot(MOD)
    print("  prefix cache built and snapshotted.\n")

    GENERATORS = {
        "index_trend": make_index_trend,
        "vol_spike": make_vol_spike,
        "fast_whipsaw": make_fast_whipsaw,
        "cluster_shift": make_cluster_shift,
    }

    S, E = nDays, nDays + T_EXT
    all_results = {}
    for name, gen in GENERATORS.items():
        cums, scores, dds = [], [], []
        for seed in SEEDS:
            full = gen(P_real, T_EXT, seed)
            restore(MOD, snap)
            POS = np.zeros((nInst, full.shape[1]))
            for t in range(S, E):
                p = np.asarray(MOD.getMyPosition(full[:, :t + 1]))
                POS[1:, t] = p[1:]
            pnl = daily_pnl_idio(POS, full, S, E)
            sc = score(pnl.mean(), pnl.std())
            dd = max_drawdown(pnl)
            cums.append(float(pnl.sum())); scores.append(sc); dds.append(dd)
            print(f"  {name:14s} seed={seed}: cumPnL={pnl.sum():9.0f}  score={sc:7.1f}  maxDD={dd:9.0f}")
        all_results[name] = (cums, scores, dds)
        print(f"  {name:14s} MEAN: cumPnL={np.mean(cums):9.0f}  score={np.mean(scores):7.1f}  "
              f"worstDD={min(dds):9.0f}\n")

    print(f"=== {CANDIDATE} summary (mean over {len(SEEDS)} seeds, worst-case drawdown) ===")
    print(f"{'scenario':<14}{'cumPnL':>10}{'score':>8}{'worstDD':>10}")
    for name, (cums, scores, dds) in all_results.items():
        print(f"{name:<14}{np.mean(cums):>10.0f}{np.mean(scores):>8.1f}{min(dds):>10.0f}")
