"""
stress_battery3.py -- a THIRD stress battery, four new scenario types not covered by either prior
battery (stress_next500.py: bootstrap/momentum/flip/noise; stress_broad_battery.py: index_trend/
vol_spike/fast_whipsaw/cluster_shift):

  corr_breakdown   a random 40% of stocks have ALL their historical relationships (beta to index,
                   mutual lead-lag) severed -- generated as independent noise at their own
                   historical vol going forward, while the rest of the market continues via
                   bootstrap-resampled real days. The mirror image of cluster_shift (which adds a
                   NEW correlation instead of removing existing ones) -- tests whether the boost's
                   full-history-fit leader relationships hurt when they suddenly stop being real,
                   arguably as plausible a risk as new correlations forming.
  fat_tail_jumps   background = bootstrap-resampled real days; each stock independently has a small
                   daily chance of an ADDITIONAL fat-tailed (Student-t, df=3) idiosyncratic jump,
                   deliberately with NO engineered reversal afterward -- tests whether the post-jump
                   fade mechanism (which assumes jumps mean-revert) loses money when a meaningful
                   fraction of large moves are actually permanent repricings, not overreactions.
  single_shock     background = bootstrap-resampled real days; on ONE random day within the window,
                   a large simultaneous market-wide shock (-10%, with some idiosyncratic dispersion)
                   hits every stock at once -- tests robustness to a single acute event rather than
                   a sustained multi-day regime.
  compound         the existing momentum-regime generator, but with volatility ALSO scaled up
                   (1.8x) at the same time -- real regime shifts rarely arrive as one clean,
                   isolated textbook type.

3 seeds per scenario. Candidates: v10, v12, v15 (the three that matter for the submission decision --
v16/v17/v18 already established as dominated or narrow-niche, not re-tested here to keep this
tractable, easy to add if wanted). Same cache-snapshot-and-restore efficiency trick and scoring
convention (score(), max_drawdown()) as the two prior batteries, so results are directly comparable.

Run: python3 stress_battery3.py <candidate_module_name>
"""
import sys, copy
import numpy as np, pandas as pd
import importlib

CANDIDATE = sys.argv[1] if len(sys.argv) > 1 else "SAFE_llboost_v15"
MOD = importlib.import_module(CANDIDATE)

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
CACHE_ATTRS = ("_SIG", "_FB", "_RET", "_XC", "_PN", "_ICD")
SEEDS = [301, 302, 303]


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


def make_corr_breakdown(P_real, T_ext, seed, breakdown_frac=0.4):
    rng = np.random.default_rng(seed)
    nInst_local = P_real.shape[0]
    logp = np.log(P_real)
    r = np.diff(logp, axis=1)
    idio_vol = r[1:].std(axis=1)
    n_break = int(round(breakdown_frac * (nInst_local - 1)))
    broken = rng.choice(np.arange(1, nInst_local), size=n_break, replace=False)
    idx = rng.integers(0, r.shape[1], size=T_ext)
    r_ext = r[:, idx].copy()
    for k in broken:
        r_ext[k] = rng.normal(0, idio_vol[k - 1], size=T_ext)
    logp_ext = np.concatenate([logp, logp[:, -1:] + np.cumsum(r_ext, axis=1)], axis=1)
    full = np.exp(logp_ext)
    full[:, :P_real.shape[1]] = P_real
    return full


def make_fat_tail_jumps(P_real, T_ext, seed, jump_prob=0.04, jump_df=3, jump_scale=4.0):
    rng = np.random.default_rng(seed)
    nInst_local = P_real.shape[0]
    logp = np.log(P_real)
    r = np.diff(logp, axis=1)
    idio_vol = r[1:].std(axis=1)
    idx = rng.integers(0, r.shape[1], size=T_ext)
    r_ext = r[:, idx].copy()
    t_std = np.sqrt(jump_df / (jump_df - 2))
    for k in range(1, nInst_local):
        jump_days = rng.random(T_ext) < jump_prob
        n_jumps = int(jump_days.sum())
        if n_jumps:
            jumps = rng.standard_t(jump_df, size=n_jumps) * idio_vol[k - 1] * jump_scale / t_std
            r_ext[k, jump_days] += jumps
    logp_ext = np.concatenate([logp, logp[:, -1:] + np.cumsum(r_ext, axis=1)], axis=1)
    full = np.exp(logp_ext)
    full[:, :P_real.shape[1]] = P_real
    return full


def make_single_shock(P_real, T_ext, seed, shock_size=-0.10):
    rng = np.random.default_rng(seed)
    nInst_local = P_real.shape[0]
    logp = np.log(P_real)
    r = np.diff(logp, axis=1)
    idio_vol = r[1:].std(axis=1)
    idx = rng.integers(0, r.shape[1], size=T_ext)
    r_ext = r[:, idx].copy()
    shock_day = int(rng.integers(20, T_ext - 20))
    shock = np.full(nInst_local, np.log(1 + shock_size))
    shock[1:] += rng.normal(0, idio_vol, nInst_local - 1) * 0.3
    r_ext[:, shock_day] = shock
    logp_ext = np.concatenate([logp, logp[:, -1:] + np.cumsum(r_ext, axis=1)], axis=1)
    full = np.exp(logp_ext)
    full[:, :P_real.shape[1]] = P_real
    return full


def make_compound(P_real, T_ext, seed, mom=0.6, vol_mult=1.8, K=5):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy()
    for _ in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        drift = mom * (tc / (tc.std() + 1e-9)) * vol
        drift -= drift.mean()
        noise = rng.normal(0, vol * vol_mult, names.shape[0]); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
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
        "corr_breakdown": make_corr_breakdown,
        "fat_tail_jumps": make_fat_tail_jumps,
        "single_shock": make_single_shock,
        "compound": make_compound,
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
    print(f"{'scenario':<16}{'cumPnL':>10}{'score':>8}{'worstDD':>10}")
    for name, (cums, scores, dds) in all_results.items():
        print(f"{name:<16}{np.mean(cums):>10.0f}{np.mean(scores):>8.1f}{min(dds):>10.0f}")
