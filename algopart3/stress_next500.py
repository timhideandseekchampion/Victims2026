"""
stress_next500.py -- generates several DISTINCT hypothetical continuations of the next 500 days
(matching the tournament's actual qualifier window per README/memory: qualifier = days 1000-1500),
and scores a candidate build across all of them: cumulative idio PnL, the official score() metric,
and max drawdown (a "did it survive" gauge), so multiple candidates can be ranked on which performs
best AND holds up best across a spread of futures we can't actually observe yet -- not just one.

Four scenario families, each = real prices.txt (1000 days) + a 500-day synthetic extension:
  bootstrap  resample WHOLE historical days (with replacement) -- preserves real cross-sectional
             correlation structure for any given day, no NEW systematic regime injected. The
             "nothing structurally changes, business continues as observed" null.
  momentum   cross-sectional trend continuation (winners keep winning), same generator used
             throughout this session's trend-regime tests, extended from 150 to 500 days.
  flip       momentum/reversion alternating every 25 days (whipsaw) -- same generator, 500 days.
  noise      no cross-sectional predictability at all -- edge just dies -- same generator, 500 days.

NOT included here (by design, not oversight): the reverse/rotate pairwise lead-lag change-point
scenarios (changepoint_synthetic.py) test a different, more specific question (a KNOWN 20-pair
lead-lag structure breaking) on a fully-synthetic (not real-prefixed) panel calibrated to real
statistics -- already tested extensively elsewhere this session (test_v14_changepoint.py,
test_v15/v16_*.py). Their numbers are cited alongside this script's output rather than re-run here,
since mixing two incompatible generator families in one battery would cost more compute for no new
information.

EFFICIENCY: for any candidate with a _SIG-style cache (v11+), the real-data prefix (day 96-999) is
walked and cached ONCE, then snapshotted (deep-copied) -- each of the 4 scenarios restores that
snapshot and only pays for the NEW 500 days, instead of redundantly re-deriving the shared real-data
prefix four times. v10 has no such cache (recomputes everything fresh every call by construction) so
it pays the full walk each time regardless -- noted, not a bug in this harness.

Run: python3 stress_next500.py <candidate_module_name>   e.g. python3 stress_next500.py SAFE_llboost_v15
"""
import sys, copy
import numpy as np, pandas as pd
import importlib

CANDIDATE = sys.argv[1] if len(sys.argv) > 1 else "SAFE_llboost_v15"
MOD = importlib.import_module(CANDIDATE)

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
CACHE_ATTRS = ("_SIG", "_FB", "_RET", "_XC", "_PN", "_ICD")


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
    dd = cum - peak
    return float(dd.min())


def make_bootstrap(P_real, T_ext=500, seed=100):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real)
    r = np.diff(logp, axis=1)
    idx = rng.integers(0, r.shape[1], size=T_ext)
    r_ext = r[:, idx]
    logp_ext = np.concatenate([logp, logp[:, -1:] + np.cumsum(r_ext, axis=1)], axis=1)
    full = np.exp(logp_ext)
    full[:, :P_real.shape[1]] = P_real
    return full


def make_regime(P_real, kind, T_ext=500, mom=0.6, period=25, K=5, seed=1):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy()
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        if kind == "noise":
            drift = np.zeros(names.shape[0])
        elif kind == "flip":
            sgn = 1.0 if (step // period) % 2 == 0 else -1.0
            drift = sgn * mom * (tc / (tc.std() + 1e-9)) * vol
        else:
            drift = mom * (tc / (tc.std() + 1e-9)) * vol
        drift -= drift.mean()
        noise = rng.normal(0, vol, names.shape[0]); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :P_real.shape[1]] = P_real
    return full


if __name__ == "__main__":
    P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nDays = P_real.shape
    T_EXT = 500

    print(f"=== {CANDIDATE}: building real-data prefix cache (day {MOD.WARMUP}-{nDays-1}) ===")
    reset_module_state(MOD)
    for t in range(MOD.WARMUP, nDays):
        MOD.getMyPosition(P_real[:, :t + 1])
    snap = snapshot(MOD)
    print("  prefix cache built and snapshotted.\n")

    scenarios = {
        "bootstrap": make_bootstrap(P_real, T_EXT, seed=100),
        "momentum": make_regime(P_real, "momentum", T_EXT, seed=101),
        "flip": make_regime(P_real, "flip", T_EXT, seed=102),
        "noise": make_regime(P_real, "noise", T_EXT, seed=103),
    }

    S, E = nDays, nDays + T_EXT
    results = {}
    for name, full in scenarios.items():
        restore(MOD, snap)
        POS = np.zeros((nInst, full.shape[1]))
        for t in range(S, E):
            p = np.asarray(MOD.getMyPosition(full[:, :t + 1]))
            POS[1:, t] = p[1:]
        pnl = daily_pnl_idio(POS, full, S, E)
        sc = score(pnl.mean(), pnl.std())
        dd = max_drawdown(pnl)
        results[name] = (float(pnl.sum()), sc, dd)
        print(f"  {name:10s}: cumPnL={pnl.sum():9.0f}  score={sc:7.1f}  maxDD={dd:9.0f}")

    print(f"\n=== {CANDIDATE} summary ===")
    print(f"{'scenario':<10}{'cumPnL':>10}{'score':>8}{'maxDD':>10}")
    for name, (cum, sc, dd) in results.items():
        print(f"{name:<10}{cum:>10.0f}{sc:>8.1f}{dd:>10.0f}")
