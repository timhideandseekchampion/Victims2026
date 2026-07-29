"""
test_v19_corr_scenarios.py -- targeted follow-up: does v19's two-hop transitive boost interact
differently than v15's plain boost specifically in the two correlation-structure stress scenarios
(cluster_shift, corr_breakdown) -- the ones most likely to matter for a leader-CHAIN mechanism.
Reuses the exact generators from stress_broad_battery.py / stress_battery3.py, 3 seeds each.

Run: python3 test_v19_corr_scenarios.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v15 as V15
import SAFE_llboost_v19 as V19

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
SEEDS = [201, 202, 203]  # matches stress_broad_battery.py's cluster_shift seeds
SEEDS_CB = [301, 302, 303]  # matches stress_battery3.py's corr_breakdown seeds


def reset_module_state(mod):
    for name in ("_SIG", "_FB", "_RET", "_XC", "_ICD", "_PN"):
        if hasattr(mod, name):
            getattr(mod, name).clear()
    if hasattr(mod, "_PREV_ALGO_SHARES"):
        mod._PREV_ALGO_SHARES = 0
    if hasattr(mod, "_PREV_T"):
        mod._PREV_T = -1
    if hasattr(mod, "_DLR"):
        mod._DLR = None


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


def run_scenario(name, gen, seeds, T_ext=150):
    P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nDays = P_real.shape
    S, E = nDays, nDays + T_ext
    print(f"=== {name} ===")
    r15, r19 = [], []
    for seed in seeds:
        full = gen(P_real, T_ext, seed)
        results = {}
        for tag, mod in (("v15", V15), ("v19", V19)):
            reset_module_state(mod)
            POS = np.zeros((nInst, full.shape[1]))
            for t in range(mod.WARMUP, E):
                p = np.asarray(mod.getMyPosition(full[:, :t + 1]))
                POS[1:, t] = p[1:]
            pnl = daily_pnl_idio(POS, full, S, E)
            results[tag] = (float(pnl.sum()), score(pnl.mean(), pnl.std()), max_drawdown(pnl))
        c15, s15, d15 = results["v15"]; c19, s19, d19 = results["v19"]
        r15.append((c15, s15, d15)); r19.append((c19, s19, d19))
        print(f"  seed={seed}: v15 cumPnL={c15:9.0f} score={s15:7.1f} maxDD={d15:9.0f}   "
              f"v19 cumPnL={c19:9.0f} score={s19:7.1f} maxDD={d19:9.0f}")
    m15 = np.mean([x[0] for x in r15]); m19 = np.mean([x[0] for x in r19])
    dd15 = min(x[2] for x in r15); dd19 = min(x[2] for x in r19)
    print(f"  MEAN cumPnL: v15={m15:.0f}  v19={m19:.0f}   worstDD: v15={dd15:.0f}  v19={dd19:.0f}\n")


if __name__ == "__main__":
    run_scenario("cluster_shift", make_cluster_shift, SEEDS)
    run_scenario("corr_breakdown", make_corr_breakdown, SEEDS_CB)
