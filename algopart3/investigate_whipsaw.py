"""
investigate_whipsaw.py -- why does v14's momentum/xsac insurance layer LOSE money vs plain v10 in
the flip/whipsaw regime (144,715 vs 193,893 over the 150-day injected window, test_v14_trend_regime.py)?

The flip generator alternates momentum/reversion every `period=25` days. HYPOTHESIS: the insurance
layer's own detection lag (ROT_W=60-day trailing PnL-sum / XSAC_W=40-day trailing xsac, PLUS
ROT_P=5-day persistence before switching) is comparable to or longer than the 25-day flip period --
so by the time it detects "champion is sick, momentum regime" and switches to a momentum fallback,
the regime may have already flipped back to reversion, meaning the switch is chasing the PREVIOUS
segment rather than anticipating the current one. Track day-by-day chosen/killed state against the
known flip schedule to check this directly, rather than only looking at the aggregate PnL numbers.

Run: python3 investigate_whipsaw.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v14 as V14

P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nDays = P_real.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5


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


def make_ext(kind, T_ext=150, mom=0.6, period=25, K=5, seed=1):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy()
    regime_log = []  # +1 momentum-phase, -1 reversion-phase, 0 pre-injection
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        sgn = 1.0 if (step // period) % 2 == 0 else -1.0
        drift = sgn * mom * (tc / (tc.std() + 1e-9)) * vol
        regime_log.append(sgn)
        drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :nDays] = P_real
    return full, np.array(regime_log)


def daily_pnl_idio(POS, full, S, E):
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


if __name__ == "__main__":
    T_EXT = 150
    period = 25
    full, regime_log = make_ext("flip", T_ext=T_EXT, period=period)
    S, E = nDays, nDays + T_EXT

    reset_module_state(V10)
    POS10 = np.zeros((nInst, full.shape[1]))
    for t in range(V10.WARMUP, E):
        p = np.asarray(V10.getMyPosition(full[:, :t + 1]))
        POS10[1:, t] = p[1:]

    reset_module_state(V14)
    POS14 = np.zeros((nInst, full.shape[1]))
    chosen_log = []; killed_log = []
    for t in range(V14.WARMUP, E):
        p = np.asarray(V14.getMyPosition(full[:, :t + 1]))
        POS14[1:, t] = p[1:]
        ready = t >= V14.WARMUP + V14.ROT_W + max(V14.ROT_P, V14.KILL_P)
        if t >= S and ready:
            c = V14._choose(t)
            k = V14._kill(t, c)
            chosen_log.append(c); killed_log.append(k)
        elif t >= S:
            chosen_log.append("champ"); killed_log.append(False)

    pnl10 = daily_pnl_idio(POS10, full, S, E)
    pnl14 = daily_pnl_idio(POS14, full, S, E)
    print(f"v10 total: {pnl10.sum():.0f}   v14 total: {pnl14.sum():.0f}   delta: {pnl14.sum()-pnl10.sum():.0f}\n")

    # per-25-day-block: regime sign, v10 pnl, v14 pnl, and what fraction of days v14 was non-champ/killed
    n_blocks = T_EXT // period
    print(f"{'block':>6}{'regime':>8}{'v10_pnl':>10}{'v14_pnl':>10}{'delta':>9}{'non_champ%':>12}{'kill%':>8}  chosen (majority)")
    for b in range(n_blocks):
        lo, hi = b * period, (b + 1) * period
        reg = int(regime_log[lo]) if lo < len(regime_log) else 0
        p10 = pnl10[lo:hi].sum() if hi <= len(pnl10) else pnl10[lo:].sum()
        p14 = pnl14[lo:hi].sum() if hi <= len(pnl14) else pnl14[lo:].sum()
        block_chosen = chosen_log[lo:hi]
        block_killed = killed_log[lo:hi]
        nc_frac = 100.0 * sum(c != "champ" for c in block_chosen) / max(1, len(block_chosen))
        k_frac = 100.0 * sum(block_killed) / max(1, len(block_killed))
        vals, counts = np.unique(block_chosen, return_counts=True)
        majority = vals[np.argmax(counts)]
        print(f"{b:>6}{'mom' if reg>0 else 'rev':>8}{p10:>10.0f}{p14:>10.0f}{p14-p10:>9.0f}{nc_frac:>12.1f}{k_frac:>8.1f}  {majority}")
