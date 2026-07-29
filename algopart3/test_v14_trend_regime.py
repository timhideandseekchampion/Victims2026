"""
test_v14_trend_regime.py -- the test that actually validates (or kills) v14's Part B insurance
layer, as opposed to test_v14_changepoint.py's reverse/rotate harness (which only breaks a PAIRWISE
lead-lag structure, with no market-wide trend factor -- confirmed this session to give mom/momJT/
residMom zero measurable edge, by construction, not because the layer is broken).

Adapted from algopart2/stress_momentum.py's genuine trend-injection generator (real idio vol,
index kept flat, injects an actual cross-sectional "winners keep winning" momentum regime), ported
to algopart3's 51-instrument/50-idio-name universe and its real prices.txt (1000 days, vs
algopart2's 750). Three regimes, same convention as the original:
  momentum  winners keep winning (cross-sectional trend continuation)
  flip      alternates momentum/reversion every `period` days (whipsaw)
  noise     no cross-sectional predictability at all (edge just dies)

Compares SAFE_llboost_v10 (no insurance) vs SAFE_llboost_v14 (momentum/xsac insurance layer) idio-only
cumulative PnL over the injected extension window, plus a pure cross-sectional momentum reference
book (upper bound on what a book fully dedicated to the regime could earn) for context.

Run: python3 test_v14_trend_regime.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v14 as V14

P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nDays = P_real.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0


def make_ext(kind, T_ext=150, mom=0.6, period=25, K=5, seed=1):
    rng = np.random.default_rng(seed)
    logp = np.log(P_real).copy()
    vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:, :].copy()
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        if kind == "noise":
            drift = np.zeros(50)
        elif kind == "flip":
            sgn = 1.0 if (step // period) % 2 == 0 else -1.0
            drift = sgn * mom * (tc / (tc.std() + 1e-9)) * vol
        else:
            drift = mom * (tc / (tc.std() + 1e-9)) * vol
        drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :nDays] = P_real
    return full


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


def momentum_pos(P, K=5, REV_W_like=10):
    """Pure cross-sectional momentum reference book (upper-bound context, not a real candidate)."""
    P = np.asarray(P, float); ni, t = P.shape; cur = P[:, -1]; pos = np.zeros(ni)
    if t < V10.WARMUP:
        return pos.astype(int)
    lp = np.log(P)
    trail = lp[1:, -1] - lp[1:, -1 - REV_W_like]
    wz = trail - trail.mean()
    lim = (dlr / cur).astype(int)
    pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
    return np.clip(pos, -lim, lim).astype(int)


def walk_idio(mod, full, S, E, track_v14=False):
    POS = np.zeros((nInst, full.shape[1]))
    non_champ = 0; kill_days = 0
    for t in range(mod.WARMUP, E):
        p = np.asarray(mod.getMyPosition(full[:, :t + 1]))
        POS[1:, t] = p[1:]
        if track_v14 and t >= S:
            ready = t >= V14.WARMUP + V14.ROT_W + max(V14.ROT_P, V14.KILL_P)
            if ready:
                c = V14._choose(t)
                if c != "champ":
                    non_champ += 1
                if V14._kill(t, c):
                    kill_days += 1
    return POS, non_champ, kill_days


def momentum_walk(full, S, E):
    POS = np.zeros((nInst, full.shape[1]))
    for t in range(V10.WARMUP, E):
        POS[:, t] = momentum_pos(full[:, :t + 1])
    return POS


if __name__ == "__main__":
    T_EXT = 150
    S, E = nDays, nDays + T_EXT
    for kind in ("momentum", "flip", "noise"):
        full = make_ext(kind, T_ext=T_EXT)
        r_ext = np.diff(np.log(full[1:]), axis=1)[:, nDays:]
        ac = np.mean([np.corrcoef(r_ext[:, tt - 1], r_ext[:, tt])[0, 1] for tt in range(1, r_ext.shape[1])])
        print(f"\n===== {kind.upper()}  (injected-window lag-1 xsectional autocorr {ac:+.3f}) =====")

        reset_module_state(V10)
        POS10, _, _ = walk_idio(V10, full, S, E)

        reset_module_state(V14)
        POS14, non_champ, kill_days = walk_idio(V14, full, S, E, track_v14=True)

        POSmom = momentum_walk(full, S, E)

        pnl10 = daily_pnl_idio(POS10, full, S, E)
        pnl14 = daily_pnl_idio(POS14, full, S, E)
        pnlmom = daily_pnl_idio(POSmom, full, S, E)

        print(f"{'book':<12}{'cumPnL':>12}")
        print(f"{'v10 (none)':<12}{pnl10.sum():12.0f}")
        print(f"{'v14 (ins.)':<12}{pnl14.sum():12.0f}")
        print(f"{'pure_mom':<12}{pnlmom.sum():12.0f}   (reference upper bound, not a real candidate)")
        print(f"v14 non-champ days: {non_champ}/{E-S}   v14 kill days: {kill_days}/{E-S}")
