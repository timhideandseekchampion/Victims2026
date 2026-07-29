"""
test_v14_changepoint.py -- re-runs the corrected change-point experiment (changepoint_synthetic.py,
same convention as test_v11_changepoint.py) with SAFE_llboost_v14 in place of v10/v11, plus a
flap-rate metric (how often `_choose`'s pick and `_kill`'s verdict change day-to-day) since v14 adds
a 5-day-persistence rotation gate on top of v11's no-persistence kill trigger, and it isn't obvious
a priori whether the two fight each other.

This is expected, per this session's own direct pre-check (mom/momJT/residMom show no edge in this
specific pairwise-break synthetic -- see SAFE_llboost_v14.py's docstring), to be a WASH relative to
v13/v11 here: the insurance layer targets a different failure mode (genuine trend regime) that this
generator doesn't inject. The real test of that layer is test_v14_trend_regime.py. This script's job
is to confirm v14 doesn't REGRESS versus v13's already-validated boost-fallback gain on this harness,
and to measure the flap rate honestly rather than assume it's fine.

Run: python3 test_v14_changepoint.py
"""
import numpy as np
from changepoint_synthetic import simulate, W_old
import SAFE_llboost_v10 as V10
import SAFE_llboost_v13 as V13
import SAFE_llboost_v14 as V14

commRate = 1e-4
NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]


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


def daily_pnl_idio(POS, out, S, E):
    nInst = out.shape[0]
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = out[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    return np.array(tot)


def walk_pos_idio_v10_v13(mod, out, nt_pre):
    nInst, nt = out.shape
    POS = np.zeros((nInst, nt))
    kill_days_post = 0
    for t in range(mod.WARMUP, nt):
        p = np.asarray(mod.getMyPosition(out[:, :t + 1]))
        POS[1:, t] = p[1:]
        if hasattr(mod, "KILL_ON") and t > nt_pre:
            ready = t >= mod.WARMUP + mod.ROT_W + mod.KILL_P
            if ready and mod._kill(t):
                kill_days_post += 1
    return POS, kill_days_post


def walk_pos_idio_v14(out, nt_pre):
    nInst, nt = out.shape
    POS = np.zeros((nInst, nt))
    kill_days_post = 0
    chosen_log = []
    transitions = 0
    prev_state = None
    for t in range(V14.WARMUP, nt):
        p = np.asarray(V14.getMyPosition(out[:, :t + 1]))
        POS[1:, t] = p[1:]
        ready = t >= V14.WARMUP + V14.ROT_W + max(V14.ROT_P, V14.KILL_P)
        if t > nt_pre and ready:
            chosen = V14._choose(t)
            killed = V14._kill(t, chosen)
            state = (chosen, killed)
            chosen_log.append(state)
            if killed:
                kill_days_post += 1
            if prev_state is not None and state != prev_state:
                transitions += 1
            prev_state = state
    return POS, kill_days_post, transitions, chosen_log


def run_one(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V10)
    POS10, _ = walk_pos_idio_v10_v13(V10, out, NT_PRE)

    reset_module_state(V13)
    POS13, kill13 = walk_pos_idio_v10_v13(V13, out, NT_PRE)

    reset_module_state(V14)
    POS14, kill14, transitions, chosen_log = walk_pos_idio_v14(out, NT_PRE)

    non_champ = sum(1 for (c, k) in chosen_log if c != "champ")

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl13 = daily_pnl_idio(POS13, out, NT_PRE + 1, nt - 1)
    pnl14 = daily_pnl_idio(POS14, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum10 = np.cumsum(pnl10)[-1]; cum13 = np.cumsum(pnl13)[-1]
    cum14 = np.cumsum(pnl14)[-1]; cum_oracle = np.cumsum(pnl_oracle)[-1]
    loss10 = cum_oracle - cum10; loss13 = cum_oracle - cum13; loss14 = cum_oracle - cum14
    saved13 = (loss10 - loss13) / max(1.0, loss10)
    saved14 = (loss10 - loss14) / max(1.0, loss10)
    print(f"  mode={mode:8s} seed={seed}: v10={cum10:9.0f}  v13={cum13:9.0f}  v14={cum14:9.0f}  "
          f"oracle={cum_oracle:9.0f}  saved13={saved13*100:5.1f}%  saved14={saved14*100:5.1f}%  "
          f"kill14={kill14}/{nt-NT_PRE-1}  non_champ_days={non_champ}  state_transitions={transitions}")
    return saved13, saved14, transitions, non_champ


if __name__ == "__main__":
    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_one(mode, seed))
    print("\n=== summary ===")
    for mode in ("reverse", "rotate"):
        s13 = [r[0] * 100 for r in results[mode]]
        s14 = [r[1] * 100 for r in results[mode]]
        trans = [r[2] for r in results[mode]]
        nc = [r[3] for r in results[mode]]
        print(f"{mode}: v13 saved% per seed = {[round(v,1) for v in s13]}  mean={np.mean(s13):.1f}%")
        print(f"{mode}: v14 saved% per seed = {[round(v,1) for v in s14]}  mean={np.mean(s14):.1f}%")
        print(f"{mode}: state transitions per seed = {trans}  non_champ days per seed = {nc}")
