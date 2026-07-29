"""
test_v11_changepoint.py -- validates SAFE_llboost_v11's kill switch against the controlled
change-point experiment that motivated it (see SAFE_llboost_v11.py's docstring and README.md).

Uses changepoint_synthetic.py to break a known 20-pair lead-lag structure at day nt_pre+1, two ways
("reverse": sign flip; "rotate": new leader assignment), across 4 seeds each, and compares:
  - v10 (no protection) idio-only cumulative PnL over the post-change window
  - v11 (kill switch) idio-only cumulative PnL, same window
  - a perfect-foresight oracle (knows the post-change truth immediately)

REFERENCE (already established this session, not re-derived here every run):
  - An IC-significance kill trigger (ported verbatim from SAFE_lldollar._kill: t < -3.0 sustained 10
    consecutive days, ROT_W=60) was tried FIRST. Verified safe (0/904 real-data false positives) but
    weak: ~15% of the reverse-scenario loss and ~0% of the rotate-scenario loss recovered. Diagnosis:
    a rotation degrades the old relationship to near-zero noise, not a confidently negative IC -- a
    significance test can't reliably catch "edge is gone", only "edge is actively hostile".
  - This is why v11 ships with a PnL-sum trigger instead (ROT_W=60, KILL_P=1, no persistence delay),
    matching the lesson already adopted in algopart2/SAFE_rotate.py + SAFE_live.py's own gate
    (memory: "a more sensitive switch than the old IC-significance gate... captures a real regime
    far faster"). This file validates THAT design.

Run: python3 test_v11_changepoint.py   (~4-5 min; each panel walk-forward is O(nt) ridge+boost fits)
"""
import numpy as np
from changepoint_synthetic import simulate, W_old
import SAFE_llboost_v10 as V10
import SAFE_llboost_v11 as V11

commRate = 1e-4
NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]


def reset_module_state(mod):
    for name in ("_SIG", "_RET", "_ICD", "_PN"):
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


def walk_pos_idio(mod, out, nt_pre):
    """POS[:, t] = position decided using info through day t (prc has columns 0..t, i.e. t+1
    columns) -- this is the convention daily_pnl_idio (and batch100_versions_shared.py's proven
    wscore) expects: POS[:,t] gets applied to the day t -> t+1 price move. Storing at t-1 instead
    (an earlier bug in this file) silently shifts every position one day off and corrupts the score
    -- verified concretely: on real prices.txt, this convention reproduces the documented 871.0/912.6
    exactly; the off-by-one version produced strongly negative garbage scores instead."""
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


def run_one(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V10)
    POS10, _ = walk_pos_idio(V10, out, NT_PRE)

    reset_module_state(V11)
    POS11, kill_days_post = walk_pos_idio(V11, out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl11 = daily_pnl_idio(POS11, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum10, cum11, cum_oracle = np.cumsum(pnl10)[-1], np.cumsum(pnl11)[-1], np.cumsum(pnl_oracle)[-1]
    loss10, loss11 = cum_oracle - cum10, cum_oracle - cum11
    frac_saved = (loss10 - loss11) / max(1.0, loss10)
    print(f"  mode={mode:8s} seed={seed}: v10={cum10:9.0f}  v11={cum11:9.0f}  oracle={cum_oracle:9.0f}  "
          f"loss10={loss10:9.0f}  loss11={loss11:9.0f}  frac_saved={frac_saved*100:5.1f}%  "
          f"kill_days_post={kill_days_post}/{nt-NT_PRE-1}")
    return frac_saved


if __name__ == "__main__":
    print("=== real-data sanity check: v11 must be byte-identical to v10 (0 kill days) ===")
    import pandas as pd
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    reset_module_state(V10); reset_module_state(V11)
    mism, kd = 0, 0
    for t in range(V10.WARMUP, P_.shape[1]):
        prc = P_[:, :t + 1]
        p10 = np.asarray(V10.getMyPosition(prc)); p11 = np.asarray(V11.getMyPosition(prc))
        if not np.array_equal(p10, p11):
            mism += 1
        if t >= V11.WARMUP + V11.ROT_W + V11.KILL_P and V11._kill(t):
            kd += 1
    print(f"  mismatches={mism}/{P_.shape[1]-V10.WARMUP}  kill_days={kd}  (expect 0, 0)\n")

    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_one(mode, seed))
    print("\n=== summary ===")
    for mode in ("reverse", "rotate"):
        vals = [v * 100 for v in results[mode]]
        print(f"{mode}: frac_saved_by_kill per seed = {[round(v,1) for v in vals]}  "
              f"mean={np.mean(vals):.1f}%")
