"""
test_v14_reverse_mechanism.py -- digs into the reverse-change-point surprise from
test_v14_changepoint.py: v14 recovers MORE of the oracle gap than v13 in reverse mode (28.0% vs
25.0% mean across 4 seeds), even though v14's own docstring pre-registered an expectation that the
momentum/xsac insurance layer should be a WASH on this harness (a plain-numpy pre-check found mom/
momJT/residMom statistically indistinguishable from noise against this specific pairwise-break
generator). test_v14_changepoint.py's own non_champ_days count (39-47% of post-change days in
reverse mode) shows the insurance layer is clearly NOT inert here, contradicting that expectation.

QUESTION: does the fallback signal itself carry real edge in this scenario, or does ANY departure
from a known actively-wrong-signed champion help, regardless of what you switch to -- i.e. is
"flatten whenever _choose picks a fallback" just as good as "trade whichever fallback _choose
picked"?

METHOD: re-run the same reverse change-point walk for v14 (same generator, same 4 seeds), but build
a COUNTERFACTUAL position series identical to v14's actual output on every day EXCEPT days where
`_choose` picked a non-champ signal AND `_kill` did not already flatten it -- on those specific
days, replace the traded position with zero (flatten) instead of the fallback signal's sizing.
Every other day (chosen=="champ", or killed=True) is untouched by construction, since kill already
flattens and champ days are identical either way. Comparing cumulative PnL isolates the causal
contribution of "trading the actual fallback signal" from "merely not trading champ on those days."

Run: python3 test_v14_reverse_mechanism.py
"""
import numpy as np
from changepoint_synthetic import simulate, W_old
import SAFE_llboost_v14 as V14

NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]
commRate = 1e-4


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


def walk_v14_with_counterfactual(out, nt_pre):
    """Returns (actual POS, flatten-fallback-days counterfactual POS, count of fallback-traded days,
    count of kill days, count of non-champ days)."""
    nInst, nt = out.shape
    POS_actual = np.zeros((nInst, nt))
    POS_cf = np.zeros((nInst, nt))
    fallback_traded_days = 0
    kill_days = 0
    non_champ_days = 0
    for t in range(V14.WARMUP, nt):
        p = np.asarray(V14.getMyPosition(out[:, :t + 1]))
        POS_actual[1:, t] = p[1:]
        ready = t >= V14.WARMUP + V14.ROT_W + max(V14.ROT_P, V14.KILL_P)
        if ready:
            chosen = V14._choose(t)
            killed = V14._kill(t, chosen)
            if chosen != "champ":
                non_champ_days += 1
            if killed:
                kill_days += 1
            if chosen != "champ" and not killed:
                POS_cf[1:, t] = 0.0     # counterfactual: flatten instead of trading the fallback
                fallback_traded_days += 1
            else:
                POS_cf[1:, t] = p[1:]   # identical to actual (champ day, or already-killed/flat)
        else:
            POS_cf[1:, t] = p[1:]
    return POS_actual, POS_cf, fallback_traded_days, kill_days, non_champ_days


def run_one(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V14)
    POS_actual, POS_cf, fb_days, kill_days, non_champ = walk_v14_with_counterfactual(out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl_actual = daily_pnl_idio(POS_actual, out, NT_PRE + 1, nt - 1)
    pnl_cf = daily_pnl_idio(POS_cf, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum_actual = np.cumsum(pnl_actual)[-1]
    cum_cf = np.cumsum(pnl_cf)[-1]
    cum_oracle = np.cumsum(pnl_oracle)[-1]

    print(f"  seed={seed}: v14_actual={cum_actual:9.0f}  flatten_fallback_cf={cum_cf:9.0f}  "
          f"oracle={cum_oracle:9.0f}  delta(actual-cf)={cum_actual-cum_cf:9.0f}  "
          f"fallback_traded_days={fb_days}  kill_days={kill_days}  non_champ_days={non_champ}"
          f"  (post-change window={nt-NT_PRE-1} days)")
    return cum_actual, cum_cf, fb_days


if __name__ == "__main__":
    print("=== reverse mode only: does trading the fallback signal beat just flattening those days? ===")
    results = []
    for seed in SEEDS:
        results.append(run_one("reverse", seed))
    actual = np.array([r[0] for r in results])
    cf = np.array([r[1] for r in results])
    print(f"\nmean v14_actual={actual.mean():.0f}  mean flatten_fallback_cf={cf.mean():.0f}  "
          f"mean delta(actual-cf)={(actual-cf).mean():.0f}")
    wins = int((actual > cf).sum())
    print(f"v14_actual beats flatten-only counterfactual in {wins}/{len(SEEDS)} seeds "
          f"(>0 delta means the traded fallback signal adds value beyond just not trading champ;"
          f" <=0 means flattening alone would have done as well or better)")
