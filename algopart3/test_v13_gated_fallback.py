"""
test_v13_gated_fallback.py -- validates SAFE_llboost_v13's gated decayed-selection boost fallback.

Two checks:
  1. Real prices.txt: how often does the fallback actually engage vs v11, and what's the net effect
     on score (OLD/NEW/rmean/rfloor, n_worse-of-61)?
  2. The same corrected change-point experiment used for v11 (changepoint_synthetic.py,
     test_v11_changepoint.py's indexing convention): does v13 recover more of the rotate-scenario
     oracle gap than v10/v11, without regressing the reverse scenario?

Run: python3 test_v13_gated_fallback.py
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10
import SAFE_llboost_v11 as V11
import SAFE_llboost_v13 as V13
from changepoint_synthetic import simulate, W_old

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0
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


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, P_, S, E, nInst):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


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
    """POS[:, t] convention -- see test_v11_changepoint.py's note on why this (not t-1) is correct."""
    nInst, nt = out.shape
    POS = np.zeros((nInst, nt))
    for t in range(mod.WARMUP, nt):
        p = np.asarray(mod.getMyPosition(out[:, :t + 1]))
        POS[1:, t] = p[1:]
    return POS


def real_data_check():
    print("=== 1. real prices.txt: v13 vs v11 ===")
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nt = P_.shape
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

    def build(mod):
        reset_module_state(mod)
        POS = np.zeros((nInst, nt))
        for t in range(1, nt):
            prcSoFar = P_[:, :t]
            p = np.asarray(mod.getMyPosition(prcSoFar))
            lim = (dlr / prcSoFar[:, -1]).astype(int)
            POS[:, t - 1] = np.clip(p, -lim, lim).astype(int)
        return POS

    POS11 = build(V11)
    POS13 = build(V13)
    curve11 = np.array([wscore(POS11, P_, E - NUMTEST, E, nInst) for E in end_days])
    curve13 = np.array([wscore(POS13, P_, E - NUMTEST, E, nInst) for E in end_days])
    n_worse = int((curve13 < curve11).sum()); n_better = int((curve13 > curve11).sum())
    mism = int((~np.all(POS11 == POS13, axis=0)).sum())
    print(f"  v11: OLD={wscore(POS11,P_,500,750,nInst):.1f} NEW={wscore(POS11,P_,750,nt,nInst):.1f} "
          f"rmean={curve11.mean():.1f} rfloor={curve11.min():.1f}")
    print(f"  v13: OLD={wscore(POS13,P_,500,750,nInst):.1f} NEW={wscore(POS13,P_,750,nt,nInst):.1f} "
          f"rmean={curve13.mean():.1f} rfloor={curve13.min():.1f}")
    print(f"  days with any position differing: {mism}/{nt-96}  n_worse={n_worse}/61  n_better={n_better}/61\n")


def run_one(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V10); POS10 = walk_pos_idio(V10, out, NT_PRE)
    reset_module_state(V11); POS11 = walk_pos_idio(V11, out, NT_PRE)
    reset_module_state(V13); POS13 = walk_pos_idio(V13, out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl11 = daily_pnl_idio(POS11, out, NT_PRE + 1, nt - 1)
    pnl13 = daily_pnl_idio(POS13, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum10, cum11, cum13, cum_oracle = (np.cumsum(pnl10)[-1], np.cumsum(pnl11)[-1],
                                        np.cumsum(pnl13)[-1], np.cumsum(pnl_oracle)[-1])
    loss10, loss11, loss13 = cum_oracle - cum10, cum_oracle - cum11, cum_oracle - cum13
    print(f"  mode={mode:8s} seed={seed}: v10={cum10:9.0f}  v11={cum11:9.0f}  v13={cum13:9.0f}  "
          f"oracle={cum_oracle:9.0f}  loss10={loss10:9.0f}  loss11={loss11:9.0f}  loss13={loss13:9.0f}")
    return dict(loss10=loss10, loss11=loss11, loss13=loss13)


if __name__ == "__main__":
    real_data_check()

    print("=== 2. change-point experiment: v10 vs v11 vs v13, both scenarios, 4 seeds ===")
    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_one(mode, seed))

    print("\n=== summary: frac of oracle-gap recovered vs plain v10 ===")
    for mode in ("reverse", "rotate"):
        f11 = [(r["loss10"] - r["loss11"]) / max(1.0, r["loss10"]) * 100 for r in results[mode]]
        f13 = [(r["loss10"] - r["loss13"]) / max(1.0, r["loss10"]) * 100 for r in results[mode]]
        print(f"{mode}: v11 frac_saved per seed = {[round(v,1) for v in f11]}  mean={np.mean(f11):.1f}%")
        print(f"{mode}: v13 frac_saved per seed = {[round(v,1) for v in f13]}  mean={np.mean(f13):.1f}%")
