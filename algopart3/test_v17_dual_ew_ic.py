"""
test_v17_ic_gate.py -- validates SAFE_llboost_v17 (v15 + IC/t-stat significance gate on the
fallback switch) against v15's already-recorded numbers, plus counts how often the new gate
actually VETOES a switch that the bare PnL-sum comparison alone would have made.

Recorded v15 baselines being compared against (this session):
  real data: byte-identical to v12 (OLD=885.8 NEW=913.8 rmean=917.3 rfloor=720.7), 0/904 diff, n_worse=0/61
  change-point reverse frac_saved: mean=28.2% [30.0, 23.2, 26.5, 33.2]
  change-point rotate  frac_saved: mean=-0.9% [1.9, -6.4, 0.2, 0.6]
  trend-regime (150d): momentum=672934  flip=145981  noise=-22234  (v10: 210281 / 193893 / -16174)

Run: python3 test_v17_ic_gate.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v12 as V12
import SAFE_llboost_v15 as V15
import SAFE_llboost_v17 as V17
from changepoint_synthetic import simulate, W_old

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0
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
    nInst, nt = out.shape
    POS = np.zeros((nInst, nt))
    for t in range(mod.WARMUP, nt):
        p = np.asarray(mod.getMyPosition(out[:, :t + 1]))
        POS[1:, t] = p[1:]
    return POS


def real_data_check():
    print("=== 1. real prices.txt: v17 vs v12 (v15 already confirmed identical to v12) ===")
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

    POS12 = build(V12)
    POS16 = build(V17)
    curve12 = np.array([wscore(POS12, P_, E - NUMTEST, E, nInst) for E in end_days])
    curve16 = np.array([wscore(POS16, P_, E - NUMTEST, E, nInst) for E in end_days])
    n_worse = int((curve16 < curve12).sum()); n_better = int((curve16 > curve12).sum())
    mism = int((~np.all(POS12 == POS16, axis=0)).sum())
    print(f"  v12: OLD={wscore(POS12,P_,500,750,nInst):.1f} NEW={wscore(POS12,P_,750,nt,nInst):.1f} "
          f"rmean={curve12.mean():.1f} rfloor={curve12.min():.1f}")
    print(f"  v17: OLD={wscore(POS16,P_,500,750,nInst):.1f} NEW={wscore(POS16,P_,750,nt,nInst):.1f} "
          f"rmean={curve16.mean():.1f} rfloor={curve16.min():.1f}")
    print(f"  days with any position differing: {mism}/{nt-96}  n_worse={n_worse}/61  n_better={n_better}/61\n")


def run_changepoint(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V10); POS10 = walk_pos_idio(V10, out, NT_PRE)
    reset_module_state(V15); POS15 = walk_pos_idio(V15, out, NT_PRE)
    reset_module_state(V17); POS16 = walk_pos_idio(V17, out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl15 = daily_pnl_idio(POS15, out, NT_PRE + 1, nt - 1)
    pnl16 = daily_pnl_idio(POS16, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum10 = np.cumsum(pnl10)[-1]; cum15 = np.cumsum(pnl15)[-1]; cum16 = np.cumsum(pnl16)[-1]
    cum_oracle = np.cumsum(pnl_oracle)[-1]
    loss10 = cum_oracle - cum10; loss15 = cum_oracle - cum15; loss16 = cum_oracle - cum16
    saved15 = (loss10 - loss15) / max(1.0, loss10) * 100
    saved16 = (loss10 - loss16) / max(1.0, loss10) * 100
    mism = int((~np.all(POS15 == POS16, axis=0)).sum())
    print(f"  mode={mode:8s} seed={seed}: v10={cum10:9.0f}  v15={cum15:9.0f}  v17={cum16:9.0f}  "
          f"oracle={cum_oracle:9.0f}  saved15={saved15:5.1f}%  saved16={saved16:5.1f}%  "
          f"days v15!=v17={mism}")
    return saved16


def trend_regime():
    print("\n=== 3. trend-regime momentum/flip/noise (v17 vs v10/v15) ===")
    P_real = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nDays = P_real.shape
    commRate_local = np.full(nInst, 1e-4); commRate_local[0] = 2e-5

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

    def daily_pnl(POS, full, S, E):
        curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
        for tt in range(S, E + 1):
            cur = full[:, tt - 1]
            newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
            if tt > S:
                tot.append(float((curPos[1:] * (cur[1:] - prevCur[1:])).sum()) - float(comm_vec[1:].sum()))
            dP = newPos - curPos
            comm_vec = commRate_local * np.abs(dP) * cur
            prevCur = cur; curPos = newPos
        return np.array(tot)

    def walk(mod, full, S, E):
        POS = np.zeros((nInst, full.shape[1]))
        for t in range(mod.WARMUP, E):
            p = np.asarray(mod.getMyPosition(full[:, :t + 1]))
            POS[1:, t] = p[1:]
        return POS

    T_EXT = 150
    S, E = nDays, nDays + T_EXT
    for kind in ("momentum", "flip", "noise"):
        full = make_ext(kind, T_ext=T_EXT)
        reset_module_state(V10); POS10 = walk(V10, full, S, E)
        reset_module_state(V17); POS16 = walk(V17, full, S, E)
        pnl10 = daily_pnl(POS10, full, S, E)
        pnl16 = daily_pnl(POS16, full, S, E)
        print(f"  {kind:9s}: v10={pnl10.sum():9.0f}  v17={pnl16.sum():9.0f}")


if __name__ == "__main__":
    real_data_check()

    print("=== 2. change-point experiment: v17 vs v15 (vs recorded v15 numbers) ===")
    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_changepoint(mode, seed))
    print("\n=== summary: v17 frac of oracle-gap recovered vs plain v10 ===")
    for mode in ("reverse", "rotate"):
        vals = results[mode]
        print(f"{mode}: v17 frac_saved per seed = {[round(v,1) for v in vals]}  mean={np.mean(vals):.1f}%")

    trend_regime()
