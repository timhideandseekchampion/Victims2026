"""
test_v28_ridge_weights_live.py -- validates the LIVE SAFE_llboost_v28.py (v22 + learned
per-half-life ridge weight, gain=2.0 cap=2.0 win=250) against real data (incl. WIN250=day 250-500),
change-point reverse/rotate, and trend-regime momentum/flip/noise -- same convention as
test_v21_g84_dense.py / test_v22_composed.py. Must reproduce test_v27_ridge_weights.py's precomputed
sweep result exactly (OLD=885.8 NEW=918.9 rmean=918.5 rfloor=720.7 n_worse=0/61 at gain=2,cap=2,
win=250) since v28 is the SAME mechanism implemented live rather than via precompute-and-postprocess.

Run: python3 test_v28_ridge_weights_live.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v22 as V22
import SAFE_llboost_v28 as V28
from changepoint_synthetic import simulate, W_old

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0
NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]


def reset_module_state(mod):
    for name in ("_SIG", "_FB", "_RET", "_XC", "_ICD", "_PN", "_FI_HIST", "_RAWRET"):
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
    print("=== 1. real prices.txt: v28 (live) vs v22 ===")
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

    POS22 = build(V22)
    POS28 = build(V28)
    curve22 = np.array([wscore(POS22, P_, E - NUMTEST, E, nInst) for E in end_days])
    curve28 = np.array([wscore(POS28, P_, E - NUMTEST, E, nInst) for E in end_days])
    n_worse = int((curve28 < curve22).sum()); n_better = int((curve28 > curve22).sum())
    mism = int((~np.all(POS22 == POS28, axis=0)).sum())

    win250_22 = wscore(POS22, P_, 250, 500, nInst); win250_28 = wscore(POS28, P_, 250, 500, nInst)
    old22 = wscore(POS22, P_, 500, 750, nInst); old28 = wscore(POS28, P_, 500, 750, nInst)
    new22 = wscore(POS22, P_, 750, nt, nInst); new28 = wscore(POS28, P_, 750, nt, nInst)
    print(f"  v22: WIN250={win250_22:.1f}  OLD={old22:.1f}  NEW={new22:.1f}  "
          f"rmean={curve22.mean():.1f}  rfloor={curve22.min():.1f}")
    print(f"  v28: WIN250={win250_28:.1f}  OLD={old28:.1f}  NEW={new28:.1f}  "
          f"rmean={curve28.mean():.1f}  rfloor={curve28.min():.1f}")
    print(f"  days with any position differing: {mism}/{nt-96}  n_worse={n_worse}/61  n_better={n_better}/61")
    print("  (expect: matches test_v27's gain=2,cap=2,win=250 precompute result exactly)\n")


def run_changepoint(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V10); POS10 = walk_pos_idio(V10, out, NT_PRE)
    reset_module_state(V22); POS22 = walk_pos_idio(V22, out, NT_PRE)
    reset_module_state(V28); POS28 = walk_pos_idio(V28, out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl22 = daily_pnl_idio(POS22, out, NT_PRE + 1, nt - 1)
    pnl28 = daily_pnl_idio(POS28, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum10 = np.cumsum(pnl10)[-1]; cum22 = np.cumsum(pnl22)[-1]; cum28 = np.cumsum(pnl28)[-1]
    cum_oracle = np.cumsum(pnl_oracle)[-1]
    loss10 = cum_oracle - cum10
    saved22 = (loss10 - (cum_oracle - cum22)) / max(1.0, loss10) * 100
    saved28 = (loss10 - (cum_oracle - cum28)) / max(1.0, loss10) * 100
    mism = int((~np.all(POS22 == POS28, axis=0)).sum())
    print(f"  mode={mode:8s} seed={seed}: v10={cum10:9.0f}  v22={cum22:9.0f}  v28={cum28:9.0f}  "
          f"oracle={cum_oracle:9.0f}  saved22={saved22:5.1f}%  saved28={saved28:5.1f}%  "
          f"days v22!=v28={mism}")
    return saved28


def trend_regime():
    print("\n=== 3. trend-regime momentum/flip/noise (v28 vs v10/v22) ===")
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
        reset_module_state(V28); POS28 = walk(V28, full, S, E)
        pnl10 = daily_pnl(POS10, full, S, E)
        pnl28 = daily_pnl(POS28, full, S, E)
        print(f"  {kind:9s}: v10={pnl10.sum():9.0f}  v28={pnl28.sum():9.0f}")


if __name__ == "__main__":
    real_data_check()

    print("=== 2. change-point experiment: v28 vs v22 (vs recorded v22 numbers) ===")
    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_changepoint(mode, seed))
    print("\n=== summary: v28 frac of oracle-gap recovered vs plain v10 ===")
    for mode in ("reverse", "rotate"):
        vals = results[mode]
        print(f"{mode}: v28 frac_saved per seed = {[round(v,1) for v in vals]}  mean={np.mean(vals):.1f}%")

    trend_regime()
