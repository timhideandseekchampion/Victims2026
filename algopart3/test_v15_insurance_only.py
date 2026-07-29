"""
test_v15_insurance_only.py -- validates SAFE_llboost_v15 (v12 + momentum/xsac insurance, no v13
gated boost fallback) against the SAME three harnesses used for v13/v14, but only runs v15 itself
(not re-running v10/v11/v12/v13/v14, whose numbers are already on record in README.md /
algothon-protection-stack memory) -- scoped down to keep runtime reasonable.

Recorded baselines being compared against (already validated this session):
  real data:      v10 OLD=871.0 NEW=912.6 | v12 OLD=885.8 NEW=913.8 (v12 rmean/rfloor not on
                   record from this session; only OLD/NEW headline was captured for v12)
  change-point reverse frac_saved: v11=25.1% v13=25.0% v14=28.0%
  change-point rotate  frac_saved: v11=-1.5% v13=-0.7% v14=-0.2%
  trend-regime (150d): momentum v10=210281 v14=672286 pure_mom=708939
                        flip     v10=193893 v14=144715 pure_mom=115648
                        noise    v10=-16174 v14=-16161 pure_mom=8847

Run: python3 test_v15_insurance_only.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v12 as V12
import SAFE_llboost_v15 as V15
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
    print("=== 1. real prices.txt: v15 vs v12 ===")
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
    POS15 = build(V15)
    curve12 = np.array([wscore(POS12, P_, E - NUMTEST, E, nInst) for E in end_days])
    curve15 = np.array([wscore(POS15, P_, E - NUMTEST, E, nInst) for E in end_days])
    n_worse = int((curve15 < curve12).sum()); n_better = int((curve15 > curve12).sum())
    mism = int((~np.all(POS12 == POS15, axis=0)).sum())
    print(f"  v12: OLD={wscore(POS12,P_,500,750,nInst):.1f} NEW={wscore(POS12,P_,750,nt,nInst):.1f} "
          f"rmean={curve12.mean():.1f} rfloor={curve12.min():.1f}")
    print(f"  v15: OLD={wscore(POS15,P_,500,750,nInst):.1f} NEW={wscore(POS15,P_,750,nt,nInst):.1f} "
          f"rmean={curve15.mean():.1f} rfloor={curve15.min():.1f}")
    print(f"  days with any position differing: {mism}/{nt-96}  n_worse={n_worse}/61  n_better={n_better}/61\n")


def run_changepoint(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    reset_module_state(V10); POS10 = walk_pos_idio(V10, out, NT_PRE)
    reset_module_state(V15); POS15 = walk_pos_idio(V15, out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl15 = daily_pnl_idio(POS15, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum10, cum15, cum_oracle = (np.cumsum(pnl10)[-1], np.cumsum(pnl15)[-1], np.cumsum(pnl_oracle)[-1])
    loss10, loss15 = cum_oracle - cum10, cum_oracle - cum15
    saved15 = (loss10 - loss15) / max(1.0, loss10) * 100
    print(f"  mode={mode:8s} seed={seed}: v10={cum10:9.0f}  v15={cum15:9.0f}  oracle={cum_oracle:9.0f}  "
          f"saved15={saved15:5.1f}%")
    return saved15


def trend_regime():
    print("\n=== 3. trend-regime momentum/flip/noise (v15 vs v10, vs recorded v14) ===")
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
        reset_module_state(V15); POS15 = walk(V15, full, S, E)
        pnl10 = daily_pnl(POS10, full, S, E)
        pnl15 = daily_pnl(POS15, full, S, E)
        print(f"  {kind:9s}: v10={pnl10.sum():9.0f}  v15={pnl15.sum():9.0f}")


if __name__ == "__main__":
    real_data_check()

    print("=== 2. change-point experiment: v15 only (vs recorded v10/v11/v13/v14 numbers) ===")
    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_changepoint(mode, seed))
    print("\n=== summary: v15 frac of oracle-gap recovered vs plain v10 ===")
    for mode in ("reverse", "rotate"):
        vals = results[mode]
        print(f"{mode}: v15 frac_saved per seed = {[round(v,1) for v in vals]}  mean={np.mean(vals):.1f}%")

    trend_regime()
