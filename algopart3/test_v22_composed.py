"""
test_v22_composed.py -- checks that v19 (two-hop boost) and v21 (G84 learned per-name RS weight)
compose cleanly into v22, rather than assuming it from each one's independent validation. Compares
v22 against v15 (base), v19 (boost-only), and v21 (RS-weight-only) on real data (incl. WIN250 =
day 250-500, per user request -- not just OLD=500-750/NEW=750-1000), change-point reverse/rotate,
and trend-regime momentum/flip/noise.

Run: python3 test_v22_composed.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v15 as V15
import SAFE_llboost_v19 as V19
import SAFE_llboost_v21 as V21
import SAFE_llboost_v22 as V22
from changepoint_synthetic import simulate, W_old

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0
NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]

MODS = {"v15": V15, "v19": V19, "v21": V21, "v22": V22}


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
    print("=== 1. real prices.txt: v22 (composed) vs v15/v19/v21 ===")
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

    POS = {name: build(mod) for name, mod in MODS.items()}
    curves = {name: np.array([wscore(POS[name], P_, E - NUMTEST, E, nInst) for E in end_days])
              for name in MODS}

    print(f"{'variant':>8}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}")
    for name in ("v15", "v19", "v21", "v22"):
        w250 = wscore(POS[name], P_, 250, 500, nInst)
        old = wscore(POS[name], P_, 500, 750, nInst)
        new = wscore(POS[name], P_, 750, nt, nInst)
        c = curves[name]
        print(f"{name:>8}{w250:>9.1f}{old:>9.1f}{new:>9.1f}{c.mean():>9.1f}{c.min():>9.1f}")

    n_worse = int((curves["v22"] < curves["v15"]).sum())
    n_better = int((curves["v22"] > curves["v15"]).sum())
    mism = int((~np.all(POS["v15"] == POS["v22"], axis=0)).sum())
    print(f"  v22 vs v15: n_worse={n_worse}/61  n_better={n_better}/61  days differing={mism}/{nt-96}\n")


def run_changepoint(mode, seed):
    out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
    nInst, nt = out.shape; nStock = nInst - 1

    POS = {}
    for name, mod in MODS.items():
        reset_module_state(mod)
        POS[name] = walk_pos_idio(mod, out, NT_PRE)
    reset_module_state(V10); POS10 = walk_pos_idio(V10, out, NT_PRE)

    price_idio = out[1:, :]
    oracle_pos = np.zeros((nStock, nt))
    for t in range(nt):
        Wt = W_old if t <= NT_PRE else W_new
        oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
    oracle_POS_full = np.zeros((nInst, nt)); oracle_POS_full[1:, :] = oracle_pos

    pnl = {name: daily_pnl_idio(POS[name], out, NT_PRE + 1, nt - 1) for name in MODS}
    pnl10 = daily_pnl_idio(POS10, out, NT_PRE + 1, nt - 1)
    pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt - 1)

    cum = {name: np.cumsum(pnl[name])[-1] for name in MODS}
    cum10 = np.cumsum(pnl10)[-1]; cum_oracle = np.cumsum(pnl_oracle)[-1]
    loss10 = cum_oracle - cum10
    saved = {name: (loss10 - (cum_oracle - cum[name])) / max(1.0, loss10) * 100 for name in MODS}
    print(f"  mode={mode:8s} seed={seed}: " +
          "  ".join(f"{name}={cum[name]:9.0f}(sv{saved[name]:5.1f}%)" for name in ("v15", "v19", "v21", "v22")))
    return saved


def trend_regime():
    print("\n=== 3. trend-regime momentum/flip/noise (v22 vs v15/v19/v21) ===")
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
        pnl = {}
        for name, mod in MODS.items():
            reset_module_state(mod)
            POS = walk(mod, full, S, E)
            pnl[name] = daily_pnl(POS, full, S, E).sum()
        print(f"  {kind:9s}: " + "  ".join(f"{name}={pnl[name]:9.0f}" for name in ("v15", "v19", "v21", "v22")))


if __name__ == "__main__":
    real_data_check()

    print("=== 2. change-point experiment: v22 vs v15/v19/v21 ===")
    results = {"reverse": [], "rotate": []}
    for mode in ("reverse", "rotate"):
        for seed in SEEDS:
            results[mode].append(run_changepoint(mode, seed))
    print("\n=== summary: frac of oracle-gap recovered vs plain v10 ===")
    for mode in ("reverse", "rotate"):
        for name in ("v15", "v19", "v21", "v22"):
            vals = [r[name] for r in results[mode]]
            print(f"  {mode:8s} {name}: per-seed={[round(v,1) for v in vals]}  mean={np.mean(vals):.1f}%")

    trend_regime()
