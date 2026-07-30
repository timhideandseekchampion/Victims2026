"""
test_v30c_rotw_scrutiny.py -- scrutinizes the jagged, non-monotonic ROT_W=38/40/45 real-data
"improvement" found in test_v30b_rotw_dense.py before trusting it. Two independent checks:

  1. Window-boundary sensitivity: does the advantage survive if OLD/NEW are shifted by +/-20/+/-40
     days, or is it an artifact of the SPECIFIC 500-750/750-1000 boundary?
  2. Synthetic battery: does a shorter ROT_W also help (or at least not hurt) on the
     independently-calibrated change-point (reverse/rotate) and trend-regime (momentum/flip/noise)
     scenarios, or does it only "work" by luck on this one real 904-day history?

Run: python3 test_v30c_rotw_scrutiny.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10
import SAFE_llboost_v22 as V22
from changepoint_synthetic import simulate, W_old

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0
NT_PRE, NT_POST = 1000, 600
SEEDS = [123, 124, 125, 126]


def reset(mod):
    for name in ("_SIG", "_FB", "_RET", "_XC", "_ICD", "_PN"):
        if hasattr(mod, name):
            getattr(mod, name).clear()
    mod._PREV_ALGO_SHARES = 0; mod._PREV_T = -1; mod._DLR = None


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


if __name__ == "__main__":
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nt = P_.shape

    print("=== 1. window-boundary sensitivity (real data) ===")

    def build(mod):
        reset(mod)
        POS = np.zeros((nInst, nt))
        for t in range(1, nt):
            prcSoFar = P_[:, :t]
            p = np.asarray(mod.getMyPosition(prcSoFar))
            lim = (dlr / prcSoFar[:, -1]).astype(int)
            POS[:, t - 1] = np.clip(p, -lim, lim).astype(int)
        return POS

    orig_rot_w = V22.ROT_W
    POS22 = build(V22)

    candidates = [38, 40, 45]
    boundary_sets = [
        ("default", 500, 750, 750, nt),
        ("shift+20", 520, 770, 770, nt),
        ("shift-20", 480, 730, 730, nt - 20 if nt - 20 > 780 else nt),
        ("shift+40", 540, 790, 790, nt),
        ("shift-40", 460, 710, 710, nt),
    ]
    POS_by_rw = {}
    for rw in candidates:
        V22.ROT_W = rw
        POS_by_rw[rw] = build(V22)
    V22.ROT_W = orig_rot_w

    print(f"{'boundary':>10}{'ROT_W':>7}{'OLD':>9}{'NEW':>9}{'OLD(v22)':>10}{'NEW(v22)':>10}{'OLD_better':>12}{'NEW_better':>12}")
    for name, os_, oe, ns_, ne in boundary_sets:
        old22 = wscore(POS22, P_, os_, oe, nInst); new22 = wscore(POS22, P_, ns_, ne, nInst)
        for rw in candidates:
            old_rw = wscore(POS_by_rw[rw], P_, os_, oe, nInst)
            new_rw = wscore(POS_by_rw[rw], P_, ns_, ne, nInst)
            print(f"{name:>10}{rw:>7}{old_rw:>9.1f}{new_rw:>9.1f}{old22:>10.1f}{new22:>10.1f}"
                  f"{str(old_rw > old22):>12}{str(new_rw > new22):>12}")
        print()

    print("=== 2. synthetic change-point: ROT_W candidates vs v22 (vs recorded v22 numbers) ===")
    for rw in candidates:
        results = {"reverse": [], "rotate": []}
        for mode in ("reverse", "rotate"):
            for seed in SEEDS:
                out, idio, algo_ret, W_new, leaders_new = simulate(NT_PRE, NT_POST, mode, seed=seed)
                nInst_s, nt_s = out.shape; nStock = nInst_s - 1

                reset(V22); POS_base = walk_pos_idio(V22, out, NT_PRE)
                V22.ROT_W = rw
                reset(V22); POS_rw = walk_pos_idio(V22, out, NT_PRE)
                V22.ROT_W = orig_rot_w

                price_idio = out[1:, :]
                oracle_pos = np.zeros((nStock, nt_s))
                for t in range(nt_s):
                    Wt = W_old if t <= NT_PRE else W_new
                    oracle_pos[:, t] = np.sign(Wt @ idio[:, t]) * (10_000.0 / price_idio[:, t])
                oracle_POS_full = np.zeros((nInst_s, nt_s)); oracle_POS_full[1:, :] = oracle_pos

                pnl_base = daily_pnl_idio(POS_base, out, NT_PRE + 1, nt_s - 1)
                pnl_rw = daily_pnl_idio(POS_rw, out, NT_PRE + 1, nt_s - 1)
                pnl_oracle = daily_pnl_idio(oracle_POS_full, out, NT_PRE + 1, nt_s - 1)

                cum_base = np.cumsum(pnl_base)[-1]; cum_rw = np.cumsum(pnl_rw)[-1]
                cum_oracle = np.cumsum(pnl_oracle)[-1]
                loss_base = cum_oracle - cum_base
                saved_rw = (loss_base - (cum_oracle - cum_rw)) / max(1.0, loss_base) * 100
                results[mode].append(saved_rw)
        for mode in ("reverse", "rotate"):
            vals = results[mode]
            print(f"  ROT_W={rw} {mode:8s}: per-seed={[round(v,1) for v in vals]}  mean={np.mean(vals):.1f}%")
    print("  (recorded v22 baseline: reverse mean=28.2%, rotate mean=-1.1%)\n")

    print("=== 3. trend-regime momentum/flip/noise: ROT_W candidates vs v22 ===")
    P_real = P_
    nDays = nt
    commRate_local = commRate

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
        reset(V22); POS_base = walk(V22, full, S, E)
        pnl_base = daily_pnl(POS_base, full, S, E)
        line = f"  {kind:9s}: v22={pnl_base.sum():9.0f}"
        for rw in candidates:
            V22.ROT_W = rw
            reset(V22); POS_rw = walk(V22, full, S, E)
            V22.ROT_W = orig_rot_w
            pnl_rw = daily_pnl(POS_rw, full, S, E)
            line += f"  ROT_W={rw}={pnl_rw.sum():9.0f}"
        print(line)
