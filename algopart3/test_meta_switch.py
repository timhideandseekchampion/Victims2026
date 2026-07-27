"""Hypothesis: since neither LLVOL nor LLDOLLAR is stable across the whole file (LLVOL wins late,
LLDOLLAR wins mid-file), build a META-SWITCH that picks WHICH MECHANISM's ALGO-leg position to use
each day, based on trailing realized performance of each mechanism's OWN historical ALGO-leg PnL --
not summing/blending the two signals (already shown to hurt: README "combining with lead-lag
hurts"), but a hard selector that runs one FULL mechanism at a time, at full conviction. Idio book
(SAFE's, identical across every strategy here) is held fixed throughout; only the ALGO leg's SOURCE
mechanism is switched.
"""
import numpy as np, pandas as pd
import SAFE, SAFE_llvol, SAFE_lldollar

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def window(POS, S, E):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return {"mu": float(tot.mean()), "sd": float(tot.std()), "score": score(tot.mean(), tot.std())}


print("computing per-strategy position matrices (causal) ...")
llvol_pos = np.zeros((nInst, nt)); lldollar_pos = np.zeros((nInst, nt)); idio_only = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    llvol_pos[:, k] = np.clip(np.asarray(SAFE_llvol.getMyPosition(P[:, :k + 1])), -lim, lim).astype(int)
    lldollar_pos[:, k] = np.clip(np.asarray(SAFE_lldollar.getMyPosition(P[:, :k + 1])), -lim, lim).astype(int)
    full = np.asarray(SAFE.getMyPosition(P[:, :k + 1])); p = full.copy(); p[0] = 0
    idio_only[:, k] = np.clip(p, -lim, lim).astype(int)
print("done")

# ---- daily ALGO-leg-only PnL for each mechanism (causal, includes its own commission) ----
def algo_leg_pnl(pos_row):
    pnl = np.zeros(nt); prev_pos = 0.0; prev_comm = 0.0
    for k in range(130, nt):
        cur = P[0, k]
        if k > 130:
            pnl[k] = prev_pos * (cur - P[0, k - 1]) - prev_comm
        newPos = pos_row[k]
        dP = newPos - prev_pos
        prev_comm = commRate[0] * abs(dP) * cur
        prev_pos = newPos
    return pnl

pnl_llvol = algo_leg_pnl(llvol_pos[0]); pnl_lldollar = algo_leg_pnl(lldollar_pos[0])

OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

def report(name, POS):
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{name:<34}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}")

print(f"{'config':<34}{'OLD':>8}{'NEW':>8}{'roll_mean':>11}{'roll_floor':>12}")
report("shipped SAFE_llvol", llvol_pos)
report("SAFE_lldollar", lldollar_pos)

# ---- oracle upper bound: pick whichever mechanism's REALIZED pnl[k] (that specific day) was
# actually better, in hindsight -- shows the theoretical ceiling of "perfect switching" ----
oracle = idio_only.copy()
for k in range(130, nt):
    oracle[0, k] = llvol_pos[0, k] if pnl_llvol[k] >= pnl_lldollar[k] else lldollar_pos[0, k]
report("oracle (perfect hindsight switch)", oracle)

# ---- meta-switch: pick mechanism with higher TRAILING L-day mean pnl (causal) ----
for L in (30, 60, 90, 120, 180, 250):
    POS = idio_only.copy()
    switches = 0; last_choice = None
    for k in range(130, nt):
        lo = max(130, k - L)
        mv = pnl_llvol[lo:k].mean() if k > lo else 0.0
        md = pnl_lldollar[lo:k].mean() if k > lo else 0.0
        choice = "V" if mv >= md else "D"
        if last_choice is not None and choice != last_choice: switches += 1
        last_choice = choice
        POS[0, k] = llvol_pos[0, k] if choice == "V" else lldollar_pos[0, k]
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{'meta-switch L='+str(L)+' (switches='+str(switches)+')':<34}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}")

print("\n--- finer sweep around L=30 ---")
for L in (10, 15, 20, 25, 35, 40, 45, 50):
    POS = idio_only.copy()
    switches = 0; last_choice = None
    for k in range(130, nt):
        lo = max(130, k - L)
        mv = pnl_llvol[lo:k].mean() if k > lo else 0.0
        md = pnl_lldollar[lo:k].mean() if k > lo else 0.0
        choice = "V" if mv >= md else "D"
        if last_choice is not None and choice != last_choice: switches += 1
        last_choice = choice
        POS[0, k] = llvol_pos[0, k] if choice == "V" else lldollar_pos[0, k]
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    print(f"{'meta-switch L='+str(L)+' (switches='+str(switches)+')':<34}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}")
