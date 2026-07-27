"""Compute the 'what went wrong' diagnostic dataset.

For each strategy it replays the EXACT eval.py accounting (lagged commission, integer clip,
score = mu * SR^2/(SR^2+1)) over:
  * the OLD graded window (days 501-750) and the NEW graded window (days 751-1000),
  * rolling 250-day windows across the whole file,
and attributes every day's PnL to the ALGO index leg (instrument 0) vs the idio book
(instruments 1..49). It also produces an ALGO-leg-OFF ("idio only") counterfactual.

Exports diag_data.json for build_diagnostics.py.
"""
import json, numpy as np, pandas as pd
import SAFE, SWING, QUAL, SAFE_llalgo, SAFE_lldollar, SAFE_llmatch, SAFE_llvol, SAFE_llvol_vo, SAFE_llboost, SAFE_llboost_v2, SAFE_llboost_v3, SAFE_llboost_v4, SAFE_llboost_v5, SAFE_llboost_v6, SAFE_llboost_v7

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(prc.columns)
P = prc.values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
print(f"nInst={nInst} nt={nt}")

STRATS = [("SAFE", SAFE), ("SWING", SWING), ("QUAL", QUAL),
          ("LLALGO", SAFE_llalgo), ("LLDOLLAR", SAFE_lldollar),
          ("LLMATCH", SAFE_llmatch), ("LLVOL", SAFE_llvol), ("LLVOL_VO", SAFE_llvol_vo),
          ("LLBOOST", SAFE_llboost), ("LLBOOST_V2", SAFE_llboost_v2),
          ("LLBOOST_V3", SAFE_llboost_v3), ("LLBOOST_V4", SAFE_llboost_v4),
          ("LLBOOST_V5", SAFE_llboost_v5), ("LLBOOST_V6", SAFE_llboost_v6),
          ("LLBOOST_V7", SAFE_llboost_v7)]


def score(mu, sd):
    if mu <= 0 or sd < 1e-10:
        return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def pos_matrix(mod):
    """POS[:,k] = clipped integer position decided from prices P[:, :k+1] (held day k -> k+1)."""
    POS = np.zeros((nInst, nt))
    for k in range(130, nt):                       # positions before day 130 are unused (warmup)
        cur = P[:, k]
        lim = (dlr / cur).astype(int)
        POS[:, k] = np.clip(np.asarray(mod.getMyPosition(P[:, :k + 1])), -lim, lim).astype(int)
    return POS


def window(POS, S, E, algo_off=False):
    """Mirror eval.calcPL exactly over 1-based day counts; score days (S, E].
    Returns daily total/algo/idio PnL, cumulative curves, and the eval score."""
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None
    tot = []; algo = []; idio = []
    for tt in range(S, E + 1):
        cur = P[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if algo_off:
            newPos[0] = 0.0
        if tt > S:                                 # scored day: attribute PnL held from prevCur -> cur
            pl = curPos * (cur - prevCur) - comm_vec
            tot.append(float(pl.sum())); algo.append(float(pl[0])); idio.append(float(pl[1:].sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur     # charged on the NEXT day (eval's lag)
        prevCur = cur; curPos = newPos
    tot = np.array(tot); algo = np.array(algo); idio = np.array(idio)
    return {"tot": tot, "algo": algo, "idio": idio,
            "mu": float(tot.mean()), "sd": float(tot.std()),
            "score": score(tot.mean(), tot.std()),
            "algo_sum": float(algo.sum()), "idio_sum": float(idio.sum())}


# ---- position matrices (one causal pass per strategy) ------------------------
POS = {}
for name, mod in STRATS:
    print("positions:", name, "...")
    POS[name] = pos_matrix(mod)

OLD = (501 - 1, 750)   # window() scores days (S, E]; (500, 750] = days 501..750
NEW = (nt - NUMTEST, nt)   # (750, 1000] = days 751..1000

diag = {"names": names, "nt": nt, "numtest": NUMTEST,
        "windows": {"OLD": [OLD[0] + 1, OLD[1]], "NEW": [NEW[0] + 1, NEW[1]]},
        "strategies": [s for s, _ in STRATS],
        "headline": {}, "cum": {}}


def curves(w, S):
    days = list(range(S + 1, S + 1 + len(w["tot"])))
    return {"days": days,
            "tot": [round(x, 1) for x in np.cumsum(w["tot"])],
            "idio": [round(x, 1) for x in np.cumsum(w["idio"])],
            "algo": [round(x, 1) for x in np.cumsum(w["algo"])]}


for name, _ in STRATS:
    wo = window(POS[name], *OLD); wn = window(POS[name], *NEW)
    diag["headline"][name] = {
        "OLD": {"score": round(wo["score"], 1), "mu": round(wo["mu"], 1), "sd": round(wo["sd"], 1),
                "algo": round(wo["algo_sum"]), "idio": round(wo["idio_sum"])},
        "NEW": {"score": round(wn["score"], 1), "mu": round(wn["mu"], 1), "sd": round(wn["sd"], 1),
                "algo": round(wn["algo_sum"]), "idio": round(wn["idio_sum"])}}
    diag["cum"][name] = {"OLD": curves(wo, OLD[0]), "NEW": curves(wn, NEW[0])}

# ALGO-leg-OFF counterfactual (idio book alone) — uses SAFE's idio positions
io_old = window(POS["SAFE"], *OLD, algo_off=True)
io_new = window(POS["SAFE"], *NEW, algo_off=True)
diag["idio_only"] = {
    "OLD": {"score": round(io_old["score"], 1), "mu": round(io_old["mu"], 1), "sd": round(io_old["sd"], 1)},
    "NEW": {"score": round(io_new["score"], 1), "mu": round(io_new["mu"], 1), "sd": round(io_new["sd"], 1)}}
diag["cum"]["IDIO_ONLY"] = {"NEW": curves(io_new, NEW[0]), "OLD": curves(io_old, OLD[0])}

# ---- rolling 250-day score across the file -----------------------------------
end_days = list(range(400, nt + 1, 10))
roll = {"end_days": end_days, "SAFE": [], "LLALGO": [], "LLMATCH": [], "LLVOL": [], "LLVOL_VO": [],
        "LLBOOST": [], "LLBOOST_V2": [], "LLBOOST_V3": [], "LLBOOST_V4": [], "LLBOOST_V5": [], "LLBOOST_V6": [],
        "LLBOOST_V7": [], "IDIO": []}
for E in end_days:
    S = E - NUMTEST
    roll["SAFE"].append(round(window(POS["SAFE"], S, E)["score"], 1))
    roll["LLALGO"].append(round(window(POS["LLALGO"], S, E)["score"], 1))
    roll["LLMATCH"].append(round(window(POS["LLMATCH"], S, E)["score"], 1))
    roll["LLVOL"].append(round(window(POS["LLVOL"], S, E)["score"], 1))
    roll["LLVOL_VO"].append(round(window(POS["LLVOL_VO"], S, E)["score"], 1))
    roll["LLBOOST"].append(round(window(POS["LLBOOST"], S, E)["score"], 1))
    roll["LLBOOST_V2"].append(round(window(POS["LLBOOST_V2"], S, E)["score"], 1))
    roll["LLBOOST_V3"].append(round(window(POS["LLBOOST_V3"], S, E)["score"], 1))
    roll["LLBOOST_V4"].append(round(window(POS["LLBOOST_V4"], S, E)["score"], 1))
    roll["LLBOOST_V5"].append(round(window(POS["LLBOOST_V5"], S, E)["score"], 1))
    roll["LLBOOST_V6"].append(round(window(POS["LLBOOST_V6"], S, E)["score"], 1))
    roll["LLBOOST_V7"].append(round(window(POS["LLBOOST_V7"], S, E)["score"], 1))
    roll["IDIO"].append(round(window(POS["SAFE"], S, E, algo_off=True)["score"], 1))
diag["rolling"] = roll

# ---- MATCH_K sweep: score-vs-k on both windows + rolling floor/mean -----------
# Reuse SAFE's idio positions (rows 1..49) and set the ALGO leg to k * net-$ tilt.
cur0 = P[0]
netdol = (POS["SAFE"][1:, :] * P[1:, :]).sum(axis=0)          # book's signed $ tilt per day


def window_k(S, E, k):
    lim0 = (dlr[0] / cur0).astype(int)
    row0 = np.clip(np.clip(k * netdol / cur0, -(dlr[0] / cur0), (dlr[0] / cur0)), -lim0, lim0).astype(int)
    POSk = POS["SAFE"].copy(); POSk[0, :] = row0
    return window(POSk, S, E)["score"]


KS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
sweep = {"k": KS, "old": [], "new": [], "roll_mean": [], "roll_floor": [], "match_k": SAFE_llmatch.MATCH_K}
for k in KS:
    sweep["old"].append(round(window_k(*OLD, k), 1))
    sweep["new"].append(round(window_k(*NEW, k), 1))
    scs = [window_k(E - NUMTEST, E, k) for E in end_days]
    sweep["roll_mean"].append(round(float(np.mean(scs)), 1))
    sweep["roll_floor"].append(round(float(min(scs)), 1))
diag["ksweep"] = sweep

json.dump(diag, open("diag_data.json", "w"))
print("\nwrote diag_data.json")
print(f"{'strategy':<12}{'OLD score':>11}{'NEW score':>11}{'OLD ALGO$':>11}{'NEW ALGO$':>11}")
for name, _ in STRATS:
    h = diag["headline"][name]
    print(f"{name:<12}{h['OLD']['score']:>11}{h['NEW']['score']:>11}{h['OLD']['algo']:>11}{h['NEW']['algo']:>11}")
print(f"{'IDIO_ONLY':<12}{diag['idio_only']['OLD']['score']:>11}{diag['idio_only']['NEW']['score']:>11}")
print(f"\nMATCH_K sweep:\n{'k':>6}{'OLD':>8}{'NEW':>8}{'roll_mean':>11}{'roll_floor':>12}")
for i, k in enumerate(sweep["k"]):
    print(f"{k:>6}{sweep['old'][i]:>8}{sweep['new'][i]:>8}{sweep['roll_mean'][i]:>11}{sweep['roll_floor'][i]:>12}")
