"""Regime test for the ALGO leg specifically (not the cross-sectional book).

Isolate the ALGO contrarian leg's OWN daily PnL (trade only col 0 with the fade-30d signal,
sized to the $100k cap; no idio book, no hedge) and test which ALGO-SPECIFIC state predicts
when that leg makes money: ALGO realized vol, contra signal strength |z|, recent |move|, and
ALGO return autocorrelation (how mean-reverting it currently is). Tercile split + correlation.
Caveat: 1 time series, positions autocorrelated -> low effective N; read as suggestive."""
import numpy as np, pandas as pd

prc_all = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000); dlr[0] = 100_000
K, WZ, DOLLARS = 30, 60, 200_000


def algo_pos_and_state(prc):
    t = prc.shape[1]; cap = 100000/prc[0, -1]
    lpA = np.log(prc[0]); rA = lpA[1:]-lpA[:-1]
    mv = lpA[K:]-lpA[:-K]; z = (mv[-1]-mv[-WZ:].mean())/(mv[-WZ:].std()+1e-12)
    rev_sh = float(np.clip(-np.clip(z, -3, 3)*DOLLARS/prc[0, -1], -cap, cap))
    # ALGO-specific causal states
    ac = np.corrcoef(rA[-40:-1], rA[-39:])[0, 1] if t > 42 else 0.0   # recent ALGO return autocorr
    state = dict(
        algo_vol=rA[-20:].std(),                 # ALGO 20d realized vol
        absz=abs(z),                             # contra signal strength |z|
        absmove=abs(mv[-1]),                     # size of the K-day move we're fading
        ret_ac=ac,                               # ALGO daily-return autocorr (neg => mean-reverting now)
    )
    return rev_sh, state


# collect the ALGO leg's isolated daily PnL, aligned with the state that set the position
cash = 0; cp = np.zeros(nInst); val = 0; cm = 0
rows = []; prev = None
for t in range(nt-440, nt+1):
    p = prc_all[:, :t]; cur = p[:, -1]; npos = np.zeros(nInst)
    if t < nt:
        rev_sh, st = algo_pos_and_state(p)
        npos[0] = int(np.clip(rev_sh, -(dlr[0]/cur[0]), (dlr[0]/cur[0])))
    else:
        npos, st = cp.copy(), None
    d = npos-cp; cash -= cur.dot(d)+cm; dv = cur*np.abs(d); cm = (dv*commRate).sum(); cp = npos.copy()
    pl = cash+cp.dot(cur)-val; val = cash+cp.dot(cur)
    if t > nt-440 and prev is not None:
        rows.append((prev, pl))
    prev = st

keys = list(rows[0][0].keys())
S = {k: np.array([r[0][k] for r in rows]) for k in keys}
PL = np.array([r[1] for r in rows])
print(f"ALGO leg alone: n={len(PL)} days, mean PnL {PL.mean():.0f}/day, "
      f"ann.Sharpe {np.sqrt(250)*PL.mean()/(PL.std()+1e-9):.2f}, %profitable days {100*np.mean(PL>0):.0f}%\n")
print(f"{'ALGO state (causal)':16} {'corr w/ leg PnL':>16} {'low-tercile':>13} {'high-tercile':>13}")
for k in keys:
    x = S[k]; c = np.corrcoef(x, PL)[0, 1]
    lo = PL[x <= np.percentile(x, 33)].mean(); hi = PL[x >= np.percentile(x, 67)].mean()
    print(f"{k:16} {c:16.3f} {lo:13.0f} {hi:13.0f}")
print("\n(is any ALGO-state a clean separator of when the index-reversion leg pays?)")
