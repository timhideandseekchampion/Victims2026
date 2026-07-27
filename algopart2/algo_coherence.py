"""algo_coherence.py — when the idio book rotates to MOMENTUM, is the $100k ALGO index leg coherent
with that, or does it keep 'trusting lead-lag/reversion' (fading a trending index)?

Two ALGO mechanisms: (1) net-$ gate transplants the idio book's skew -> already signal-agnostic, follows
the rotation on conviction days; (2) on non-conviction days it FADES vs TRENDS the index on the index's
OWN gate, independent of the idio rotation. This tests whether COUPLING (2) to the rotation helps:
  current : non-gated ALGO uses only the index fade-vs-trend gate
  coupled : non-gated ALGO also flips to TREND when the idio book is on a momentum signal or xsac fires
Regimes injected after 750: (A) cross-sectional momentum, index FLAT; (B) cross-sec momentum + index UPTREND.
Report ALGO branch mix (gate/fade/trend), ALGO-leg PnL, total PnL. Real 500-750 must stay 694 for both.
"""
import numpy as np, pandas as pd
import SAFE_rotate as R

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
MOMSET = {"mom", "momJT", "residMom"}

def make_ext(cross_mom=0.6, idx_trend=0.0, idx_revert=0.0, T_ext=150, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc).copy()
    vol = np.diff(logp[1:], axis=1).std(); names = logp[1:, :].copy(); K = 5
    for _ in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        drift = cross_mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()   # cross-sectional (index-neutral)
        common = idx_trend * vol                                                    # common -> index trends
        if idx_revert > 0.0:                                                        # common -> index mean-reverts
            idx = names.mean(0)
            if idx.shape[0] > 30: common += -idx_revert * (idx[-1] - idx[-30])
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + common + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :nDays] = prc
    return full

def pos_for(P, t, couple):
    """mirrors SAFE_rotate.getMyPosition; couple=False == the shipped book. Returns (pos, branch).
    MUST use only prcSoFar = P[:, :t] (no look-ahead), exactly as getMyPosition receives it."""
    Pt = P[:, :t]; cur = Pt[:, -1]; pos = np.zeros(nInst)
    ready = t >= R.WARMUP + R.ROT_W + R.ROT_P
    chosen = R._choose(t) if ready else "champ"
    wz = R._SIG[t][chosen]
    if not (ready and R._kill(t, chosen)):
        pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
    ii = np.clip(pos[1:], -(dlr[1:] / cur[1:]).astype(int), (dlr[1:] / cur[1:]).astype(int)).astype(int)
    net = float((ii * cur[1:]).sum()); cap = dlr[0] / cur[0]
    logp = np.log(Pt)
    if abs(net) >= R.ALGO_LL_DOLLAR:
        av = float(np.sign(net) * cap); branch = "gate"
    else:
        lpA = logp[0]; mv = lpA[R.CONTRA_K:] - lpA[:-R.CONTRA_K]
        z = (mv[-1] - mv[-R.CONTRA_WZ:].mean()) / (mv[-R.CONTRA_WZ:].std() + 1e-12)
        zc = np.clip(z, -3, 3) / 3.0 * (R.CONTRA_DOL / cur[0])
        idx_trend_gate = (t >= R.WARMUP + R.ALGO_ROT_W + R.ALGO_P) and (R._algo_leg_mode(lpA, t) == "trend")
        idio_mom = couple and ready and (chosen in MOMSET or R._xsac_flag(t))
        trend = idx_trend_gate or idio_mom
        av = float(np.clip(zc if trend else -zc, -cap, cap)); branch = "trend" if trend else "fade"
    pos[0] = av; lim = (dlr / cur).astype(int)
    return np.clip(pos, -lim, lim).astype(int), branch

def run(P, couple, S, E):
    R._SIG.clear(); R._RET.clear(); R._ICD.clear(); R._AZ.clear(); R._XC.clear()
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; tot = []; algo = []; br = []
    for t in range(S, E + 1):
        cur = P[:, t - 1]
        if t > S:
            mv = cur - P[:, t - 2]; algo.append(cp[0] * mv[0])
        if t < E:
            R._ensure_cache(P[:, :t]); newPos, branch = pos_for(P, t, couple); br.append(branch)
        else:
            newPos = cp
        dP = newPos - cp; cash -= cur.dot(dP) + comm; comm = np.sum(cur * np.abs(dP) * commRate); cp = newPos
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: tot.append(pl)
    from collections import Counter
    return np.array(tot), np.array(algo), Counter(br)

def sc(p):
    mu, sd = p.mean(), p.std(); return mu * (np.sqrt(250) * mu / sd) ** 2 / ((np.sqrt(250) * mu / sd) ** 2 + 1) if mu > 0 else mu

print("(0) REAL 500-750 (coupling must never trigger -> both == 694):")
for cpl in (False, True):
    tot, algo, br = run(prc, cpl, 500, 750)
    print(f"    couple={str(cpl):<5} SCORE={sc(tot):.0f}  ALGO branch mix={dict(br)}")

REGIMES = [("A: momentum, index FLAT", dict()),
           ("B: momentum + index UPTREND", dict(idx_trend=0.6)),
           ("C: momentum + index MEAN-REVERT (blind risk case)", dict(idx_revert=0.8))]
for label, kw in REGIMES:
    full = make_ext(**kw)
    print(f"\n=== {label} (days 750-900) ===")
    print(f"    {'policy':<9}{'totalPnL':>10}{'ALGOleg':>9}   ALGO branch mix")
    for cpl in (False, True):
        tot, algo, br = run(full, cpl, nDays, nDays + 150)
        print(f"    {'coupled' if cpl else 'current':<9}{tot.sum():>10.0f}{algo.sum():>9.0f}   {dict(br)}")
