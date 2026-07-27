"""rot_gate_compare.py — is the rotation gate too slow, and would a short-term PROFITABILITY/SHARPE
gate be better? Compare gate METRICS head-to-head on the same cached signals:
  IC-sig   : current -- paired t(IC_c - IC_champ) > bar, sustained P   (significance on IC)
  PnL-raw  : rotate if trailing mean as-if-traded PnL beats champion (+margin), sustained P (no sig test)
  PnL-sig  : paired t(PnL_c - PnL_champ) > bar, sustained P            (significance on PnL)
  Sharpe   : trailing Sharpe_c > Sharpe_champ (+margin), sustained P
Each at window W and persistence P. Score them on THREE axes:
  WHIPSAW-real : # non-champ days on real 500-750  (want 0 -> no false switching / 694 preserved)
  LAG-momentum : days after a real regime onset before it switches (want small -> responsive)
  WHIPSAW-flip : # signal flips in a choppy regime  (want small -> not thrashing)
The honest question: can any PnL/Sharpe gate cut LAG without blowing up WHIPSAW?
"""
import numpy as np, pandas as pd
import SAFE_rotate as R
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
SIGS = ("champ",) + tuple(R.CHALLENGERS)

def cache_metrics(P):
    """causal per-signal daily IC and as-if-traded equal-weight PnL over all gradable days."""
    R._SIG.clear(); R._RET.clear(); R._ICD.clear(); R._AZ.clear(); R._XC.clear(); R._ensure_cache(P)
    logp = np.log(P); T = P.shape[1]; ic = {s: {} for s in SIGS}; pnl = {s: {} for s in SIGS}
    for n in range(R.WARMUP, T - 1):
        rr = logp[1:, n] - logp[1:, n - 1]                     # realized idio return graded vs _SIG[n]
        for s in SIGS:
            f = R._SIG[n][s]
            ic[s][n] = R._ic1(s, n)
            pnl[s][n] = float((np.sign(f) * rr).sum())         # equal-weight idio book daily PnL proxy
    return ic, pnl

def gate_series(ic, pnl, T0, T1, metric, W, P, bar=2.91, margin=0.0):
    """chosen signal per day for a given gate metric."""
    ch = R.CHALLENGERS
    def qual(a, c):
        lo = a - W + 1
        if lo < R.WARMUP: return False
        rng = range(lo, a + 1)
        if metric == "ic":
            d = np.array([ic[c][n] - ic["champ"][n] for n in rng]); ci = np.array([ic[c][n] for n in rng])
            td = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-18); ti = ci.mean() / (ci.std() / np.sqrt(len(ci)) + 1e-18)
            return d.mean() >= 0 and td > bar and ci.mean() > 0 and ti > bar
        pc = np.array([pnl[c][n] for n in rng]); pch = np.array([pnl["champ"][n] for n in rng]); d = pc - pch
        if metric == "pnl_raw":
            return d.mean() > margin
        if metric == "pnl_sig":
            td = d.mean() / (d.std() / np.sqrt(len(d)) + 1e-18); return d.mean() > 0 and td > bar
        if metric == "sharpe":
            shc = pc.mean() / (pc.std() + 1e-9); shx = pch.mean() / (pch.std() + 1e-9)
            return shc > shx + margin and pc.mean() > 0
    out = {}
    for t in range(T0, T1):
        pick = "champ"
        for c in ch:
            if all(qual(a, c) for a in range(t - P, t)): pick = c; break
        out[t] = pick
    return out

def make_ext(kind, cross_mom=0.6, T_ext=150, period=25, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc).copy(); vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:].copy(); K = 5
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        sgn = -1.0 if (kind == "flip" and (step // period) % 2 == 1) else 1.0
        drift = sgn * cross_mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :nDays] = prc
    return full

GATES = [("IC-sig (current)", "ic", 40, 7), ("PnL-raw W40", "pnl_raw", 40, 7),
         ("PnL-raw W20", "pnl_raw", 20, 5), ("PnL-sig W40", "pnl_sig", 40, 7),
         ("Sharpe W40", "sharpe", 40, 7), ("Sharpe W20", "sharpe", 20, 5)]

ic_r, pnl_r = cache_metrics(prc)
mom = make_ext("momentum"); ic_m, pnl_m = cache_metrics(mom); D1 = nDays
flp = make_ext("flip");     ic_f, pnl_f = cache_metrics(flp)

print(f"{'gate':<18}{'WHIPSAW-real':>13}{'LAG-momentum':>14}{'WHIPSAW-flip':>14}")
for name, metric, W, P in GATES:
    gr = gate_series(ic_r, pnl_r, 500, 750, metric, W, P)
    wr = sum(1 for v in gr.values() if v != "champ")
    gm = gate_series(ic_m, pnl_m, nDays, nDays + 150, metric, W, P)
    lag = next((t - D1 for t in sorted(gm) if gm[t] != "champ"), None)
    gf = gate_series(ic_f, pnl_f, nDays, nDays + 150, metric, W, P)
    picks = [gf[t] for t in sorted(gf)]; flips = sum(1 for i in range(1, len(picks)) if picks[i] != picks[i - 1])
    print(f"{name:<18}{wr:>13}{str(lag):>14}{flips:>14}")
print("\nWHIPSAW-real: non-champ days on REAL 500-750 (want 0). LAG-momentum: days to switch in a real")
print("regime (want small). WHIPSAW-flip: signal flips in 25d chop (want small). The tradeoff to judge.")
