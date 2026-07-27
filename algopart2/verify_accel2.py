"""verify_accel2.py — CLEAN isolation of the accel index gate: same book (SAFE_rotate), accel ON vs
OFF (OFF = fast params set equal to slow, so the accelerant is a no-op). Same idio rotation both ways,
so any ALGO-leg difference is purely the accelerant. Tests whether it (a) captures an index uptrend,
(b) stays safe on a mean-reverting index, (c) doesn't whipsaw in chop — and how a SHORT vs LONG trend
matters (the 120-day IC window may lag a short regime)."""
import numpy as np, pandas as pd
import SAFE_rotate as R

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def make_ext(cross_mom=0.6, idx_trend=0.0, idx_revert=0.0, flip=False, period=25, T_ext=150, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc).copy()
    vol = np.diff(logp[1:], axis=1).std(); names = logp[1:, :].copy(); K = 5
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        sgn = -1.0 if (flip and (step // period) % 2 == 1) else 1.0
        drift = sgn * cross_mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()
        common = idx_trend * vol
        if idx_revert > 0.0:
            idx = names.mean(0)
            if idx.shape[0] > 30: common += -idx_revert * (idx[-1] - idx[-30])
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + common + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :nDays] = prc
    return full

def evalbook(P, S, E, accel):
    R._SIG.clear(); R._RET.clear(); R._ICD.clear(); R._AZ.clear(); R._XC.clear()
    if accel: R.ALGO_P_FAST, R.ALGO_TCRIT_FAST = 3, 2.0
    else:     R.ALGO_P_FAST, R.ALGO_TCRIT_FAST = R.ALGO_P, R.ALGO_TCRIT      # no-op
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; tot = []; algo = []
    for t in range(S, E + 1):
        cur = P[:, t - 1]
        if t > S: algo.append(cp[0] * (cur - P[:, t - 2])[0])
        newPos = R.getMyPosition(P[:, :t]) if t < E else cp
        newPos = np.clip(newPos, -(dlr / cur).astype(int), (dlr / cur).astype(int)).astype(int)
        dP = newPos - cp; cash -= cur.dot(dP) + comm; comm = np.sum(cur * np.abs(dP) * commRate); cp = newPos
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: tot.append(pl)
    return np.array(tot).sum(), np.array(algo).sum()

cases = [
    ("real 500-750",         prc,                                 500, 750),
    ("B uptrend 150d",       make_ext(idx_trend=0.6, T_ext=150),  nDays, nDays + 150),
    ("B uptrend 300d",       make_ext(idx_trend=0.6, T_ext=300),  nDays, nDays + 300),
    ("C mean-revert 150d",   make_ext(idx_revert=0.8, T_ext=150), nDays, nDays + 150),
    ("flip chop 150d",       make_ext(flip=True, T_ext=150),      nDays, nDays + 150),
]
print(f"{'case':<20}{'ACCEL-ON tot':>13}{'ON ALGO':>9}{'ACCEL-OFF tot':>14}{'OFF ALGO':>9}{'ALGO delta':>11}")
for name, P, S, E in cases:
    on_t, on_a = evalbook(P, S, E, True)
    off_t, off_a = evalbook(P, S, E, False)
    print(f"{name:<20}{on_t:>13.0f}{on_a:>9.0f}{off_t:>14.0f}{off_a:>9.0f}{on_a-off_a:>11.0f}")
R.ALGO_P_FAST, R.ALGO_TCRIT_FAST = 3, 2.0     # restore shipped
print("\nALGO delta = ON minus OFF, isolating the accelerant (same idio book both).")
print("want: real ~0 (inert); uptrend >0 (captures, esp 300d once window fills); mean-revert ~0 (safe); flip ~0 (no whipsaw).")
