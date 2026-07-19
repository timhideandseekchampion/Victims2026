"""verify_accel.py — the accelerated index fade->trend gate, four honest checks:
 (1) REAL 500-750  : inert -> SAFE_rotate == SAFE_live == SAFE_lldollar, score 694
 (2) B index UPTREND: captures the trend (ALGO leg >> the fade-only baseline)
 (3) C index MEAN-REVERTING (+ names momentum): must NOT force trend -> ALGO leg ~= fade baseline
 (4) FLIP chop      : no whipsaw blow-up on the ALGO leg
 + SAFE_live must stay position-identical to SAFE_rotate everywhere.
SAFE_lldollar = the plain fade-only book (no rotation, no trend gate) = the safety reference.
"""
import numpy as np, pandas as pd
import SAFE_rotate as R, SAFE_live as LV, SAFE_lldollar as LL

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
        drift = sgn * cross_mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()   # cross-sectional
        common = idx_trend * vol
        if idx_revert > 0.0:                                                              # mean-reverting index
            idx = names.mean(0)
            if idx.shape[0] > 30:
                common += -idx_revert * (idx[-1] - idx[-30])
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + common + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :nDays] = prc
    return full

def clear(M):
    for c in ("_SIG", "_RET", "_ICD", "_AZ", "_XC"):
        getattr(M, c, {}).clear()

def evalbook(M, P, S, E):
    clear(M)
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; tot = []; algo = []; pos_by_t = {}
    for t in range(S, E + 1):
        cur = P[:, t - 1]
        if t > S:
            algo.append(cp[0] * (cur - P[:, t - 2])[0])
        newPos = M.getMyPosition(P[:, :t]) if t < E else cp
        newPos = np.clip(newPos, -(dlr / cur).astype(int), (dlr / cur).astype(int)).astype(int)
        pos_by_t[t] = newPos
        dP = newPos - cp; cash -= cur.dot(dP) + comm; comm = np.sum(cur * np.abs(dP) * commRate); cp = newPos
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: tot.append(pl)
    return np.array(tot), np.array(algo), pos_by_t

def score(p):
    mu, sd = p.mean(), p.std(); return mu * (np.sqrt(250) * mu / sd) ** 2 / ((np.sqrt(250) * mu / sd) ** 2 + 1) if mu > 0 else mu

cases = [("(1) REAL 500-750", prc, 500, 750),
         ("(2) B index UPTREND", make_ext(idx_trend=0.6), nDays, nDays + 150),
         ("(3) C index MEAN-REVERT", make_ext(idx_revert=0.8), nDays, nDays + 150),
         ("(4) FLIP chop", make_ext(flip=True), nDays, nDays + 150)]

print(f"{'case':<26}{'ROTATE tot':>12}{'ROT ALGO':>10}{'FADE-only tot':>14}{'FADE ALGO':>11}{'LV==ROT':>9}")
for name, P, S, E in cases:
    rt, ra, rp = evalbook(R, P, S, E)
    lt, la, lp = evalbook(LV, P, S, E)
    ll, lla, llp = evalbook(LL, P, S, E)
    diff = max(int(np.abs(rp[t] - lp[t]).max()) for t in rp)
    extra = f"  score R={score(rt):.0f} LL={score(ll):.0f}" if "REAL" in name else ""
    print(f"{name:<26}{rt.sum():>12.0f}{ra.sum():>10.0f}{ll.sum():>14.0f}{lla.sum():>11.0f}{diff:>9}{extra}")
print("\nchecks: (1) REAL score 694 & LV==ROT 0 diff;  (2) ROT ALGO >> FADE ALGO (captured trend);")
print("(3) ROT ALGO ~= FADE ALGO (stayed faded on a mean-reverting index -> accel is SAFE);")
print("(4) ROT ALGO not blown up vs FADE (no whipsaw). LV==ROT must be 0 everywhere.")
