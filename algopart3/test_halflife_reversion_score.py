"""Actually trade it: build a pure cross-sectional reversion book (no ridge at all, isolating just
the reversion signal) two ways -- (a) SAFE.py's shipped REV_W=10 fixed window for every stock, (b)
each stock's OWN measured OU half-life as its window -- and score both with the same eval-mirroring
accounting used all night. Also sweep a few UNIFORM window lengths for context (is there a better
single global window, separate from the "does per-stock customization help" question).
"""
import numpy as np, pandas as pd

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)
r = np.diff(logp, axis=1)
r0 = r[0]


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


beta = np.array([np.polyfit(r0, r[k], 1)[0] for k in range(nInst)])
resid_lvl_full = [np.cumsum(r[k] - beta[k] * r0) for k in range(1, nInst)]
per_stock_window = {}
for idx, s in enumerate(resid_lvl_full):
    s = s - s.mean()
    phi = np.polyfit(s[:-1], s[1:], 1)[0]
    hl = -np.log(2) / np.log(phi) if 0 < phi < 1 else 10.0
    per_stock_window[idx + 1] = int(np.clip(round(hl), 5, 300))
print("per-stock window (rounded half-life, days):", {names[j]: w for j, w in list(per_stock_window.items())[:8]}, "...")

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_pos_uniform(REV_W):
    POS = np.zeros((nInst, nt))
    for k in range(REV_W + 5, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        rr = logp[1:, k] - logp[1:, k - REV_W]
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        POS[1:, k] = np.clip(np.sign(rv) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


def build_pos_per_stock():
    POS = np.zeros((nInst, nt))
    maxw = max(per_stock_window.values())
    for k in range(maxw + 5, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        rr = np.array([logp[j, k] - logp[j, k - per_stock_window[j]] for j in range(1, nInst)])
        rr = rr - rr.mean()
        rv = -rr / (rr.std() + 1e-12)
        POS[1:, k] = np.clip(np.sign(rv) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    return POS


print(f"\n{'config':<28}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}")
for REV_W in (5, 10, 20, 40, 60, 100, 139, 200):
    POS = build_pos_uniform(REV_W)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    mark = "  <-- shipped" if REV_W == 10 else ""
    print(f"{'uniform REV_W='+str(REV_W):<28}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>8.1f}{min(scs):>9.1f}{mark}")

POS_ps = build_pos_per_stock()
wo = window(POS_ps, *OLD); wn = window(POS_ps, *NEW)
scs = [window(POS_ps, E - NUMTEST, E)["score"] for E in end_days]
print(f"{'per-stock half-life window':<28}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>8.1f}{min(scs):>9.1f}")
