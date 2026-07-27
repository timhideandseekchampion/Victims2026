"""Portfolio-level vol targeting: scale the ENTIRE book (idio + ALGO) by a single time-varying
factor based on the PORTFOLIO'S OWN trailing realized PnL volatility, not any per-name or per-leg
characteristic. Mechanically different from every size-modulation attempt tonight (all of which
touched relative weights between names/legs) -- this only touches overall exposure over time, the
way real-world vol-targeting overlays do. Since everything is already at its $ cap, this can only
reduce exposure on high-vol days, never increase it.
"""
import numpy as np, pandas as pd
import SAFE, SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P)


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


print("computing shipped SAFE_llvol full book (idio + ALGO, unchanged) ...")
shipped_pos = np.zeros((nInst, nt))
for k in range(130, nt):
    cur = P[:, k]; lim = (dlr / cur).astype(int)
    shipped_pos[:, k] = np.clip(np.asarray(M.getMyPosition(P[:, :k + 1])), -lim, lim).astype(int)
print("done")

print("computing the shipped strategy's own daily realized PnL series (for trailing vol) ...")
daily_pnl = np.zeros(nt)
curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None
for tt in range(131, nt):
    cur = P[:, tt - 1]
    newPos = shipped_pos[:, tt - 1]
    if prevCur is not None:
        pl = curPos * (cur - prevCur) - comm_vec
        daily_pnl[tt - 1] = pl.sum()
    dP = newPos - curPos
    comm_vec = commRate * np.abs(dP) * cur
    prevCur = cur; curPos = newPos

OLD = (500, 750); NEW = (750, nt); end_days = list(range(400, nt + 1, 10))


def build_scaled_pos(vol_lb, target_pctile, min_scale):
    POS = np.zeros((nInst, nt))
    trailing_vols = []
    for k in range(250, nt):
        lo = max(131, k - vol_lb)
        hist = daily_pnl[lo:k]
        trailing_vol = hist.std() if len(hist) > 20 else None
        trailing_vols.append(trailing_vol)
    valid_vols = [v for v in trailing_vols if v is not None and v > 0]
    target_vol = np.percentile(valid_vols, target_pctile) if valid_vols else 1.0

    for k in range(250, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        lo = max(131, k - vol_lb)
        hist = daily_pnl[lo:k]
        trailing_vol = hist.std() if len(hist) > 20 else target_vol
        scale = min(1.0, target_vol / (trailing_vol + 1e-9)) if trailing_vol > 0 else 1.0
        scale = max(scale, min_scale)
        POS[:, k] = np.clip(np.round(shipped_pos[:, k] * scale), -lim, lim)
    return POS


base_POS = shipped_pos.copy()
base_scs = np.array([window(base_POS, E - NUMTEST, E)["score"] for E in end_days if E >= 500])
wo0 = window(base_POS, *OLD); wn0 = window(base_POS, *NEW)
print(f"\n{'config':<34}{'OLD':>8}{'NEW':>8}{'rmean':>8}{'rfloor':>9}{'n_worse':>9}")
print(f"{'shipped (no vol targeting)':<34}{wo0['score']:>8.1f}{wn0['score']:>8.1f}{base_scs.mean():>8.1f}{base_scs.min():>9.1f}")

for vol_lb, pctile, min_scale in [(20, 75, 0.3), (20, 90, 0.3), (60, 75, 0.3), (60, 90, 0.3),
                                    (20, 75, 0.5), (60, 75, 0.5), (20, 60, 0.3), (60, 60, 0.3)]:
    POS = build_scaled_pos(vol_lb, pctile, min_scale)
    scs = np.array([window(POS, E - NUMTEST, E)["score"] for E in end_days if E >= 500])
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    nworse = int((scs < base_scs).sum())
    print(f"{'lb='+str(vol_lb)+',pctile='+str(pctile)+',minsc='+str(min_scale):<34}{wo['score']:>8.1f}{wn['score']:>8.1f}{scs.mean():>8.1f}{scs.min():>9.1f}{nworse:>9}/{len(scs)}")
