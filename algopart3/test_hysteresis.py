"""Test: the idio book (SAFE/LLVOL idio leg) is 'sign(wz) at full $10k every name every day' with
NO deadband — positions_data.json shows each of the 49 names flips sign ~113x per 250-day window
(~once every 2.2 days). That's very high turnover for a book that is always at max size regardless
of conviction. Hypothesis: many of those flips happen when |wz| is near zero (low conviction) and
are pure noise -> a small hysteresis deadband (hold the previous sign unless |wz| clears a threshold)
should cut commission-churn without giving up much of the real edge, and may even raise Sharpe (lower
variance from fewer noise-driven side-switches).

This rebuilds the idio wz series exactly as SAFE_llvol.py computes it (identical ridge ensemble +
reversion blend), applies a hysteresis deadband on top, and rescer using compute_diagnostics.py's
exact eval-mirroring accounting (lagged commission, integer clip, score = mu*SR^2/(SR^2+1)).
The ALGO leg (instrument 0) is left EXACTLY as SAFE_llvol computes it in every variant, so any score
delta is attributable purely to the idio-book entry rule.
"""
import json, numpy as np, pandas as pd
import SAFE_llvol as M

P = pd.read_csv("prices.txt", sep=r"\s+", header=0)
names = list(P.columns)
P = P.values.T.astype(float)
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


def wz_at(logp, r, t):
    """Reproduce SAFE_llvol's idio wz (49-vector) using history through day index t (0-based, t+1 days)."""
    rr_hist = r[:, :t]                                   # returns up to day t (t = nInst x t matrix slice)
    fs = []
    for hl in M.HALF_LIVES:
        B, mx, my = M._ewls_ridge(rr_hist[:, :-1].T, rr_hist[1:, 1:].T, hl, M.RIDGE_A)
        pred = my + (rr_hist[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    if M.BLEND > 0:
        rv_ = logp[1:, t] - logp[1:, t - M.REV_W]
        rv_ = rv_ - rv_.mean()
        rv = -rv_ / (rv_.std() + 1e-12)
        wz = (1 - M.BLEND) * wz + M.BLEND * rv
    return wz


print("precomputing wz series for all days (this replicates SAFE_llvol's idio forecast) ...")
logp = np.log(P)
r = logp[:, 1:] - logp[:, :-1]
WZ = {}
for t in range(M.WARMUP, nt):
    WZ[t] = wz_at(logp, r, t)
print(f"done: {len(WZ)} days")

# ---- baseline ALGO leg (unchanged, identical every variant) + baseline idio (no deadband) ----
def build_pos(thresh, min_hold=0):
    POS = np.zeros((nInst, nt))
    prev_sign = np.zeros(nInst - 1)
    hold_days = np.zeros(nInst - 1, dtype=int)
    for k in range(130, nt):
        cur = P[:, k]
        lim = (dlr / cur).astype(int)
        algo_pos = M._algo_vol_shares(logp[0, :k + 1], cur[0], dlr[0])
        if k in WZ:
            wz = WZ[k]
            new_sign = np.sign(wz)
            if thresh > 0 or min_hold > 0:
                flip_ok = (np.abs(wz) >= thresh) & (hold_days >= min_hold)
                keep = ~flip_ok & (prev_sign != 0)
                new_sign = np.where(keep, prev_sign, new_sign)
            hold_days = np.where(new_sign == prev_sign, hold_days + 1, 0)
            prev_sign = new_sign
        else:
            new_sign = prev_sign
        idio_pos = new_sign * (dlr[1:] / cur[1:])
        pos = np.concatenate(([algo_pos], idio_pos))
        POS[:, k] = np.clip(pos, -lim, lim).astype(int)
    return POS


def flips_per_name(POS, S, E):
    sg = np.sign(POS[1:, S:E])
    ch = np.abs(np.diff(sg, axis=1)) > 0
    return ch.sum(axis=1).mean()


OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

print(f"\n{'thresh':>7} {'min_hold':>9} {'OLD':>8} {'NEW':>8} {'roll_mean':>10} {'roll_floor':>11} {'flips/250d(new)':>16}")
for thresh, min_hold in [(0.0, 0), (0.1, 0), (0.2, 0), (0.3, 0), (0.5, 0), (0.75, 0), (1.0, 0),
                          (0.0, 1), (0.0, 2), (0.0, 3), (0.2, 2), (0.3, 2)]:
    POS = build_pos(thresh, min_hold)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    fl = flips_per_name(POS, NEW[0], NEW[1])
    print(f"{thresh:>7.2f} {min_hold:>9} {wo['score']:>8.1f} {wn['score']:>8.1f} {np.mean(scs):>10.1f} {min(scs):>11.1f} {fl:>16.1f}")
