import numpy as np, pandas as pd
import SAFE_llvol as M
import SAFE_llvol_vo as VO

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

lpA = logp[0]; r = np.diff(lpA); T = len(lpA)
vol = np.full(T, np.nan); vol[M.VOL_WIN:] = M._roll_std(r, M.VOL_WIN)
volz = np.full(T, np.nan)
for s in range(M.VOL_WIN + M.VOL_Z, T):
    wv = vol[s - M.VOL_Z:s]; volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]
edge = volz * ret1
valid = np.where(~np.isnan(edge))[0]; E0 = int(valid[0])

def ph_side_series(delta, lam, std_win=250):
    side = np.zeros(T, dtype=int)
    m_up = m_down = M_up_min = M_down_min = 0.0
    cur_side = 0; n_cp = 0; buf = []
    for t in range(E0, T):
        x = edge[t - 1] if t - 1 >= E0 else np.nan
        if t - 1 >= E0 and not np.isnan(x):
            buf.append(x)
            if len(buf) > std_win: buf.pop(0)
            sd = (np.std(buf) if len(buf) > 5 else 1.0) + 1e-12
            xz = x / sd
            m_up += xz - delta; m_down += -xz - delta
            M_up_min = min(M_up_min, m_up); M_down_min = min(M_down_min, m_down)
            ph_up = m_up - M_up_min; ph_down = m_down - M_down_min
            if ph_up > lam:
                cur_side = 1; n_cp += 1
                m_up = m_down = M_up_min = M_down_min = 0.0
            elif ph_down > lam:
                cur_side = -1; n_cp += 1
                m_up = m_down = M_up_min = M_down_min = 0.0
        side[t] = cur_side
    return side, n_cp

def build_pos_from_side(side):
    POS = np.zeros((nInst, nt))
    for k in range(130, nt):
        cur = P[:, k]; lim = (dlr / cur).astype(int)
        fh = np.clip(volz[k], -3, 3) / 3.0 if not np.isnan(volz[k]) else 0.0
        av = M.SWITCH_GAIN * side[k] * fh * 100_000.0
        POS[0, k] = int(np.clip(np.clip(av, -dlr[0], dlr[0]) / cur[0], -lim[0], lim[0]))
    return POS

OLD = (500, 750); NEW = (750, nt)
end_days = list(range(400, nt + 1, 10))

vo_pos = np.zeros((nInst, nt))
for k in range(130, nt):
    cur0 = P[0, k]; lim0 = int(dlr[0] / cur0)
    vo_pos[0, k] = int(np.clip(VO._algo_vol_shares(lpA[:k + 1], cur0, dlr[0]), -lim0, lim0))
wo = window(vo_pos, *OLD); wn = window(vo_pos, *NEW)
scs = [window(vo_pos, E - NUMTEST, E)["score"] for E in end_days]
print(f"{'config':<36}{'OLD':>8}{'NEW':>8}{'roll_mean':>11}{'roll_floor':>12}{'cp(slow/fast)':>15}")
print(f"{'LLVOL_VO (baseline, shipped)':<36}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}{'':>15}")

single = ph_side_series(0.05, 10)[0]
POS = build_pos_from_side(single)
wo = window(POS, *OLD); wn = window(POS, *NEW)
scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
print(f"{'single PH(0.05,10) [prev best]':<36}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}{'':>15}")

# cache fast/slow side series across the sweep to avoid recompute
cache = {}
def get_side(delta, lam):
    key = (delta, lam)
    if key not in cache: cache[key] = ph_side_series(delta, lam)
    return cache[key]

configs = [
    (0.02, 8, 0.1, 25),
    (0.02, 8, 0.1, 35),
    (0.02, 10, 0.1, 30),
    (0.05, 10, 0.15, 30),
    (0.05, 10, 0.2, 40),
    (0.02, 6, 0.05, 20),
    (0.02, 6, 0.1, 20),
    (0.03, 8, 0.1, 25),
]
for fd, fl, sd_, sl in configs:
    fast_side, fcp = get_side(fd, fl)
    slow_side, scp = get_side(sd_, sl)
    combined = np.where((fast_side == slow_side), slow_side, 0)
    POS = build_pos_from_side(combined)
    wo = window(POS, *OLD); wn = window(POS, *NEW)
    scs = [window(POS, E - NUMTEST, E)["score"] for E in end_days]
    lbl = f"fast({fd},{fl})+slow({sd_},{sl})"
    print(f"{lbl:<36}{wo['score']:>8.1f}{wn['score']:>8.1f}{np.mean(scs):>11.1f}{min(scs):>12.1f}{str(scp)+'/'+str(fcp):>15}")
