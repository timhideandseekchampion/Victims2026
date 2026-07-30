"""
test_v24_vol_scale.py -- sweeps (VOL_SCALE_WIN, VOL_SCALE_MIN, VOL_SCALE_MAX) for
SAFE_llboost_v24.py (v22 + volatility-scaled idio position sizing) against the real-data bar
(WIN250=day 250-500, OLD=500-750, NEW=750-1000, rolling rmean/rfloor, n_worse/61) vs v22.

EFFICIENCY: walks V22's own getMyPosition once to get the exact wz/killed sequence (V24's signal
generation is byte-identical to V22's -- only the position-sizing step differs), then applies each
vol-scale config as a cheap post-processing step, instead of re-running the ridge/boost pipeline
once per swept config.

Run: python3 test_v24_vol_scale.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v22 as V22

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0


def reset(mod):
    for name in ("_SIG", "_FB", "_RET", "_XC", "_ICD", "_PN"):
        if hasattr(mod, name):
            getattr(mod, name).clear()
    mod._PREV_ALGO_SHARES = 0; mod._PREV_T = -1; mod._DLR = None


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, P_, S, E, nInst):
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = P_[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


if __name__ == "__main__":
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nt = P_.shape
    logp_full = np.log(P_)
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

    print("=== precompute v22's wz/killed sequence + baseline positions once ===")
    reset(V22)
    POS22 = np.zeros((nInst, nt))
    WZ_HIST = np.full((nInst - 1, nt), np.nan)
    KILLED = np.zeros(nt, dtype=bool)
    for t in range(1, nt):
        prcSoFar = P_[:, :t]
        p = np.asarray(V22.getMyPosition(prcSoFar))
        lim = (dlr / prcSoFar[:, -1]).astype(int)
        POS22[:, t - 1] = np.clip(p, -lim, lim).astype(int)
        if t >= V22.WARMUP:
            ready = t >= V22.WARMUP + V22.ROT_W + max(V22.ROT_P, V22.KILL_P)
            chosen = V22._choose(t) if ready else "champ"
            wz = V22._sig_at(chosen, t)
            WZ_HIST[:, t - 1] = wz
            KILLED[t - 1] = ready and V22._kill(t, chosen)
    curve22 = np.array([wscore(POS22, P_, E - NUMTEST, E, nInst) for E in end_days])
    win250_22 = wscore(POS22, P_, 250, 500, nInst); old22 = wscore(POS22, P_, 500, 750, nInst)
    new22 = wscore(POS22, P_, 750, nt, nInst)
    print(f"  v22: WIN250={win250_22:.1f}  OLD={old22:.1f}  NEW={new22:.1f}  "
          f"rmean={curve22.mean():.1f}  rfloor={curve22.min():.1f}\n")

    def vol_scale_hist(win, vmin, vmax):
        r_full = logp_full[1:, 1:] - logp_full[1:, :-1]   # 50 x (nt-1), r_full[:,k]=day k+1 return
        n = r_full.shape[0]
        scale_hist = np.ones((n, nt))
        for t in range(1, nt):
            idx = t - 1
            hi = t - 1  # returns available up to prcSoFar=P_[:,:t] => r_full[:, :t-1]
            if hi < win:
                continue
            vol = r_full[:, hi - win:hi].std(axis=1)
            ok = vol > 1e-12
            if not ok.any():
                continue
            med = np.median(vol[ok])
            sc = np.ones(n); sc[ok] = med / vol[ok]
            scale_hist[:, idx] = np.clip(sc, vmin, vmax)
        return scale_hist

    def build_pos(win, vmin, vmax, min_day):
        scale_hist = vol_scale_hist(win, vmin, vmax)
        POS = np.zeros((nInst, nt))
        POS[0, :] = POS22[0, :]
        for t in range(V22.WARMUP, nt):
            idx = t - 1
            if KILLED[idx]:
                continue
            wz = WZ_HIST[:, idx]
            scale = scale_hist[:, idx] if t >= min_day else np.ones(nInst - 1)
            cur = P_[1:, idx]; lim = (dlr[1:] / cur).astype(int)
            POS[1:, idx] = np.clip(np.sign(wz) * (dlr[1:] * scale / cur), -lim, lim).astype(int)
        return POS

    # sanity check: vmin=vmax=1.0 must reproduce v22 exactly
    POS_check = build_pos(60, 1.0, 1.0, 200)
    max_diff = np.max(np.abs(POS_check - POS22))
    print(f"=== sanity check: clip=[1,1] (no-op) must reproduce v22 exactly ===\n"
          f"  max|diff|={max_diff:.2e} (should be 0)\n")
    if max_diff > 0:
        print("  *** WARNING: no-op config does not reproduce v22 -- do not trust results below. ***\n")

    print(f"{'WIN':>5}{'MIN':>6}{'MAX':>6}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    configs = [
        (20, 0.5, 2.0), (40, 0.5, 2.0), (60, 0.5, 2.0), (90, 0.5, 2.0),
        (60, 0.33, 3.0), (60, 0.7, 1.5), (60, 0.8, 1.25),
    ]
    for win, vmin, vmax in configs:
        POS = build_pos(win, vmin, vmax, 200)
        curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
        win250 = wscore(POS, P_, 250, 500, nInst); old = wscore(POS, P_, 500, 750, nInst)
        new = wscore(POS, P_, 750, nt, nInst)
        n_worse = int((curve < curve22).sum()); n_better = int((curve > curve22).sum())
        passed = (win250 >= win250_22) and (old > old22) and (new > new22) and (curve.mean() > curve22.mean())
        tag = "PASS" if passed else ""
        print(f"{win:>5}{vmin:>6.2f}{vmax:>6.2f}{win250:>9.1f}{old:>9.1f}{new:>9.1f}{curve.mean():>9.1f}"
              f"{curve.min():>9.1f}{n_worse:>9}/61{tag:>7}   n_better={n_better}/61")
