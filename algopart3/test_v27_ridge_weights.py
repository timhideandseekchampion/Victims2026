"""
test_v27_ridge_weights.py -- learned per-half-life weighting for the ridge ensemble, replacing v22's
flat equal-weighted mean(fs, 0) over the 4 EWLS half-lives (250/500/1000/2000).

LESSON APPLIED FROM THE JUST-REJECTED DEADBAND EXPERIMENTS: per-NAME significance testing (even for a
single simple factor) is close to unsatisfiable here -- the market is closer to one-factor, so no
individual stock clears a Bonferroni-corrected bar over any realistic window. This idea avoids that
trap by testing significance/quality POOLED ACROSS ALL 50 NAMES (not per-name), so each half-life's
IC estimate has ~50x the effective sample size of a per-name test, and there are only 4 comparisons
(one per half-life) instead of 50 -- both make this comparable to how ALGO's own signal is judged
(a single, clean, low-multiple-comparison test), not to the per-name idio tests that just failed.

Mechanism: mult_hl(t) = clip(1 + GAIN*pooled_IC_hl(t), 0, CAP), pooled_IC_hl computed by flattening
(name, day) pairs of half-life hl's OWN standalone z-scored forecast vs realized return over a
trailing window (POOLED across all 50 names, not per-name). w_hl(t) = mult_hl(t) / sum(mult).
wz0(t) = sum_hl(w_hl(t) * fi_hl(t)), replacing the flat 0.25-each average. Falls back to flat
weighting before enough history exists.

EFFICIENCY: the ridge fit itself (4 half-lives x every day) and every step independent of the
ridge-combine choice (BLEND's reversion feature, the two-hop boost, the RS_RAW/G84 IC machinery,
the fade inputs) are precomputed ONCE; only the combine step + the parts of _idio_signal that flow
from it are recomputed per swept (GAIN, CAP, WINDOW) config.

Run: python3 test_v27_ridge_weights.py
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
    idio_logp = logp_full[1:]
    ret_full = idio_logp[:, 1:] - idio_logp[:, :-1]   # 50 x (nt-1)
    n_names = nInst - 1
    n_hl = len(V22.HALF_LIVES)
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250
    days = list(range(V22.WARMUP, nt))

    print("=== precompute v22 baseline + everything independent of ridge-combine weights ===")
    reset(V22)
    POS22 = np.zeros((nInst, nt))
    for t in range(1, nt):
        prcSoFar = P_[:, :t]
        p = np.asarray(V22.getMyPosition(prcSoFar))
        lim = (dlr / prcSoFar[:, -1]).astype(int)
        POS22[:, t - 1] = np.clip(p, -lim, lim).astype(int)
    curve22 = np.array([wscore(POS22, P_, E - NUMTEST, E, nInst) for E in end_days])
    win250_22 = wscore(POS22, P_, 250, 500, nInst); old22 = wscore(POS22, P_, 500, 750, nInst)
    new22 = wscore(POS22, P_, 750, nt, nInst)
    print(f"  v22: WIN250={win250_22:.1f}  OLD={old22:.1f}  NEW={new22:.1f}  "
          f"rmean={curve22.mean():.1f}  rfloor={curve22.min():.1f}")

    # 1) per-half-life raw z-scored ridge forecast, every day (the expensive part -- done once)
    FI_HIST = np.full((n_hl, n_names, nt), np.nan)
    # 2) BLEND's reversion feature, boost, RS_RAW, fade inputs -- all independent of ridge weights
    RV_HIST = np.full((n_names, nt), np.nan)
    BOOST_HIST = np.zeros((n_names, nt))
    FADE_SIGMA = np.zeros((n_names, nt)); FADE_JUMP = np.zeros((n_names, nt))
    FADE_READY = np.zeros(nt, dtype=bool)

    print("=== precompute: ridge fits (4 half-lives/day), boost, reversion, fade inputs ===")
    for t in days:
        prcSoFar = P_[:, :t]
        logp = np.log(prcSoFar)
        r = logp[:, 1:] - logp[:, :-1]
        idx = t - 1

        Y = V22._beta_adjusted_target(r)
        for hi, hl in enumerate(V22.HALF_LIVES):
            B, mx, my = V22._ewls_ridge(r[:, :-1].T, Y, hl, V22.RIDGE_A)
            pred = my + (r[:, -1] - mx) @ B
            fi = pred - pred.mean()
            FI_HIST[hi, :, idx] = fi / (fi.std() + 1e-12)

        if V22.BLEND > 0:
            rr = logp[1:, -1] - logp[1:, -1 - V22.REV_W]
            rr = rr - rr.mean()
            RV_HIST[:, idx] = -rr / (rr.std() + 1e-12)

        BOOST_HIST[:, idx] = V22._pairwise_boost(r[1:])

        idio_r = r[1:]
        if idio_r.shape[1] >= V22.FADE_W + 1:
            FADE_READY[idx] = True
            FADE_SIGMA[:, idx] = idio_r[:, -1 - V22.FADE_W:-1].std(axis=1)
            FADE_JUMP[:, idx] = idio_r[:, -1]
    print("  done.\n")

    # 3) RS_RAW full history + G84's per-name IC matrix (both independent of ridge weights)
    RS_RAW = V22._rs_raw_hist(logp_full)
    L = V22.BOOST_IC_L
    min_day_g84 = max(V22.BOOST_MIN_DAY, V22.WARMUP + L)
    IC_MAT = np.zeros((n_names, nt))
    for t in range(min_day_g84, nt):
        idx = t - 1
        a = idx - L
        xs = RS_RAW[:, a:idx]; ys = ret_full[:, a:idx]
        finite = np.isfinite(xs).all(axis=1)
        mx = xs.mean(1); my_ = ys.mean(1)
        vx = xs.var(1); vy = ys.var(1)
        cov = ((xs - mx[:, None]) * (ys - my_[:, None])).mean(1)
        denom = np.sqrt(vx * vy)
        ok = finite & (denom > 1e-20)
        ic = np.zeros(n_names)
        ic[ok] = cov[ok] / denom[ok]
        IC_MAT[:, idx] = ic
    RS_STD = np.nanstd(RS_RAW, axis=0); RS_MEAN = np.nanmean(RS_RAW, axis=0)

    def pooled_ic_per_hl(win, min_day):
        """pooled_ic[hi, idx] = pooled (all-names, trailing-win-days) IC of half-life hi's raw
        forecast vs realized return, as of day idx+1. NaN before enough history."""
        pooled = np.full((n_hl, nt), np.nan)
        for t in range(min_day, nt + 1):
            idx = t - 1
            a = idx - win
            if a < 0 or idx > ret_full.shape[1]:
                continue
            ys = ret_full[:, a:idx].ravel()
            for hi in range(n_hl):
                xs = FI_HIST[hi, :, a:idx].ravel()
                ok = np.isfinite(xs) & np.isfinite(ys)
                if ok.sum() < 200 or xs[ok].std() < 1e-12:
                    continue
                pooled[hi, idx] = np.corrcoef(xs[ok], ys[ok])[0, 1]
        return pooled

    def build_pos(gain, cap, win, min_day):
        pooled = pooled_ic_per_hl(win, min_day)
        POS = np.zeros((nInst, nt))
        POS[0, :] = POS22[0, :]
        for t in days:
            idx = t - 1
            ic_hl = pooled[:, idx]
            if np.isfinite(ic_hl).all() and t >= min_day:
                mult = np.clip(1.0 + gain * ic_hl, 0.0, cap)
                if mult.sum() < 1e-6:
                    w = np.full(n_hl, 1.0 / n_hl)
                else:
                    w = mult / mult.sum()
            else:
                w = np.full(n_hl, 1.0 / n_hl)
            wz = (w[:, None] * FI_HIST[:, :, idx]).sum(0)

            if V22.BLEND > 0 and np.isfinite(RV_HIST[:, idx]).all():
                wz = (1 - V22.BLEND) * wz + V22.BLEND * RV_HIST[:, idx]

            wz = wz + V22.BOOST_K * BOOST_HIST[:, idx]

            s = RS_RAW[:, idx]
            if np.isfinite(s).all():
                sstd = RS_STD[idx]
                s_z = (s - RS_MEAN[idx]) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(n_names)
                day_scale = np.abs(wz).mean() + 1e-12
                if t >= min_day_g84:
                    rw = V22.RS_WEIGHT * np.clip(1.0 + V22.G84_GAIN * IC_MAT[:, idx], 0.0, V22.G84_CAP)
                else:
                    rw = V22.RS_WEIGHT
                wz = (1 - rw) * wz + rw * s_z * day_scale

            if FADE_READY[idx]:
                sigma = FADE_SIGMA[:, idx]; jump = FADE_JUMP[:, idx]
                flagged = np.abs(jump) > V22.FADE_K_SIGMA * (sigma + 1e-12)
                if flagged.any():
                    scale = np.abs(wz).mean() + 1e-12
                    fade_dir = -np.sign(jump)
                    wz = wz.copy()
                    wz[flagged] = wz[flagged] + V22.FADE_EXTRA_W * fade_dir[flagged] * scale

            cur = P_[1:, idx]; lim = (dlr[1:] / cur).astype(int)
            POS[1:, idx] = np.clip(np.sign(wz) * (dlr[1:] / cur), -lim, lim).astype(int)
        return POS

    # sanity check: gain=0 (mult==1 always, flat weight) must reproduce v22 exactly
    POS_check = build_pos(0.0, 2.0, 120, nt + 1)
    max_diff = np.max(np.abs(POS_check - POS22))
    print(f"=== sanity check: gain=0 (flat 1/4 weights) must reproduce v22 exactly ===\n"
          f"  max|diff|={max_diff:.2e} (should be 0)\n")
    if max_diff > 0:
        print("  *** WARNING: gain=0 does not reproduce v22 -- do not trust results below. ***\n")

    print(f"{'gain':>6}{'cap':>6}{'win':>6}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    for win in (120, 250):
        min_day = max(V22.WARMUP + win, 400)
        for gain in (1.0, 2.0, 3.0, 5.0):
            for cap in (1.5, 2.0):
                POS = build_pos(gain, cap, win, min_day)
                curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
                win250 = wscore(POS, P_, 250, 500, nInst); old = wscore(POS, P_, 500, 750, nInst)
                new = wscore(POS, P_, 750, nt, nInst)
                n_worse = int((curve < curve22).sum()); n_better = int((curve > curve22).sum())
                passed = (win250 >= win250_22) and (old > old22) and (new > new22) and (curve.mean() > curve22.mean())
                tag = "PASS" if passed else ""
                print(f"{gain:>6.1f}{cap:>6.1f}{win:>6}{win250:>9.1f}{old:>9.1f}{new:>9.1f}{curve.mean():>9.1f}"
                      f"{curve.min():>9.1f}{n_worse:>9}/61{tag:>7}   n_better={n_better}/61")
