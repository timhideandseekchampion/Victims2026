"""
test_v29_boostk_learned.py -- replaces v22's flat BOOST_K=1.5 (how much the pairwise boost
contributes to the overall idio forecast) with a confidence multiplier learned from the boost
signal's OWN trailing pooled (across all 50 names, not per-name -- same lesson as the ridge-weight
and RS-weight ideas) realized IC against actual returns.

Mechanism: mult(t) = clip(1 + GAIN*pooled_ic_boost(t), FLOOR, CAP), BOOST_K_eff(t) = BOOST_K*mult(t).
pooled_ic_boost(t) is the trailing L-day IC of the RAW (pre-BOOST_K) boost value vs realized return,
pooled ONLY over (name, day) pairs where the boost actually fired (boost != 0 -- zeros are "no
signal", not meaningful data points, and would just dilute the estimate). Falls back to the flat
BOOST_K before enough fired-boost observations exist.

EFFICIENCY: everything independent of the boost-scaling choice (the ridge ensemble + BLEND
reversion, the raw boost values themselves, RS_RAW/G84's IC matrix, fade inputs) is precomputed
once; only the boost-scale recombination + RS-blend day_scale + fade are cheaply redone per swept
(GAIN, CAP, WINDOW) config.

Run: python3 test_v29_boostk_learned.py
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
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250
    days = list(range(V22.WARMUP, nt))

    print("=== precompute v22 baseline + everything independent of boost-scale choice ===")
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

    WZ_PREBOOST = np.zeros((n_names, nt))
    BOOST_RAW = np.zeros((n_names, nt))
    FADE_SIGMA = np.zeros((n_names, nt)); FADE_JUMP = np.zeros((n_names, nt))
    FADE_READY = np.zeros(nt, dtype=bool)

    print("=== precompute: ridge+BLEND (pre-boost), raw boost, fade inputs ===")
    for t in days:
        prcSoFar = P_[:, :t]
        logp = np.log(prcSoFar)
        r = logp[:, 1:] - logp[:, :-1]
        idx = t - 1

        Y = V22._beta_adjusted_target(r)
        fs = []
        for hl in V22.HALF_LIVES:
            B, mx, my = V22._ewls_ridge(r[:, :-1].T, Y, hl, V22.RIDGE_A)
            pred = my + (r[:, -1] - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz0 = np.mean(fs, 0)
        if V22.BLEND > 0:
            rr = logp[1:, -1] - logp[1:, -1 - V22.REV_W]
            rr = rr - rr.mean()
            rv = -rr / (rr.std() + 1e-12)
            wz0 = (1 - V22.BLEND) * wz0 + V22.BLEND * rv
        WZ_PREBOOST[:, idx] = wz0

        BOOST_RAW[:, idx] = V22._pairwise_boost(r[1:])

        idio_r = r[1:]
        if idio_r.shape[1] >= V22.FADE_W + 1:
            FADE_READY[idx] = True
            FADE_SIGMA[:, idx] = idio_r[:, -1 - V22.FADE_W:-1].std(axis=1)
            FADE_JUMP[:, idx] = idio_r[:, -1]
    print("  done.\n")

    RS_RAW = V22._rs_raw_hist(logp_full)
    L84 = V22.BOOST_IC_L
    min_day_g84 = max(V22.BOOST_MIN_DAY, V22.WARMUP + L84)
    IC_MAT = np.zeros((n_names, nt))
    for t in range(min_day_g84, nt):
        idx = t - 1
        a = idx - L84
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

    def pooled_ic_boost_hist(win, min_day):
        pooled = np.full(nt, np.nan)
        for t in range(min_day, nt + 1):
            idx = t - 1
            a = idx - win
            if a < 0 or idx > ret_full.shape[1]:
                continue
            xs = BOOST_RAW[:, a:idx]; ys = ret_full[:, a:idx]
            nz = xs != 0
            ok = nz & np.isfinite(xs) & np.isfinite(ys)
            if ok.sum() < 100 or xs[ok].std() < 1e-12:
                continue
            pooled[idx] = np.corrcoef(xs[ok], ys[ok])[0, 1]
        return pooled

    def build_pos(gain, cap, win, min_day):
        pooled = pooled_ic_boost_hist(win, min_day)
        POS = np.zeros((nInst, nt))
        POS[0, :] = POS22[0, :]
        for t in days:
            idx = t - 1
            ic = pooled[idx]
            mult = np.clip(1.0 + gain * ic, 0.0, cap) if (np.isfinite(ic) and t >= min_day) else 1.0
            wz = WZ_PREBOOST[:, idx] + V22.BOOST_K * mult * BOOST_RAW[:, idx]

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

    POS_check = build_pos(0.0, 1.0, 250, nt + 1)
    max_diff = np.max(np.abs(POS_check - POS22))
    print(f"=== sanity check: gain=0 (mult==1 always) must reproduce v22 exactly ===\n"
          f"  max|diff|={max_diff:.2e} (should be 0)\n")
    if max_diff > 0:
        print("  *** WARNING: gain=0 does not reproduce v22 -- do not trust results below. ***\n")

    print(f"{'gain':>6}{'cap':>6}{'win':>6}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    for win in (250, 500):
        min_day = max(V22.BOOST_MIN_DAY, V22.WARMUP + win)
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
