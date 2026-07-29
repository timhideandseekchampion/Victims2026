"""
test_g84_learned_rs_dense.py -- denser follow-up sweep on G84 (learned per-name rank-stability
blend weight), flagged in this repo's own README as "worth a denser follow-up sweep, not
investigated further" -- the original test only swept GAIN in {1,2,3,5} (cap=2.0 fixed) and CAP in
{1.5,2,3} (gain=3.0 fixed), a coarse single-axis-at-a-time sweep, and missed the bar by 1.3 on NEW.

MECHANISM (ported verbatim from test_batch100_G84.py's own description, re-derived fresh against
the CURRENT champion, not the stale v10-era baseline): the shipped rank-stability blend uses ONE
global RS_WEIGHT=0.015 applied uniformly to every name, every day. This computes each name's own
trailing-BOOST_IC_L(=250)-day causal IC of the RAW (pre-standardization) rank-stability signal
against realized idio return, and maps it to a per-name multiplier on the shipped weight:
  mult_i(t) = clip(1 + GAIN*ic_i(t), 0, CAP)
  w_i(t) = RS_WEIGHT * mult_i(t)
A name where RS has recently, causally, actually been predictive gets an amplified weight; a name
where it hasn't gets shrunk toward/at zero. Before MIN_DAY (not enough IC history yet), falls back
to the shipped uniform RS_WEIGHT exactly.

EFFICIENCY: everything upstream of the RS blend (ridge ensemble, beta-adjusted target, BLEND
reversion, pairwise boost) is IDENTICAL regardless of (gain, cap) -- precomputed once as
WZ_PREBOOST, matching this repo's own established pattern for parameter sweeps of this kind.

Run: python3 test_g84_learned_rs_dense.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v15 as V15

commRate = np.full(51, 1e-4); commRate[0] = 2e-5
dlr = np.full(51, 10_000.0); dlr[0] = 100_000.0


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
    logp = np.log(P_)
    r = np.diff(logp, axis=1)
    rs = r[1:]
    nIdio = rs.shape[0]
    days = list(range(V15.WARMUP, nt))
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

    print("=== precompute: everything upstream of the RS blend (identical regardless of gain/cap) ===")
    WZ_PREBOOST = np.zeros((nIdio, nt))
    RS_RAW = np.full((nIdio, nt), np.nan)
    for t in days:
        prcSoFar = P_[:, :t]
        # prcSoFar has t price columns => real _idio_signal's internal log-return matrix has
        # t-1 columns (r = logp[:,1:]-logp[:,:-1]); the globally-precomputed `r` here must be
        # sliced to :t-1, NOT :t, to match exactly (t columns would leak one day of future price).
        rr = r[:, :t - 1]
        Y = V15._beta_adjusted_target(rr)
        fs = []
        for hl in V15.HALF_LIVES:
            B, mx, my = V15._ewls_ridge(rr[:, :-1].T, Y, hl, V15.RIDGE_A)
            pred = my + (rr[:, -1] - mx) @ B
            fi = pred - pred.mean()
            fs.append(fi / (fi.std() + 1e-12))
        wz = np.mean(fs, 0)
        if V15.BLEND > 0:
            rr2 = logp[1:, t - 1] - logp[1:, t - 1 - V15.REV_W]
            rr2 = rr2 - rr2.mean()
            rv = -rr2 / (rr2.std() + 1e-12)
            wz = (1 - V15.BLEND) * wz + V15.BLEND * rv
        boost = V15._pairwise_boost(rs[:, :t - 1])
        wz = wz + V15.BOOST_K * boost
        WZ_PREBOOST[:, t - 1] = wz
        rs_sig = V15._rank_stability_signal(logp[:, :t])
        if rs_sig is not None:
            RS_RAW[:, t - 1] = rs_sig
    print("  done.\n")

    L = V15.BOOST_IC_L
    MIN_DAY = max(V15.BOOST_MIN_DAY, V15.WARMUP + L)

    print("=== trailing per-name IC of RS_RAW (250d causal window) ===")
    IC_MAT = np.zeros((nIdio, nt))
    for t in range(MIN_DAY, nt):
        a = t - L
        xs = RS_RAW[:, a:t]; ys = rs[:, a:t]
        finite = np.isfinite(xs).all(axis=1)
        mx = xs.mean(1); my = ys.mean(1)
        vx = xs.var(1); vy = ys.var(1)
        cov = ((xs - mx[:, None]) * (ys - my[:, None])).mean(1)
        denom = np.sqrt(vx * vy)
        ok = finite & (denom > 1e-20)
        ic = np.zeros(nIdio)
        ic[ok] = cov[ok] / denom[ok]
        IC_MAT[:, t] = ic
    valid_ic = IC_MAT[:, MIN_DAY:]
    print(f"  mean IC={valid_ic.mean():.3f}  std={valid_ic.std():.3f}  frac>0={(valid_ic>0).mean():.3f}\n")

    def build_pos_learned(gain, cap):
        WZ = WZ_PREBOOST.copy()
        for t in days:
            idx = t - 1
            wz = WZ_PREBOOST[:, idx]
            s = RS_RAW[:, idx]
            if not np.isfinite(s).all():
                continue
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            day_scale = np.abs(wz).mean() + 1e-12
            if t >= MIN_DAY:
                mult = np.clip(1.0 + gain * IC_MAT[:, idx], 0.0, cap)
                w = V15.RS_WEIGHT * mult
            else:
                w = np.full(nIdio, V15.RS_WEIGHT)
            wz2 = (1 - w) * wz + w * s_z * day_scale
            # post-jump fade, identical to v15, applied after the (now per-name) RS blend
            if idio_r_ready[idx]:
                sigma = fade_sigma[:, idx]; jump = fade_jump[:, idx]
                flagged = np.abs(jump) > V15.FADE_K_SIGMA * (sigma + 1e-12)
                if flagged.any():
                    scale = np.abs(wz2).mean() + 1e-12
                    fade_dir = -np.sign(jump)
                    wz2 = wz2.copy()
                    wz2[flagged] = wz2[flagged] + V15.FADE_EXTRA_W * fade_dir[flagged] * scale
            WZ[:, idx] = wz2
        return WZ

    # precompute fade inputs once too (identical regardless of gain/cap)
    idio_r_ready = np.zeros(nt, dtype=bool)
    fade_sigma = np.zeros((nIdio, nt)); fade_jump = np.zeros((nIdio, nt))
    for t in days:
        idx = t - 1
        idio_r = rs[:, :t - 1]
        if idio_r.shape[1] >= V15.FADE_W + 1:
            idio_r_ready[idx] = True
            fade_sigma[:, idx] = idio_r[:, -1 - V15.FADE_W:-1].std(axis=1)
            fade_jump[:, idx] = idio_r[:, -1]

    # precompute the REAL ALGO leg once (identical regardless of gain/cap -- fully independent of
    # the idio side, same convention as batch100_common_gi.py's shared `algo_pos`)
    print("=== precompute the real ALGO leg once (independent of gain/cap) ===")
    algo_pos = np.zeros(nt)
    for t in range(130, nt):
        cur0 = P_[0, t]; lim0 = int(dlr[0] / cur0)
        algo_pos[t] = np.clip(V15._algo_vol_shares(logp[0, :t + 1], cur0, dlr[0]), -lim0, lim0)
    print("  done.\n")

    def wz_to_pos(WZ):
        POS = np.zeros((nInst, nt))
        for t in days:
            idx = t - 1
            wz = WZ[:, idx]
            cur = P_[1:, idx]; lim = (dlr[1:] / cur).astype(int)
            POS[1:, idx] = np.clip(np.sign(wz) * (dlr[1:] / cur), -lim, lim)
        POS[0, :] = algo_pos
        return POS

    print("=== sanity check: gain=0 (mult==1 everywhere) must reproduce v15's plain RS blend ===")
    WZ_plain = build_pos_learned(0.0, 2.0)
    WZ_v15 = np.zeros((nIdio, nt))
    for t in days:
        idx = t - 1
        wz = WZ_PREBOOST[:, idx]
        s = RS_RAW[:, idx]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - V15.RS_WEIGHT) * wz + V15.RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        if idio_r_ready[idx]:
            sigma = fade_sigma[:, idx]; jump = fade_jump[:, idx]
            flagged = np.abs(jump) > V15.FADE_K_SIGMA * (sigma + 1e-12)
            if flagged.any():
                scale = np.abs(wz).mean() + 1e-12
                fade_dir = -np.sign(jump)
                wz = wz.copy()
                wz[flagged] = wz[flagged] + V15.FADE_EXTRA_W * fade_dir[flagged] * scale
        WZ_v15[:, idx] = wz
    max_diff = np.nanmax(np.abs(WZ_plain - WZ_v15))
    print(f"  max|diff| gain=0 vs plain v15 blend: {max_diff:.2e} (should be ~0)\n")

    POS_base = wz_to_pos(WZ_v15)
    curve_base = np.array([wscore(POS_base, P_, E - NUMTEST, E, nInst) for E in end_days])
    old_base = wscore(POS_base, P_, 500, 750, nInst); new_base = wscore(POS_base, P_, 750, nt, nInst)
    print(f"reconstructed v15 baseline (should match OLD=885.8 NEW=913.8 rmean=917.3 rfloor=720.7): "
          f"OLD={old_base:.1f} NEW={new_base:.1f} rmean={curve_base.mean():.1f} rfloor={curve_base.min():.1f}")
    if abs(old_base - 885.8) > 1.0 or abs(new_base - 913.8) > 1.0:
        print("  *** WARNING: reconstructed baseline does NOT match v15 -- do not trust results below. ***")
    print()

    def evaluate(gain, cap):
        WZ = build_pos_learned(gain, cap)
        POS = wz_to_pos(WZ)
        curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
        old = wscore(POS, P_, 500, 750, nInst); new = wscore(POS, P_, 750, nt, nInst)
        n_worse = int((curve < curve_base).sum())
        passed = (old > old_base) and (new > new_base) and (curve.mean() > curve_base.mean())
        return dict(gain=gain, cap=cap, old=old, new=new, rmean=curve.mean(), rfloor=curve.min(),
                    n_worse=n_worse, passed=passed)

    GAINS = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]
    CAPS = [1.5, 2.0, 2.5, 3.0]
    print(f"=== dense 2D grid: GAIN x CAP ===")
    print(f"{'gain':>6}{'cap':>6}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    results = []
    for g in GAINS:
        for c in CAPS:
            r_ = evaluate(g, c)
            results.append(r_)
            tag = "PASS" if r_["passed"] else ""
            print(f"{g:>6.1f}{c:>6.1f}{r_['old']:>9.1f}{r_['new']:>9.1f}{r_['rmean']:>9.1f}"
                  f"{r_['rfloor']:>9.1f}{r_['n_worse']:>9}/61{tag:>7}")

    passing = [r_ for r_ in results if r_["passed"]]
    print(f"\n{len(passing)}/{len(results)} configs pass (OLD+NEW+rmean jointly beat v15 idio-only baseline).")
    for r_ in sorted(results, key=lambda r_: -r_["rmean"])[:8]:
        print(f"  gain={r_['gain']:.1f} cap={r_['cap']:.1f}  rmean={r_['rmean']:.1f}  n_worse={r_['n_worse']}/61")
