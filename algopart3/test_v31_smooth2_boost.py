"""
test_v31_smooth2_boost.py -- a SECOND, parallel boost term testing whether a 2-day-SMOOTHED version
of a candidate leader's return has additional predictive power, on top of v22's existing single-day
lead-lag boost. Motivated by an externally-sourced, independently-verified finding (this session's
own re-derivation): pair-signal IC is strongest at the intended 1-day horizon (0.076) but decays,
not vanishes, at 2d/3d (0.055/0.038) -- suggesting the lead-lag relationship persists a bit beyond a
single day, which the existing mechanism structurally can't see (it only ever looks at yesterday's
single-day return).

MECHANISM (fits the existing 1-step-ahead prediction framework -- doesn't need a new holding
period): rs_smooth[:,m] = avg(rs[:,m], rs[:,m+1]) -- a 2-day rolling average of each name's own
return. This is used as BOTH the candidate-selection feature and the predictive signal (same
Bonferroni-significance + trailing-IC-validation gating as the original boost), but targeting the
follower's return ONE DAY AFTER the smoothed window ends (i.e. still predicting the SAME "tomorrow"
target the main wz already predicts) -- so this is a genuinely separate signal, addable to the same
wz, not a new prediction horizon requiring new holding logic.

BOOST_SMOOTH_K controls how much this new term contributes, added on top of v22's existing boost.
BOOST_SMOOTH_K=0 must reproduce v22 exactly (sanity-checked below).

Run: python3 test_v31_smooth2_boost.py
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


def _pairwise_boost_smooth2(rs):
    """Parallel to V22._pairwise_boost: uses a 2-day rolling-average return as the candidate feature
    and predictive signal, targeting the follower's return one day after the smoothing window ends
    (still the same 1-step-ahead target the main forecast predicts)."""
    n, T = rs.shape
    boost = np.zeros(n)
    if T < V22.BOOST_MIN_DAY + 1:
        return boost
    rs_smooth = (rs[:, 1:] + rs[:, :-1]) / 2.0        # (n, T-1); rs_smooth[:,m] known as of day m+1
    Ts = rs_smooth.shape[1]                            # = T-1
    Xi_full = rs_smooth[:, :-1]                         # (n, Ts-1) = (n, T-2)
    Yj = rs[:, 2:]                                      # (n, T-2)
    n_samples = Xi_full.shape[1]
    thr = V22._sig_threshold(n_samples)
    vol_causal = np.nanstd(Xi_full, axis=1)
    cand_idx = np.argsort(-vol_causal)[:V22.BOOST_N_CANDIDATES]
    Xi = Xi_full[cand_idx]
    C = V22._corrmat(Xi, Yj)
    for j in range(n):
        col = C[:, j].copy()
        cand_pos = np.where(cand_idx == j)[0]
        if len(cand_pos):
            col[cand_pos[0]] = np.nan
        if np.all(np.isnan(col)):
            continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr:
            continue
        i = cand_idx[ci]
        lead = rs_smooth[i]
        scale = np.nanstd(lead[max(0, Ts - 1 - V22.BOOST_SCALE_W):Ts - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** V22.BOOST_P
        a = max(0, Ts - 1 - V22.BOOST_IC_L)
        xs = lead_boost[a:Ts - 1]; ys = rs[j, a + 2:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12:
            continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0:
            continue
        boost[j] = lead_boost[-1]
    return boost


if __name__ == "__main__":
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nt = P_.shape
    logp_full = np.log(P_)
    n_names = nInst - 1
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250
    days = list(range(V22.WARMUP, nt))

    print("=== precompute v22 baseline ===")
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

    WZ_FULL = np.zeros((n_names, nt))   # v22's complete wz (post-fade), for the gain=0 sanity check
    BOOST_SMOOTH_RAW = np.zeros((n_names, nt))

    print("=== precompute: v22's full wz + the new smooth-2 boost term (independent of BOOST_SMOOTH_K) ===")
    reset(V22)
    for t in days:
        prcSoFar = P_[:, :t]
        WZ_FULL[:, t - 1] = V22._idio_signal(prcSoFar)
        logp = np.log(prcSoFar)
        r = logp[:, 1:] - logp[:, :-1]
        BOOST_SMOOTH_RAW[:, t - 1] = _pairwise_boost_smooth2(r[1:])
    print("  done.\n")

    engaged = int((BOOST_SMOOTH_RAW != 0).any(axis=0).sum())
    print(f"  smooth-2 boost engages (nonzero for >=1 name) on {engaged}/{len(days)} days\n")

    def build_pos(smooth_k):
        POS = np.zeros((nInst, nt))
        POS[0, :] = POS22[0, :]
        for t in days:
            idx = t - 1
            wz = WZ_FULL[:, idx] + smooth_k * BOOST_SMOOTH_RAW[:, idx]
            cur = P_[1:, idx]; lim = (dlr[1:] / cur).astype(int)
            POS[1:, idx] = np.clip(np.sign(wz) * (dlr[1:] / cur), -lim, lim).astype(int)
        return POS

    POS_check = build_pos(0.0)
    max_diff = np.max(np.abs(POS_check - POS22))
    print(f"=== sanity check: BOOST_SMOOTH_K=0 must reproduce v22 exactly ===\n"
          f"  max|diff|={max_diff:.2e} (should be 0)\n")
    if max_diff > 0:
        print("  *** WARNING: BOOST_SMOOTH_K=0 does not reproduce v22 -- do not trust results below. ***\n")

    print(f"{'BOOST_SMOOTH_K':>15}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    for k in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        POS = build_pos(k)
        curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
        win250 = wscore(POS, P_, 250, 500, nInst); old = wscore(POS, P_, 500, 750, nInst)
        new = wscore(POS, P_, 750, nt, nInst)
        n_worse = int((curve < curve22).sum()); n_better = int((curve > curve22).sum())
        passed = (win250 >= win250_22) and (old > old22) and (new > new22) and (curve.mean() > curve22.mean())
        tag = "PASS" if passed else ""
        print(f"{k:>15.1f}{win250:>9.1f}{old:>9.1f}{new:>9.1f}{curve.mean():>9.1f}"
              f"{curve.min():>9.1f}{n_worse:>9}/61{tag:>7}   n_better={n_better}/61")
