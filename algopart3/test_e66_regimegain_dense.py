"""
test_e66_regimegain_dense.py -- denser follow-up sweep on E66 (regime-conditional COMBINE_GAIN),
flagged in this repo's own README as "worth a denser follow-up sweep, not investigated further" --
the original test only tried 4 coarse single-axis perturbations ((12,16),(20,16),(16,12),(16,20))
around the shipped uniform COMBINE_GAIN=16.0 and missed the bar by 1.8 on rmean. This does a full
2D grid instead of perturbing one axis at a time.

The ALGO leg (_algo_vol_shares) is fully independent of the idio book -- confirmed by reading the
code, `_algo_vol_shares` never touches _SIG/_FB/_pairwise_boost or any idio state. So this tests
against v15's ALREADY-COMPUTED idio positions (cheap to reuse) and only recomputes the ALGO leg
per grid point -- the conclusion transfers identically to any idio variant (v15, v19, v20, or any
combination), since they all share the exact same, unmodified _algo_vol_shares.

Regime split: fh = clip(volz[tnow],-3,3)/3 (the same fast vol-z feature V15 already uses to pick
MOM_LB_SHORT vs MOM_LB_LONG) -- GAIN_HI applies when fh>0, GAIN_LO otherwise.

Run: python3 test_e66_regimegain_dense.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v15 as V15

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


def _algo_vol_shares_regime(lpA, cur0, cap_dol, gain_hi, gain_lo, state):
    """Faithful copy of V15._algo_vol_shares' VOL_MODE='switch', VOL_COMBINE=True branch (the only
    branch V15's shipped constants take), with COMBINE_GAIN split into a regime-conditional
    GAIN_HI/GAIN_LO pair. Uses its own LOCAL state dict, not V15's module globals."""
    T = len(lpA)
    tnow = T - 1
    have_prev = (tnow == state["prev_t"] + 1)
    if T < V15.VOL_WIN + V15.VOL_Z + 60:
        state["prev_t"] = tnow; state["prev_shares"] = 0
        return 0
    r = np.diff(lpA)
    vol = np.full(T, np.nan)
    vol[V15.VOL_WIN:] = V15._roll_std(r, V15.VOL_WIN)
    lo = max(V15.VOL_WIN + V15.VOL_Z, tnow - V15.IC_LOOKBACK)
    volz = np.full(T, np.nan)
    for s in range(lo, T):
        wv = vol[s - V15.VOL_Z:s]
        volz[s] = (vol[s] - wv.mean()) / (wv.std() + 1e-12)
    ret1 = np.full(T, np.nan); ret1[:T - 1] = lpA[1:] - lpA[:-1]

    def _ic(feat, L):
        a = max(0, tnow - L); xs = feat[a:tnow]; ys = ret1[a:tnow]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys = xs[ok], ys[ok]
        if xs.std() < 1e-12: return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _ic_ew(feat, HL, W):
        a = max(0, tnow - W); xs = feat[a:tnow]; ys = ret1[a:tnow]
        w = (0.5 ** (1.0 / HL)) ** ((tnow - 1) - np.arange(a, tnow))
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60: return None
        xs, ys, w = xs[ok], ys[ok], w[ok]; sw = w.sum()
        mx = (w * xs).sum() / sw; my = (w * ys).sum() / sw
        cxy = (w * (xs - mx) * (ys - my)).sum() / sw
        vx = (w * (xs - mx) ** 2).sum() / sw; vy = (w * (ys - my) ** 2).sum() / sw
        if vx < 1e-24 or vy < 1e-24: return None
        return float(cxy / np.sqrt(vx * vy))

    def _side(feat, fhv):
        icf = _ic(feat, V15.IC_FAST)
        if icf is None: return None
        sf = 1.0 if icf >= 0 else -1.0
        if not V15.IC_BLEND: return sf * fhv
        ics = [x for x in (_ic_ew(feat, hl, V15.IC_EW_W) for hl in V15.IC_EW_HL) if x is not None]
        if len(ics) < len(V15.IC_EW_HL): return sf * fhv
        ice = float(np.mean(ics))
        return (sf * fhv) if (ice >= 0) == (icf >= 0) else 0.0

    fh = np.clip(volz[tnow], -3, 3) / 3.0
    if np.isnan(fh):
        state["prev_t"] = tnow; state["prev_shares"] = 0
        return 0
    sig = _side(volz, fh)
    if sig is None:
        state["prev_t"] = tnow; state["prev_shares"] = 0
        return 0
    mom_lb = V15.MOM_LB_SHORT if fh > 0 else V15.MOM_LB_LONG
    mom = np.full(T, np.nan); mom[mom_lb:] = lpA[mom_lb:] - lpA[:-mom_lb]
    z10 = np.full(T, np.nan)
    for s in range(max(mom_lb + V15.VOL_Z, tnow - V15.IC_EW_W), T):
        wm = mom[s - V15.VOL_Z:s]; z10[s] = (mom[s] - wm.mean()) / (wm.std() + 1e-12)
    fhm = np.clip(z10[tnow], -3, 3) / 3.0
    msig = _side(z10, fhm) if not np.isnan(fhm) else None
    gain = gain_hi if fh > 0 else gain_lo
    if msig is not None:
        av = gain * (sig + msig) * 100_000.0
    else:
        av = V15.SWITCH_GAIN * sig * 100_000.0

    av = float(np.clip(av, -cap_dol, cap_dol))
    lim = int(cap_dol / cur0)
    if have_prev and tnow >= V15.DEADBAND_MIN_DAY and abs(av) < V15.DEADBAND_THRESH_FRAC * cap_dol:
        shares = int(np.clip(state["prev_shares"], -lim, lim))
    else:
        shares = int(np.clip(av / cur0, -lim, lim))
    state["prev_shares"] = shares
    state["prev_t"] = tnow
    return shares


if __name__ == "__main__":
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nt = P_.shape
    logp = np.log(P_)
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

    print("=== precompute v15's idio positions once (ALGO leg is independent, cheap to swap) ===")
    reset(V15)
    POS_BASE = np.zeros((nInst, nt))
    for t in range(1, nt):
        prcSoFar = P_[:, :t]
        p = np.asarray(V15.getMyPosition(prcSoFar))
        lim = (dlr / prcSoFar[:, -1]).astype(int)
        POS_BASE[:, t - 1] = np.clip(p, -lim, lim).astype(int)

    curve_base = np.array([wscore(POS_BASE, P_, E - NUMTEST, E, nInst) for E in end_days])
    old_base = wscore(POS_BASE, P_, 500, 750, nInst); new_base = wscore(POS_BASE, P_, 750, nt, nInst)
    print(f"  v15 baseline (uniform COMBINE_GAIN={V15.COMBINE_GAIN}): OLD={old_base:.1f} NEW={new_base:.1f} "
          f"rmean={curve_base.mean():.1f} rfloor={curve_base.min():.1f}\n")

    def compute_algo(gain_hi, gain_lo):
        state = {"prev_t": -1, "prev_shares": 0}
        algo = np.zeros(nt)
        for k in range(130, nt):
            cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
            algo[k] = np.clip(_algo_vol_shares_regime(logp[0, :k + 1], cur0, dlr[0], gain_hi, gain_lo, state),
                               -lim0, lim0)
        return algo

    def evaluate(gain_hi, gain_lo):
        algo = compute_algo(gain_hi, gain_lo)
        POS = POS_BASE.copy(); POS[0, :] = algo
        curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
        old = wscore(POS, P_, 500, 750, nInst); new = wscore(POS, P_, 750, nt, nInst)
        n_worse = int((curve < curve_base).sum()); n_better = int((curve > curve_base).sum())
        passed = (old > old_base) and (new > new_base) and (curve.mean() > curve_base.mean())
        return dict(hi=gain_hi, lo=gain_lo, old=old, new=new, rmean=curve.mean(), rfloor=curve.min(),
                    n_worse=n_worse, n_better=n_better, passed=passed)

    GAINS = [10.0, 13.0, 16.0, 19.0, 22.0, 25.0]
    print(f"=== dense 2D grid: GAIN_HI x GAIN_LO in {GAINS} ===")
    print(f"{'GAIN_HI':>9}{'GAIN_LO':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    results = []
    for hi in GAINS:
        for lo in GAINS:
            r = evaluate(hi, lo)
            results.append(r)
            tag = "PASS" if r["passed"] else ""
            print(f"{hi:>9.1f}{lo:>9.1f}{r['old']:>9.1f}{r['new']:>9.1f}{r['rmean']:>9.1f}"
                  f"{r['rfloor']:>9.1f}{r['n_worse']:>9}/61{tag:>7}")

    passing = [r for r in results if r["passed"]]
    print(f"\n{len(passing)}/{len(results)} configs pass (OLD+NEW+rmean jointly beat v15).")
    for r in sorted(results, key=lambda r: -r["rmean"])[:8]:
        print(f"  HI={r['hi']:.1f} LO={r['lo']:.1f}  rmean={r['rmean']:.1f}  n_worse={r['n_worse']}/61")
