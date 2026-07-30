"""
test_v30b_rotw_dense.py -- denser ROT_W sweep around the surprising ROT_W=40 result found in
test_v30_insurance_resweep.py (NEW+10.1, rmean+1.0, n_worse=0/61, 5 genuine switches -- vs ROT_W=50
which was WORSE, n_worse=6/61). Need to know if 40 is a genuine local optimum (smooth, both-sides
worse-or-flat) or an isolated lucky spike before trusting it, per this repo's own established bar.

Run: python3 test_v30b_rotw_dense.py
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
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

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
          f"rmean={curve22.mean():.1f}  rfloor={curve22.min():.1f}\n")

    print("=== precompute champion + all 3 fallback signals for every day ===")
    NAMES = ("champ",) + V22.FALLBACKS
    SIG_HIST = {name: np.full((nInst - 1, nt), np.nan) for name in NAMES}
    RET_HIST = np.full((nInst - 1, nt), np.nan)
    reset(V22)
    for t in range(V22.WARMUP, nt):
        prcSoFar = P_[:, :t]
        idx = t - 1
        SIG_HIST["champ"][:, idx] = V22._idio_signal(prcSoFar)
        fb = V22._fallback_signals(prcSoFar)
        for name in V22.FALLBACKS:
            SIG_HIST[name][:, idx] = fb[name] if fb[name] is not None else SIG_HIST["champ"][:, idx]
        if t < nt:
            R = np.log(P_[1:, t]) - np.log(P_[1:, t - 1])
            RET_HIST[:, idx] = R - R.mean()
    print("  done.\n")

    def xc1_hist():
        xc = np.full(nt, np.nan)
        for idx in range(1, nt):
            a = RET_HIST[:, idx - 1]; b = RET_HIST[:, idx]
            if np.isfinite(a).all() and np.isfinite(b).all():
                d = np.sqrt((a @ a) * (b @ b))
                xc[idx] = float(a @ b / d) if d > 1e-18 else 0.0
        return xc

    XC_HIST = xc1_hist()

    def xsac_hist(xsac_w):
        xs = np.full(nt, np.nan)
        for idx in range(nt):
            lo = idx - xsac_w + 1
            if lo < 0:
                continue
            vals = XC_HIST[lo:idx + 1]
            vals = vals[np.isfinite(vals)]
            if len(vals) >= xsac_w // 2:
                xs[idx] = vals.mean()
        return xs

    def pn1_hist(name):
        s = SIG_HIST[name]
        pn = np.full(nt, np.nan)
        finite = np.isfinite(s).all(axis=0) & np.isfinite(RET_HIST).all(axis=0)
        pn[finite] = (np.sign(s[:, finite]) * RET_HIST[:, finite]).sum(axis=0)
        return pn

    PN_HIST = {name: pn1_hist(name) for name in NAMES}

    def build_pos(xsac_th, xsac_p, xsac_w, rot_p, rot_w, kill_margin, kill_p):
        xs_full = xsac_hist(xsac_w)
        POS = np.zeros((nInst, nt))
        POS[0, :] = POS22[0, :]
        n_kill_days = 0; n_switch_days = 0
        for t in range(V22.WARMUP, nt):
            idx = t - 1
            ready = t >= V22.WARMUP + rot_w + max(rot_p, kill_p)
            if not ready:
                chosen = "champ"
            else:
                picks = []
                for a in range(t - rot_p, t):
                    lo = a - rot_w + 1
                    if lo < V22.WARMUP:
                        picks.append("champ"); continue
                    pn_c = np.nansum(PN_HIST["champ"][lo:a + 1])
                    xsv = xs_full[a] if a - xsac_p + 1 >= 0 and np.all(np.isfinite(xs_full[max(0, a - xsac_p + 1):a + 1])) and np.all(xs_full[max(0, a - xsac_p + 1):a + 1] > xsac_th) else None
                    champ_sick = (pn_c < kill_margin) or (xsv is not None)
                    if not champ_sick:
                        picks.append("champ"); continue
                    best = None; best_v = -1e18
                    for name in V22.FALLBACKS:
                        pf = np.nansum(PN_HIST[name][lo:a + 1])
                        pf_minus_c = np.nansum(PN_HIST[name][lo:a + 1] - PN_HIST["champ"][lo:a + 1])
                        if pf_minus_c > 0.0 and pf > 0.0 and pf > best_v:
                            best_v = pf; best = name
                    picks.append(best if best is not None else "champ")
                if picks and picks[0] != "champ" and all(p == picks[0] for p in picks):
                    chosen = picks[0]
                else:
                    chosen = "champ"
            if chosen != "champ":
                n_switch_days += 1

            killed = False
            if ready and V22.KILL_ON:
                killed = True
                for a in range(t - kill_p, t):
                    lo = a - rot_w + 1
                    if lo < V22.WARMUP:
                        killed = False; break
                    pn = np.nansum(PN_HIST[chosen][lo:a + 1])
                    if not (pn < kill_margin):
                        killed = False; break
            if killed:
                n_kill_days += 1

            if not killed:
                wz = SIG_HIST[chosen][:, idx]
                cur = P_[1:, idx]; lim = (dlr[1:] / cur).astype(int)
                POS[1:, idx] = np.clip(np.sign(wz) * (dlr[1:] / cur), -lim, lim).astype(int)
        return POS, n_switch_days, n_kill_days

    print(f"{'ROT_W':>7}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'n_worse':>9}{'switch_days':>13}{'kill_days':>11}{'pass':>7}")
    for rw in (25, 30, 32, 35, 38, 40, 42, 45, 48, 50, 55):
        POS, n_sw, n_kd = build_pos(V22.XSAC_TH, V22.XSAC_P, V22.XSAC_W, V22.ROT_P, rw, V22.KILL_MARGIN, V22.KILL_P)
        curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
        win250 = wscore(POS, P_, 250, 500, nInst); old = wscore(POS, P_, 500, 750, nInst)
        new = wscore(POS, P_, 750, nt, nInst)
        n_worse = int((curve < curve22).sum())
        passed = (win250 >= win250_22) and (old > old22) and (new > new22) and (curve.mean() > curve22.mean())
        tag = "PASS" if passed else ""
        print(f"{rw:>7}{win250:>9.1f}{old:>9.1f}{new:>9.1f}{curve.mean():>9.1f}{n_worse:>9}/61{n_sw:>13}{n_kd:>11}{tag:>7}")
