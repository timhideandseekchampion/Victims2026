"""
test_v26_factor_deadband.py -- v25's significance-gated deadband tested the per-name IC of the FULL,
already-blended `wz` signal and found it catastrophically bad (most names never individually clear a
Bonferroni bar on an aggregate of several already-weak signals, so nearly everything freezes). User's
suggestion: instead gate on a SIMPLE, single, raw factor's own per-name significance -- either the
short-horizon reversion leg (REV_W=10, already blended into champ via BLEND=0.3) or a
Jegadeesh-Titman-style momentum leg (skip-S=20, look back L=120, already used as the "momJT"
fallback signal) -- rather than the noisy composite.

Mechanism: for each name, test whether ONE simple raw factor (reversion or momentum) has a
statistically significant (Bonferroni-corrected across 50 names) trailing-L_test-day causal IC
against that name's own realized return. If significant, trade the fresh sign(wz) as normal. If
NOT significant for that name, hold its previous position instead of introducing a fresh bet.

EFFICIENCY: same precompute-once pattern as v23/v24/v25.

Run: python3 test_v26_factor_deadband.py
"""
import numpy as np, pandas as pd
from scipy import stats
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
    idio_logp = logp_full[1:]              # 50 x nt
    ret_full = idio_logp[:, 1:] - idio_logp[:, :-1]   # 50 x (nt-1)
    n_names = nInst - 1
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

    print("=== precompute v22's wz/killed sequence + baseline positions once ===")
    reset(V22)
    POS22 = np.zeros((nInst, nt))
    WZ_HIST = np.full((n_names, nt), np.nan)
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

    def reversion_hist(w):
        """out[:, idx] = -(idio_logp[:,idx] - idio_logp[:,idx-w]) -- raw REV_W-style reversion
        factor value known as of day idx+1 (0-indexed idx matches WZ_HIST's day-1 convention)."""
        out = np.full((n_names, nt), np.nan)
        if w >= nt: return out
        out[:, w:] = -(idio_logp[:, w:] - idio_logp[:, :-w])
        return out

    def momentum_hist(skip, look):
        """out[:, idx] = idio_logp[:,idx-skip] - idio_logp[:,idx-look] -- Jegadeesh-Titman-style
        raw momentum factor value known as of day idx+1."""
        out = np.full((n_names, nt), np.nan)
        if look >= nt: return out
        out[:, look:] = idio_logp[:, look - skip:nt - skip] - idio_logp[:, :nt - look]
        return out

    def sig_threshold(n_samples, alpha):
        alpha_adj = alpha / n_names
        tcrit = stats.t.ppf(1 - alpha_adj / 2, df=n_samples - 2)
        return float(tcrit / np.sqrt(n_samples - 2 + tcrit ** 2))

    def weak_mask_hist(factor_hist, L_test, alpha, min_day):
        """weak[:, idx] = True where a name's trailing-L_test causal IC of factor_hist vs its
        realized return is NOT Bonferroni-significant (or insufficient history)."""
        weak = np.ones((n_names, nt), dtype=bool)
        thr = sig_threshold(L_test, alpha)
        for t in range(min_day, nt + 1):
            idx = t - 1
            a = idx - L_test
            if a < 0 or idx > ret_full.shape[1]:
                continue
            xs = factor_hist[:, a:idx]; ys = ret_full[:, a:idx]
            ok = np.isfinite(xs).all(axis=1) & np.isfinite(ys).all(axis=1)
            mx = xs.mean(1); my_ = ys.mean(1)
            vx = xs.var(1); vy = ys.var(1)
            cov = ((xs - mx[:, None]) * (ys - my_[:, None])).mean(1)
            denom = np.sqrt(vx * vy)
            ic = np.zeros(n_names)
            valid = ok & (denom > 1e-20)
            ic[valid] = cov[valid] / denom[valid]
            weak[:, idx] = (~valid) | (np.abs(ic) <= thr)
        return weak

    def build_pos(factor_hist, L_test, alpha, min_day):
        weak = weak_mask_hist(factor_hist, L_test, alpha, min_day)
        POS = np.zeros((nInst, nt))
        POS[0, :] = POS22[0, :]
        prev_sign = None; prev_t = -1
        for t in range(V22.WARMUP, nt):
            idx = t - 1
            if KILLED[idx]:
                prev_sign = np.zeros(n_names); prev_t = t
                continue
            wz = WZ_HIST[:, idx]
            sign = np.sign(wz)
            have_prev = (prev_sign is not None) and (prev_t == t - 1)
            if have_prev and t >= min_day:
                keep = weak[:, idx] & (prev_sign != 0)
                sign = np.where(keep, prev_sign, sign)
            cur = P_[1:, idx]; lim = (dlr[1:] / cur).astype(int)
            POS[1:, idx] = np.clip(sign * (dlr[1:] / cur), -lim, lim).astype(int)
            prev_sign = sign; prev_t = t
        return POS

    # sanity check
    dummy = reversion_hist(10)
    POS_check = build_pos(dummy, 120, 0.05, nt + 1)
    max_diff = np.max(np.abs(POS_check - POS22))
    print(f"=== sanity check: min_day > nt (deadband never engages) must reproduce v22 exactly ===\n"
          f"  max|diff|={max_diff:.2e} (should be 0)\n")
    if max_diff > 0:
        print("  *** WARNING: no-op config does not reproduce v22 -- do not trust results below. ***\n")

    print(f"{'factor':>10}{'L_test':>8}{'ALPHA':>7}{'WIN250':>9}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'pass':>7}")
    factor_configs = [
        ("rev10", reversion_hist(10)),
        ("rev20", reversion_hist(20)),
        ("momJT", momentum_hist(20, 120)),
        ("mom60", momentum_hist(0, 60)),
    ]
    for fname, fhist in factor_configs:
        for L_test, alpha in ((120, 0.05), (120, 0.10), (250, 0.05), (250, 0.10)):
            min_day = max(V22.WARMUP + L_test, 400)
            POS = build_pos(fhist, L_test, alpha, min_day)
            curve = np.array([wscore(POS, P_, E - NUMTEST, E, nInst) for E in end_days])
            win250 = wscore(POS, P_, 250, 500, nInst); old = wscore(POS, P_, 500, 750, nInst)
            new = wscore(POS, P_, 750, nt, nInst)
            n_worse = int((curve < curve22).sum()); n_better = int((curve > curve22).sum())
            passed = (win250 >= win250_22) and (old > old22) and (new > new22) and (curve.mean() > curve22.mean())
            tag = "PASS" if passed else ""
            print(f"{fname:>10}{L_test:>8}{alpha:>7.2f}{win250:>9.1f}{old:>9.1f}{new:>9.1f}{curve.mean():>9.1f}"
                  f"{curve.min():>9.1f}{n_worse:>9}/61{tag:>7}   n_better={n_better}/61")
