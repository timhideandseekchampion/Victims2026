"""
test_v20_resid_diversifier.py -- sweeps RESID_WEIGHT (the blend weight for the new 'resid'
diversifier signal) against the real-data bar used throughout this repo (OLD/NEW/rmean/rfloor,
n_worse/61 vs v15), rather than trusting an arbitrary first-guess value.

Run: python3 test_v20_resid_diversifier.py
"""
import numpy as np, pandas as pd
import SAFE_llboost_v15 as V15
import SAFE_llboost_v20 as V20

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


def build(mod, P_, nInst, nt):
    reset(mod)
    POS = np.zeros((nInst, nt))
    for t in range(1, nt):
        prcSoFar = P_[:, :t]
        p = np.asarray(mod.getMyPosition(prcSoFar))
        lim = (dlr / prcSoFar[:, -1]).astype(int)
        POS[:, t - 1] = np.clip(p, -lim, lim).astype(int)
    return POS


if __name__ == "__main__":
    P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
    nInst, nt = P_.shape
    end_days = list(range(400, nt + 1, 10))
    NUMTEST = 250

    POS15 = build(V15, P_, nInst, nt)
    curve15 = np.array([wscore(POS15, P_, E - NUMTEST, E, nInst) for E in end_days])
    old15 = wscore(POS15, P_, 500, 750, nInst); new15 = wscore(POS15, P_, 750, nt, nInst)
    print(f"v15 baseline: OLD={old15:.1f} NEW={new15:.1f} rmean={curve15.mean():.1f} rfloor={curve15.min():.1f}\n")

    print(f"{'RESID_WEIGHT':>13}{'OLD':>9}{'NEW':>9}{'rmean':>9}{'rfloor':>9}{'n_worse':>9}{'n_better':>10}")
    for w in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        V20.RESID_WEIGHT = w
        POS20 = build(V20, P_, nInst, nt)
        curve20 = np.array([wscore(POS20, P_, E - NUMTEST, E, nInst) for E in end_days])
        old20 = wscore(POS20, P_, 500, 750, nInst); new20 = wscore(POS20, P_, 750, nt, nInst)
        n_worse = int((curve20 < curve15).sum()); n_better = int((curve20 > curve15).sum())
        passed = (old20 > old15) and (new20 > new15) and (curve20.mean() > curve15.mean())
        tag = "  <== PASS" if passed else ""
        print(f"{w:>13.3f}{old20:>9.1f}{new20:>9.1f}{curve20.mean():>9.1f}{curve20.min():>9.1f}"
              f"{n_worse:>9}{n_better:>10}{tag}")
