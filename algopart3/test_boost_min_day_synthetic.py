"""
test_boost_min_day_synthetic.py -- cross-validation check: does the "boost needs enough history
before it's net-positive, not just active" finding from real prices.txt (BOOST_MIN_DAY sweep in the
prior conversation turn) reproduce on an INDEPENDENTLY-calibrated synthetic dataset with a KNOWN,
controlled lead-lag structure (changepoint_synthetic.py's rho=0.25, 20-pair structure) -- not fit to
prices.txt at all? If the same qualitative pattern shows up here too, that's real cross-dataset
evidence the phenomenon is a genuine statistical fact (small-sample correlation estimates are
noisier) rather than an artifact of this one real dataset's specific noise realization.

METHODOLOGY, corrected from a mechanistic re-check: the earlier real-data sweep's "high BOOST_MIN_DAY
hurts" finding on the OLD/NEW headline windows was a trivial artifact of pushing the activation
threshold INTO the graded evaluation windows (fewer active days = lower score, unrelated to
estimate quality) -- OLD/NEW were EXACTLY unchanged for every BOOST_MIN_DAY from 150 to 480, only
the 250-500 window and rolling mean/floor (which include early windows) moved. The genuine
statistical effect is ONLY on the low side. This script isolates that correctly: evaluates an
EARLY window (comparable to real data's 250-500 test) and a LATE window that stays safely after
every tested BOOST_MIN_DAY value (so if the "low BOOST_MIN_DAY hurts" pattern is genuine, it should
show up ONLY in the early window, with the late window unaffected -- exactly mirroring what real
data showed).

Uses ONLY the pre-change portion of the synthetic generator (no regime change involved) -- a stable,
independently-calibrated lead-lag structure, not real market data.

Run: python3 test_boost_min_day_synthetic.py
"""
import numpy as np
import algopart3.ArbitrageVictimsV2 as AV
from changepoint_synthetic import simulate

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


def window_score(POS, out, S, E):
    nInst = out.shape[0]
    curPos = np.zeros(nInst); comm_vec = np.zeros(nInst); prevCur = None; tot = []
    for tt in range(S, E + 1):
        cur = out[:, tt - 1]
        newPos = (POS[:, tt - 1].copy() if tt < E else curPos.copy())
        if tt > S:
            tot.append(float((curPos * (cur - prevCur) - comm_vec).sum()))
        dP = newPos - curPos
        comm_vec = commRate * np.abs(dP) * cur
        prevCur = cur; curPos = newPos
    tot = np.array(tot)
    return score(tot.mean(), tot.std())


def pos_matrix(mod, out, nt_use):
    nInst = out.shape[0]
    POS = np.zeros((nInst, nt_use))
    for k in range(130, nt_use):
        cur = out[:, k]; lim = (dlr / cur).astype(int)
        POS[:, k] = np.clip(np.asarray(mod.getMyPosition(out[:, :k + 1])), -lim, lim).astype(int)
    return POS


if __name__ == "__main__":
    # pre-change-only: NT_PRE=1000 stable rho=0.25 structure, tiny NT_POST just to satisfy the
    # generator's signature -- we only ever use columns [:1000], entirely pre-change.
    out, idio, algo_ret, W_new, leaders_new = simulate(1000, 10, "reverse", seed=123)
    nt_use = 1000

    print("=== synthetic (known rho=0.25 lead-lag, pre-change only): BOOST_MIN_DAY sweep ===")
    print(f"{'BOOST_MIN_DAY':>14}{'early(250-500)':>16}{'late(800-1000)':>16}")
    for bmd in (150, 200, 250, 300, 400, 480, 550, 650):
        AV.BOOST_MIN_DAY = bmd
        reset(AV)
        POS = pos_matrix(AV, out, nt_use)
        early = window_score(POS, out, 250, 500)
        late = window_score(POS, out, 800, 1000)
        print(f"{bmd:>14}{early:>16.1f}{late:>16.1f}")
