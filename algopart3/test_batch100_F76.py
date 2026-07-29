"""
test_batch100_F76.py

F76: Test same-day cross-sectional RANK of ALGO's own move (not just its average/beta) as an extra
idio-book conditioning feature, distinct from the beta-adjusted demeaning already shipped.

INTERPRETATION (stated explicitly, since the idea text underspecifies the exact mechanism): the
shipped beta-adjusted demeaning (_beta_adjusted_target) conditions the idio ridge target on each
name's beta to the cross-sectional MEAN of the OTHER idio names (cf = rs.mean(0)) -- it never
references ALGO's return itself. This idea is about ALGO's OWN move specifically: on each day, rank
ALGO's realized same-day return among that day's full cross-section of 51 instrument returns (where
does ALGO sit in the day's return distribution -- a top-mover day, a bottom-mover day, or unremarkable
-- expressed as a rank in [-1,1]). Since position sizing in this pipeline is sign(wz)*fixed-dollar (a
uniform positive rescaling of wz is a no-op on positions), the only way this feature can matter is by
ADDING a signed tilt that can flip marginal (near-zero) names' signs -- so it is blended in as a
uniform book-wide additive tilt, same style as the shipped rank-stability blend: on ALGO "big up day"
rank, tilt the whole idio book slightly; on "big down day" rank, tilt the other way.
Sweep the blend GAIN (one free parameter) over {0.01, 0.02, 0.05, 0.10, 0.20}.
"""
import numpy as np, pandas as pd
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


def wscore(POS, S, E):
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


end_days = list(range(400, nt + 1, 10))
OLD = (500, 750); NEW = (750, nt)
scs_curve = lambda POS: np.array([wscore(POS, E - NUMTEST, E) for E in end_days])

CACHE = np.load("batch100_cache.npz")
algo_pos = CACHE["algo_pos"]; WZ_V10 = CACHE["WZ_V10"]
days = CACHE["days"].tolist()


def build_pos_from_wz(WZfull):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZfull[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("=== sanity check: reproduce SAFE_llboost_v10 exactly from cache ===")
POS_base = build_pos_from_wz(WZ_V10)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} "
      f"rfloor={base_scs.min():.1f}  (v10 docstring: 871.0/912.6/909.8/709.7)")
sanity_ok = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
print("  OK -- matches v10 to within rounding." if sanity_ok else "  *** WARNING: mismatch ***")

print("\n=== F76: same-day cross-sectional rank of ALGO's own move, as a uniform book-wide additive "
      "tilt ===")

RANK = np.full(nt, np.nan)
for t in days:
    if t - 1 < 0 or t - 1 >= r.shape[1]:
        continue
    day_rets = r[:, t - 1]  # same-day (through day t) realized move for all 51 instruments -- causal
    order = np.argsort(day_rets)
    rank0 = int(np.where(order == 0)[0][0])
    RANK[t] = 2.0 * (rank0 / (nInst - 1) - 0.5)  # in [-1, 1]


def build_wz_algorank(gain):
    WZ = np.zeros((nIdio if False else WZ_V10.shape[0], nt))
    for t in days:
        wz = WZ_V10[:, t].copy()
        rk = RANK[t]
        if not np.isnan(rk):
            wz = wz + gain * rk * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return WZ


results = []
for gain in [0.01, 0.02, 0.05, 0.10, 0.20]:
    WZf = build_wz_algorank(gain)
    POS = build_pos_from_wz(WZf)
    scs = scs_curve(POS)
    wo, wn = wscore(POS, *OLD), wscore(POS, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    results.append(dict(gain=gain, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed))
    tag = "  <== PASS" if passed else ""
    print(f"  gain={gain:<6}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  "
          f"rfloor={scs.min():7.1f}  n_worse={nworse}/{len(scs)}{tag}")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} gain values beat v10 on OLD+NEW+rmean jointly.")
