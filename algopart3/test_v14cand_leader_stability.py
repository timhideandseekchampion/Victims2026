"""
test_v14cand_leader_stability.py

CANDIDATE: only trust the pairwise boost if the SAME leader has persisted recently (require the
identified leader to have also been among follower j's best leader over the trailing STAB_W days;
if not, shrink or discard that stock's boost).

NOTE BEFORE RUNNING: this is the SAME mechanism as `test_h3_leader_stability.py` /
`test_h3_stage2_backtest.py` ("H3 leader-identity-stability gate"), already re-confirmed rejected
earlier this session (README: "Why Bonferroni alone isn't enough... H3, re-confirmed") -- every
`min_stab` threshold from 10-200 days made rmean worse monotonically, and the rolling FLOOR did not
move AT ALL (563.8 unchanged at every threshold), against the OLD SAFE_llboost baseline. This re-runs
the same idea against the CURRENT best (SAFE_llboost_v9) rather than reciting the old numbers, for a
fully current, rigorous answer -- not because there was reason to expect a different mechanism-level
outcome. Reuses v9's `_beta_adjusted_target`, `_algo_vol_shares` verbatim; only the boost gets a
persistence filter layered on top.

MECHANISM: for each follower j, track which of the 39 candidates was selected as its leader each day
(the identity, independent of whether it passed the significance bar). Only trust today's boost if
the SAME leader was ALSO j's identified leader on at least a fraction FRAC_REQ of the trailing STAB_W
days -- otherwise shrink the boost by SHRINK_FACTOR (0 = hard gate, discard entirely; >0 = soft
multiplier, matching H3's own two variants).

Baseline = SAFE_llboost_v9. Must beat it on OLD, NEW, rolling-mean JOINTLY to pass.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v9 as V9

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V9.WARMUP, V9.BOOST_MIN_DAY, V9.BOOST_K
BOOST_N_CANDIDATES, BOOST_IC_L, BOOST_P, BOOST_SCALE_W = (
    V9.BOOST_N_CANDIDATES, V9.BOOST_IC_L, V9.BOOST_P, V9.BOOST_SCALE_W)
RIDGE_A = V9.RIDGE_A
HALF_LIVES = V9.HALF_LIVES


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

print("=== precompute: reversal leg, ALGO leg, ridge WZ (unchanged -- reused verbatim from v9) ===",
      flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V9.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V9._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)

WZ_RIDGE = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V9._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V9._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    WZ_RIDGE[:, t] = (1 - V9.BLEND) * wz + V9.BLEND * REV[:, t]
print(f"  done ({time.time()-t0:.0f}s)", flush=True)

# ==================================================================================================
# precompute boost value AND leader identity per day (verbatim V9._pairwise_boost logic, but also
# recording which candidate index was selected -- needed for the stability check)
# ==================================================================================================
print("=== precompute: boost values + leader identity per day (shared across every stability config) ===",
      flush=True)
t0 = time.time()
n_days = nt - BOOST_MIN_DAY
BOOST_AT = np.zeros((nIdio, nt))
LEADER_ID = np.full((nIdio, nt), -1, dtype=int)   # -1 = no significant leader that day
for t in range(BOOST_MIN_DAY, nt):
    rsl = rs[:, :t]
    n, T = rsl.shape
    Xi_full = rsl[:, :-1]; Yj = rsl[:, 1:]
    thr = V9._sig_threshold(Xi_full.shape[1])
    cand_idx = np.argsort(-np.nanstd(Xi_full, axis=1))[:BOOST_N_CANDIDATES]
    C = V9._corrmat(Xi_full[cand_idx], Yj)
    for j in range(n):
        col = C[:, j].copy()
        cp = np.where(cand_idx == j)[0]
        if len(cp): col[cp[0]] = np.nan
        if np.all(np.isnan(col)): continue
        ci = int(np.nanargmax(np.abs(col)))
        if abs(col[ci]) <= thr: continue
        i = cand_idx[ci]
        lead = rsl[i]
        scale = np.nanstd(lead[max(0, T - 1 - BOOST_SCALE_W):T - 1]) + 1e-12
        lead_boost = np.sign(lead) * (np.abs(lead) / scale) ** BOOST_P
        a = max(0, T - 1 - BOOST_IC_L)
        xs = lead_boost[a:T - 1]; ys = rsl[j, a + 1:T]
        ok = ~np.isnan(xs) & ~np.isnan(ys)
        if ok.sum() < 60 or xs[ok].std() < 1e-12: continue
        ic = float(np.corrcoef(xs[ok], ys[ok])[0, 1])
        if ic <= 0: continue
        BOOST_AT[j, t] = lead_boost[-1]
        LEADER_ID[j, t] = i
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def stability_mask(stab_w, frac_req):
    """mask[j,t] = fraction of the trailing stab_w days (before t) where j's leader identity matched
    today's leader (only meaningful where LEADER_ID[j,t] != -1); returns a bool 'trusted' array."""
    trusted = np.zeros((nIdio, nt), dtype=bool)
    for t in range(BOOST_MIN_DAY, nt):
        lo = max(BOOST_MIN_DAY, t - stab_w)
        if t - lo < max(10, stab_w // 4):
            continue
        today = LEADER_ID[:, t]
        hist = LEADER_ID[:, lo:t]
        match_frac = (hist == today[:, None]).mean(1)
        trusted[:, t] = (today != -1) & (match_frac >= frac_req)
    return trusted


def build_pos(stab_w=None, frac_req=0.0, shrink=0.0):
    POS = np.zeros((nInst, nt))
    if stab_w is not None:
        trusted = stability_mask(stab_w, frac_req)
    for t in days:
        wz = WZ_RIDGE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            b = BOOST_AT[:, t]
            if stab_w is not None:
                b = np.where(trusted[:, t], b, b * shrink)
            wz = wz + BOOST_K * b
        cur = P_[:, t]; lim = (dlr / cur).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: stab_w=None must reproduce SAFE_llboost_v9 exactly ===")
POS_base = build_pos(None)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v9 docstring: 848.8/893.3/894.1/708.6)")
if not (abs(base_wo - 848.8) < 0.5 and abs(base_wn - 893.3) < 0.5):
    print("  *** WARNING: baseline does NOT reproduce v9 -- do not trust results below. ***")
else:
    print("  OK -- matches v9 to within rounding.")


def evaluate(nm, stab_w, frac_req, shrink, verbose=True):
    Pz = build_pos(stab_w, frac_req, shrink); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<34}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, passed=passed, rm=scs.mean(), nworse=nworse, scs=scs)


print("\n=== HARD GATE: require the SAME leader for >=frac_req of the trailing stab_w days, else discard ===")
results = []
for stab_w in (20, 40, 60, 100):
    for frac_req in (0.5, 0.7, 0.9):
        results.append(evaluate(f"stab_w={stab_w} frac_req={frac_req} (hard)", stab_w, frac_req, 0.0))

print("\n=== SOFT MULTIPLIER: shrink (not discard) when the persistence check fails ===")
for stab_w in (20, 40, 60):
    for shrink in (0.3, 0.5, 0.7):
        results.append(evaluate(f"stab_w={stab_w} shrink={shrink} (soft)", stab_w, 0.7, shrink))

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v9 on OLD+NEW+rmean jointly.")
if not passing:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  {c['name']:<40} rmean={c['rm']:>7.1f}  n_worse={c['nworse']}/61")
