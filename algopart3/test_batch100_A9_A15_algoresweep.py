"""
test_batch100_A9_A15_algoresweep.py

Batch-100 ideas A9-A15: resweep several ALGO-leg parameters of SAFE_llboost_v10 (current best)
one at a time, holding everything else at shipped values:
  A9:  IC_EW_HL pair    in {(15,40),(20,45),(25,50),(20,60)}   (shipped (20,45))
  A10: IC_EW_W          in {150,175,200,225,250}               (shipped 200)
  A11: MOM_LB_SHORT     in {5,6,7,8,9}                          (shipped 7)
  A12: MOM_LB_LONG      in {10,11,12,13,14}                     (shipped 12)
  A13: COMBINE_GAIN     in {12,14,16,18,20,25}                  (shipped 16.0)
  A14: DEADBAND_THRESH_FRAC in {0.10,0.15,0.20,0.25,0.30,0.35}  (shipped 0.25)
  A15: DEADBAND_MIN_DAY in {350,400,450,480,520}                (shipped 400)

All seven parameters live ONLY inside SAFE_llboost_v10._algo_vol_shares (the ALGO leg) -- none of
them touch the idio ridge ensemble, BLEND reversion, pairwise boost, or rank-stability blend. So the
expensive idio precompute (WZ_V10, identical to test_v20cand_idio_deadband.py's) is built ONCE and
reused verbatim across all sweeps; only the cheap ALGO leg (_algo_vol_shares, a single-instrument
scan) is recomputed per candidate value, by monkey-patching the module-level constant that
_algo_vol_shares reads at call time (its body references V10.<NAME> as globals, resolved fresh on
every call) and resetting the module's cross-call state (_PREV_ALGO_SHARES, _PREV_T) before each
fresh run, matching how getMyPosition would cold-start.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v10 as V10

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_MIN_DAY, BOOST_K = V10.WARMUP, V10.BOOST_MIN_DAY, V10.BOOST_K
RIDGE_A, HALF_LIVES = V10.RIDGE_A, V10.HALF_LIVES
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = V10.RS_SHORT_W, V10.RS_LONG_W, V10.RS_WEIGHT

# shipped values, to restore after each sweep
SHIP_IC_EW_HL = V10.IC_EW_HL
SHIP_IC_EW_W = V10.IC_EW_W
SHIP_MOM_LB_SHORT = V10.MOM_LB_SHORT
SHIP_MOM_LB_LONG = V10.MOM_LB_LONG
SHIP_COMBINE_GAIN = V10.COMBINE_GAIN
SHIP_DEADBAND_THRESH_FRAC = V10.DEADBAND_THRESH_FRAC
SHIP_DEADBAND_MIN_DAY = V10.DEADBAND_MIN_DAY


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

print("=== precompute: full SAFE_llboost_v10 idio wz (ridge ensemble + BLEND + boost + "
      "rank-stability), verbatim -- unaffected by any ALGO-leg parameter tested here ===", flush=True)
t0 = time.time()
days = list(range(WARMUP, nt))
REV = np.zeros((nIdio, nt))
for t in days:
    rv_ = logp[1:, t] - logp[1:, t - V10.REV_W]
    rv_ = rv_ - rv_.mean()
    REV[:, t] = -rv_ / (rv_.std() + 1e-12)

BOOST = np.zeros((nIdio, nt))
for k in range(BOOST_MIN_DAY, nt):
    BOOST[:, k] = V10._pairwise_boost(rs[:, :k])

WZ_V10 = np.full((nIdio, nt), np.nan)
for t in days:
    rr_ = r[:, :t]
    X = rr_[:, :-1].T
    Y = V10._beta_adjusted_target(rr_)
    xq = rr_[:, -1]
    fs = []
    for hl in HALF_LIVES:
        B, mx, my = V10._ewls_ridge(X, Y, hl, RIDGE_A)
        pred = my + (xq - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    wz = (1 - V10.BLEND) * wz + V10.BLEND * REV[:, t]
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    if t >= max(RS_SHORT_W, RS_LONG_W) + 5:
        short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
        long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
        sz = short_ret - short_ret.mean(); sstd = sz.std()
        lz = long_ret - long_ret.mean(); lstd = lz.std()
        if sstd > 1e-12 and lstd > 1e-12:
            sz = sz / sstd; lz = lz / lstd
            disagree = np.sign(lz) != np.sign(sz)
            rs_sig = np.where(disagree, -sz, 0.0)
            s_std = rs_sig.std()
            s_z = (rs_sig - rs_sig.mean()) / (s_std + 1e-12) if s_std > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
    WZ_V10[:, t] = wz
print(f"  done ({time.time()-t0:.0f}s)", flush=True)


def reset_algo_state():
    V10._PREV_ALGO_SHARES = 0
    V10._PREV_T = -1


def compute_algo_pos():
    reset_algo_state()
    algo_pos = np.zeros(nt)
    for k in range(130, nt):
        cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
        algo_pos[k] = np.clip(V10._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
    return algo_pos


def build_pos(algo_pos):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_V10[:, t]
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n=== sanity check: shipped ALGO params must reproduce SAFE_llboost_v10 exactly ===")
algo_pos_base = compute_algo_pos()
POS_base = build_pos(algo_pos_base)
base_scs = scs_curve(POS_base)
base_wo, base_wn = wscore(POS_base, *OLD), wscore(POS_base, *NEW)
print(f"  baseline: OLD={base_wo:.1f}  NEW={base_wn:.1f}  rmean={base_scs.mean():.1f}  "
      f"rfloor={base_scs.min():.1f}   (v10 docstring: 871.0/912.6/909.8/709.7)")
SANITY_OK = abs(base_wo - 871.0) < 0.5 and abs(base_wn - 912.6) < 0.5
if not SANITY_OK:
    print("  *** WARNING: baseline does NOT reproduce v10 -- do not trust results below. ***")
else:
    print("  OK -- matches v10 to within rounding.")


def evaluate(nm, verbose=True):
    algo_pos = compute_algo_pos()
    Pz = build_pos(algo_pos); scs = scs_curve(Pz)
    wo = wscore(Pz, *OLD); wn = wscore(Pz, *NEW)
    passed = (wo > base_wo) and (wn > base_wn) and (scs.mean() > base_scs.mean())
    nworse = int((scs < base_scs).sum())
    if verbose:
        tag = "  <== PASS" if passed else ""
        print(f"  {nm:<28}OLD={wo:7.1f}  NEW={wn:7.1f}  rmean={scs.mean():7.1f}  rfloor={scs.min():7.1f}  "
              f"n_worse={nworse}/{len(scs)}{tag}")
    return dict(name=nm, wo=wo, wn=wn, rm=scs.mean(), rf=scs.min(), nworse=nworse, passed=passed)


def run_sweep(title, attr, values, fmt=lambda v: str(v)):
    print(f"\n=== A-sweep: {title} (shipped {attr}={getattr(V10, attr)}) ===")
    ship = getattr(V10, attr)
    results = []
    for v in values:
        setattr(V10, attr, v)
        results.append(evaluate(f"{attr}={fmt(v)}"))
    setattr(V10, attr, ship)
    passing = [c for c in results if c["passed"]]
    print(f"  -> {len(passing)}/{len(results)} beat v10 jointly on OLD+NEW+rmean.")
    return results


# --- A9: IC_EW_HL pair (tuple attribute, not a scalar -- handled directly) ---
print(f"\n=== A9: IC_EW_HL pair sweep (shipped {V10.IC_EW_HL}) ===")
A9_results = []
for pair in [(15, 40), (20, 45), (25, 50), (20, 60)]:
    V10.IC_EW_HL = pair
    A9_results.append(evaluate(f"IC_EW_HL={pair}"))
V10.IC_EW_HL = SHIP_IC_EW_HL
A9_passing = [c for c in A9_results if c["passed"]]
print(f"  -> {len(A9_passing)}/{len(A9_results)} beat v10 jointly on OLD+NEW+rmean.")

# --- A10: IC_EW_W ---
A10_results = run_sweep("A10 IC_EW_W", "IC_EW_W", [150, 175, 200, 225, 250])

# --- A11: MOM_LB_SHORT ---
A11_results = run_sweep("A11 MOM_LB_SHORT", "MOM_LB_SHORT", [5, 6, 7, 8, 9])

# --- A12: MOM_LB_LONG ---
A12_results = run_sweep("A12 MOM_LB_LONG", "MOM_LB_LONG", [10, 11, 12, 13, 14])

# --- A13: COMBINE_GAIN ---
A13_results = run_sweep("A13 COMBINE_GAIN", "COMBINE_GAIN", [12.0, 14.0, 16.0, 18.0, 20.0, 25.0])

# --- A14: DEADBAND_THRESH_FRAC ---
A14_results = run_sweep("A14 DEADBAND_THRESH_FRAC", "DEADBAND_THRESH_FRAC",
                         [0.10, 0.15, 0.20, 0.25, 0.30, 0.35])

# --- A15: DEADBAND_MIN_DAY ---
A15_results = run_sweep("A15 DEADBAND_MIN_DAY", "DEADBAND_MIN_DAY", [350, 400, 450, 480, 520])

print("\n\n=== SUMMARY (best by rolling mean per sweep) ===")
for tag, res in [("A9 IC_EW_HL", A9_results), ("A10 IC_EW_W", A10_results),
                 ("A11 MOM_LB_SHORT", A11_results), ("A12 MOM_LB_LONG", A12_results),
                 ("A13 COMBINE_GAIN", A13_results), ("A14 DEADBAND_THRESH_FRAC", A14_results),
                 ("A15 DEADBAND_MIN_DAY", A15_results)]:
    best = max(res, key=lambda c: c["rm"])
    npass = sum(c["passed"] for c in res)
    print(f"  {tag:<26} best={best['name']:<28} rmean={best['rm']:.1f} (base={base_scs.mean():.1f})  "
          f"npass={npass}/{len(res)}")
