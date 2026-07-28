"""
test_v7_boost_min_day_200.py

Direct question: what happens if BOOST_MIN_DAY is lowered from 480 to 200 in the CURRENT
SAFE_llboost_v7 book (N=39 vol-restricted candidates, BOOST_IC_L=250, BOOST_SCALE_W=1000, K=1.5,
COMBINE_GAIN=16)?

The ORIGINAL SAFE_llboost.py docstring already answered this question once, but against a now-stale
book (N=49 candidates -- no vol-restriction existed yet --, BOOST_IC_L=220, BOOST_SCALE_W=500, no
COMBINE_GAIN retune): lowering the gate lifted OLD/NEW/rolling-mean at every boost strength but
monotonically WORSENED the rolling floor (563.8 -> 531.9 -> 522.3 -> 506.0 -> 497.7 -> 479.9 as K
rose 0->3.0 with no min-day gate at all), traced to Bonferroni controlling false-discovery WITHIN
one re-estimate but not ACROSS the ~15 sequential re-estimates made over the file -- a "significant"
leader found on a thin sample can be a lucky false positive that then trades with real size for many
subsequent days.

Two things have changed since that diagnosis that could plausibly change the answer:
  - v3's candidate pool restriction (39 highest-vol names instead of all 49) shrinks the Bonferroni
    correction itself (39 tests instead of 49), which changes the threshold at every sample size.
  - v7's BOOST_IC_L=250 (vs the original 220) and BOOST_SCALE_W=1000 (vs 500) change how each pair's
    own IC-positivity check and boost scaling behave.
Rather than assume the old mechanism still applies unchanged, this re-runs the exact same class of
diagnostic against the true v7 book: does the floor still degrade, is it still concentrated in the
same failure mode (early false-positive leaders persisting), and by how much in absolute score.

_pairwise_boost's BOOST_MIN_DAY gate is a bare module-global lookup, so it can be monkey-patched at
call time on the real V7 module -- no reimplementation, so this can never silently diverge from the
shipped code's actual behavior.
"""
import numpy as np, pandas as pd, time
import SAFE_llboost_v7 as V7

P_ = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T.astype(float)
nInst, nt = P_.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0
NUMTEST = 250
logp = np.log(P_)
r = np.diff(logp, axis=1)
rs = r[1:]
nIdio = rs.shape[0]
WARMUP, BOOST_K = V7.WARMUP, V7.BOOST_K
SHIPPED_MIN_DAY = V7.BOOST_MIN_DAY  # 480


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

# ==================================================================================================
print("=== precompute: v7 WZ (ridge+blend, unaffected by BOOST_MIN_DAY) + ALGO leg ===", flush=True)
t0 = time.time()
WZ = np.full((nIdio, nt), np.nan)
for t in range(WARMUP, nt):
    rr = r[:, :t]
    fs = []
    for hl in V7.HALF_LIVES:
        B, mx, my = V7._ewls_ridge(rr[:, :-1].T, rr[1:, 1:].T, hl, V7.RIDGE_A)
        pred = my + (rr[:, -1] - mx) @ B
        fi = pred - pred.mean()
        fs.append(fi / (fi.std() + 1e-12))
    wz = np.mean(fs, 0)
    rv_ = logp[1:, t] - logp[1:, t - V7.REV_W]
    rv_ = rv_ - rv_.mean()
    WZ[:, t] = (1 - V7.BLEND) * wz + V7.BLEND * (-rv_ / (rv_.std() + 1e-12))

algo_pos = np.zeros(nt)
for k in range(130, nt):
    cur0 = P_[0, k]; lim0 = int(dlr[0] / cur0)
    algo_pos[k] = np.clip(V7._algo_vol_shares(logp[0, :k + 1], cur0, dlr[0]), -lim0, lim0)
print(f"  done ({time.time()-t0:.0f}s)")


def boost_series(min_day):
    """V7._pairwise_boost, verbatim, called fresh at every day from min_day on -- exactly what
    getMyPosition does. Monkey-patches the module global so the real function's own gate fires."""
    V7.BOOST_MIN_DAY = min_day
    B = np.zeros((nIdio, nt))
    for k in range(min_day, nt):
        B[:, k] = V7._pairwise_boost(rs[:, :k])
    V7.BOOST_MIN_DAY = SHIPPED_MIN_DAY
    return B


def build(min_day):
    B = boost_series(min_day)
    POS = np.zeros((nInst, nt))
    for k in range(WARMUP, nt):
        cur = P_[:, k]; lim = (dlr / cur).astype(int)
        wz = WZ[:, k] + (BOOST_K * B[:, k] if k >= min_day else 0.0)
        POS[1:, k] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim[1:], lim[1:])
    POS[0, :] = algo_pos
    return POS, B


print("\n=== sanity: min_day=480 (shipped v7) vs README (830.3/888.5/876.8/674.4) ===")
POS_ship, B_ship = build(SHIPPED_MIN_DAY)
base_scs = scs_curve(POS_ship)
print(f"  min_day=480: OLD={wscore(POS_ship,*OLD):.1f}  NEW={wscore(POS_ship,*NEW):.1f}  "
      f"rmean={base_scs.mean():.1f}  rfloor={base_scs.min():.1f}")

print("\n=== sweep: BOOST_MIN_DAY ===")
print(f"{'min_day':>8} {'OLD':>8} {'NEW':>8} {'rmean':>8} {'rfloor':>8} {'n_worse':>9}")
rows = {}
for md in (150, 200, 250, 300, 350, 400, 450, 480, 500):
    t0 = time.time()
    POS, B = build(md)
    scs = scs_curve(POS)
    nworse = int((scs < base_scs).sum())
    print(f"{md:>8} {wscore(POS,*OLD):>8.1f} {wscore(POS,*NEW):>8.1f} {scs.mean():>8.1f} "
          f"{scs.min():>8.1f} {nworse:>9}/{len(scs)}   ({time.time()-t0:.0f}s)")
    rows[md] = (POS, B, scs)

# ==================================================================================================
print("\n" + "=" * 96)
print("DIAGNOSIS at min_day=200: where does the floor damage come from?")
print("=" * 96)
POS200, B200, scs200 = rows[200]
diff = scs200 - base_scs
worst = np.argsort(diff)[:10]
print("worst windows vs shipped (min_day=480), end_day / base / md200 / diff:")
for i in worst:
    print(f"  end_day={end_days[i]:4d}  base={base_scs[i]:7.1f}  md200={scs200[i]:7.1f}  diff={diff[i]:+7.1f}")
print(f"worse on {int((diff<0).sum())}/{len(diff)} windows overall; mean diff {diff.mean():+.1f}, "
      f"min diff {diff.min():+.1f}")

# significance threshold at each sample size -- the actual mechanism
print("\nsignificance bar shrinks as more (thinner) history becomes eligible:")
for n in (198, 250, 300, 400, 478):
    thr = V7._sig_threshold(n)
    print(f"  n_samples={n:4d}  Bonferroni thr (N={V7.BOOST_N_CANDIDATES}) = {thr:.4f}")

# persistence: of the boosts active in days [200,480) that would NOT exist under the shipped gate,
# how long does a given (follower, leader-at-formation) relationship keep firing, and does its
# realized sign-agreement hold up out of the formation window?
print("\nextra activity unlocked in days [200, 480) (inactive under shipped v7):")
extra_active = (B200[:, 200:480] != 0.0)
print(f"  {int(extra_active.sum())} stock-days boosted in [200,480) that the shipped gate skips "
      f"entirely (0 there today)")
n_names_ever = int((extra_active.any(1)).sum())
print(f"  {n_names_ever}/{nIdio} names get boosted at least once in that window")

# does the early activity's realized sign-agreement (the same ic<=0 continue check inside
# _pairwise_boost, but measured OUT-OF-SAMPLE over the *next* 60 days after formation) hold up?
print("\nout-of-sample check: for each stock-day boosted in [200,480), does boost*next-return>0 "
      "over the FOLLOWING 60 days (a look-AHEAD check done here only for diagnosis, never inside "
      "the traded signal)?")
hits, tot = 0, 0
for k in range(200, 420):
    active = np.where(B200[:, k] != 0.0)[0]
    if len(active) == 0: continue
    fut = rs[active, k:k + 60]
    if fut.shape[1] < 60: continue
    sgn = np.sign(B200[active, k])[:, None]
    hits += int((np.sign(fut) == sgn).sum())
    tot += fut.size
print(f"  next-60-day same-sign hit rate: {100*hits/tot:.1f}% (50% = coin flip) over {tot} obs")

print("\n=== ranking vs shipped v7 (must beat OLD, NEW, rmean jointly) ===")
for md in (150, 200, 250, 300, 350, 400, 450, 500):
    POS, B, scs = rows[md]
    passed = (wscore(POS, *OLD) > wscore(POS_ship, *OLD) and
              wscore(POS, *NEW) > wscore(POS_ship, *NEW) and
              scs.mean() > base_scs.mean())
    print(f"  min_day={md:<5} {'PASS' if passed else 'fail'}")
