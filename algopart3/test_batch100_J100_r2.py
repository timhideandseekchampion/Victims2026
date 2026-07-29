"""
test_batch100_J100_daytrend.py

J100: test a within-1000-day position/trend feature (a simple linear day-index term) both as (a) a
DIAGNOSTIC on the existing rolling-score curve -- does score correlate with day-index in a way not
explained by a realized-vol-regime proxy, suggesting genuine secular drift in the data-generating
process -- and (b) as a literal CANDIDATE mechanism: a uniform (same for every idio name) linear-in-
day-index tilt added to v10's final wz, swept over sign and gain.

Note on causality: the day-index feature used for the tradable mechanism, trend_val(t) = (t-500)/500,
uses only fixed constants (no foreknowledge of total sample length nt) and t itself is always known
live (elapsed day count) -- so it is a legitimate causal feature, unlike a feature normalized by nt.
"""
import numpy as np, time
from batch100_common_gi import (
    nInst, nt, nIdio, P_, rs, days, WZ_V10, algo_pos, build_pos_from_wz, evaluate, print_sanity,
    base_wo, base_wn, base_scs, scs_curve, wscore, OLD, NEW,
)

SANITY_OK = print_sanity("(J100 day-trend)")

# ============================================================================
# (a) DIAGNOSTIC: does the rolling score curve correlate with day-index, beyond what a realized-vol
#     regime proxy explains?
# ============================================================================
end_days = list(range(400, nt + 1, 10))
assert len(end_days) == len(base_scs)

VOL_WIN_DIAG = 250
vol_proxy = []
for E in end_days:
    lo = max(0, E - VOL_WIN_DIAG)
    seg = rs[:, lo:E]
    vol_proxy.append(float(np.nanstd(seg.mean(axis=0))))  # vol of the cross-sectional-mean idio return
vol_proxy = np.array(vol_proxy)
day_idx = np.array(end_days, dtype=float)

def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = a.std() * b.std()
    return float((a * b).mean() / denom) if denom > 1e-15 else float('nan')

c_day = corr(base_scs, day_idx)
c_vol = corr(base_scs, vol_proxy)
c_day_vol = corr(day_idx, vol_proxy)

# partial corr of score vs day, controlling for vol_proxy (residualize both on vol_proxy via simple OLS)
def resid_on(y, x):
    xb = x - x.mean(); yb = y - y.mean()
    b = (xb * yb).sum() / (xb * xb).sum()
    return yb - b * xb

score_resid = resid_on(base_scs, vol_proxy)
day_resid = resid_on(day_idx, vol_proxy)
c_partial = corr(score_resid, day_resid)

print(f"\n=== DIAGNOSTIC: rolling score vs day-index vs vol-regime proxy (n={len(end_days)} windows) ===")
print(f"  corr(score, day_index)              = {c_day:+.3f}")
print(f"  corr(score, vol_proxy)               = {c_vol:+.3f}")
print(f"  corr(day_index, vol_proxy)           = {c_day_vol:+.3f}")
print(f"  partial corr(score, day | vol_proxy) = {c_partial:+.3f}")
print(f"  --> {'a nontrivial day-index correlation SURVIVES controlling for vol regime -- suggests genuine secular drift.' if abs(c_partial) > 0.3 and abs(c_partial) > 0.5*abs(c_day) else 'the raw day/score correlation is largely explained by vol regime, OR both are weak -- no strong evidence of secular drift distinct from vol.'}")

# ============================================================================
# (b) CANDIDATE mechanism: uniform linear day-trend tilt added to v10's wz
# ============================================================================
def build_trend(gain):
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wzbase = WZ_V10[:, t]
        trend_val = (t - 500) / 500.0
        tilt = gain * trend_val * (np.abs(wzbase).mean() + 1e-12)
        WZ[:, t] = wzbase + tilt
    return build_pos_from_wz(WZ)


print("\n=== SWEEP: uniform day-trend tilt, gain in {+-0.02, +-0.05, +-0.1, +-0.2} ===")
GAINS = [0.02, 0.05, 0.1, 0.2, -0.02, -0.05, -0.1, -0.2]
results = [evaluate(f"trend gain={g:+.2f}", build_trend(g)) for g in GAINS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for c in passing:
        print(f"  {c['name']:<20} rmean={c['rm']:.1f} n_worse={c['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for c in sorted(results, key=lambda c: -c["rm"])[:6]:
        print(f"  {c['name']:<20} OLD={c['wo']:>7.1f} NEW={c['wn']:>7.1f} rmean={c['rm']:>7.1f} "
              f"rfloor={c['rf']:>7.1f} n_worse={c['nworse']}/61")

best = max(results, key=lambda c: c["rm"])
print(f"\nBest by rolling mean: {best['name']} (rmean={best['rm']:.1f} vs v10 rmean={base_scs.mean():.1f})")
