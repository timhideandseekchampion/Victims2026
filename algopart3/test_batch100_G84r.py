"""
test_batch100_G84r.py

QUESTION (G84): the shipped rank-stability blend uses ONE global RS_WEIGHT=0.015 applied uniformly to
all 50 idio names, every day. Does a LEARNED per-name weight -- higher where the RS signal has
recently, causally, actually been predictive for that specific name, lower/zero where it hasn't --
beat the uniform weight?

MECHANISM: for each name i and day t (t >= BOOST_MIN_DAY, so it turns on together with the boost --
before that, exactly the shipped uniform RS_WEIGHT), compute a trailing causal IC of the raw
(pre-standardization, already-gated) RS signal against the same-index realized idio return, over a
BOOST_IC_L=250-day window (same window length + same same-index feat/ret pairing convention already
used elsewhere in v10, e.g. V10._pairwise_boost / _algo_vol_shares._ic):
  ic_i(t) = corr( RS_RAW[i, t-L:t], rs[i, t-L:t] )
Map IC to a per-name multiplier on the shipped weight: mult_i(t) = clip(1 + GAIN*ic_i(t), 0, CAP)
  w_i(t) = RS_WEIGHT * mult_i(t)
GAIN=0 is a no-op (mult=1 for every name/day) -- reproduces the shipped uniform weight EXACTLY, so
this doubles as an idea-specific check on top of the shared baseline sanity check.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

RS_RAW, rs, days, nIdio, nt = B.RS_RAW, B.rs, B.days, B.nIdio, B.nt
WZ_PRE, BOOST, BOOST_MIN_DAY, BOOST_K, RS_WEIGHT, BOOST_IC_L = (
    B.WZ_PRE, B.BOOST, B.BOOST_MIN_DAY, B.BOOST_K, B.RS_WEIGHT, B.BOOST_IC_L)

L = BOOST_IC_L  # 250, reuse boost's own IC-window convention

IC = np.zeros((nIdio, nt))
for t in range(BOOST_MIN_DAY, nt):
    a = max(0, t - L)
    xs = RS_RAW[:, a:t]; ys = rs[:, a:t]
    mx = xs.mean(1); my = ys.mean(1)
    xc = xs - mx[:, None]; yc = ys - my[:, None]
    vx = (xc * xc).mean(1); vy = (yc * yc).mean(1)
    cov = (xc * yc).mean(1)
    ok = (vx > 1e-16) & (vy > 1e-16)
    ic = np.zeros(nIdio)
    ic[ok] = cov[ok] / np.sqrt(vx[ok] * vy[ok])
    IC[:, t] = ic


def build_pos_learned(gain, cap=3.0):
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        s = RS_RAW[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            mult = np.clip(1.0 + gain * IC[:, t], 0.0, cap)
            w = RS_WEIGHT * mult
            wz = (1 - w) * wz + w * s_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return B.build_pos_from_wz(WZ)


print("\n=== idea-specific check: GAIN=0 must reproduce shipped v10 exactly (mult=1 for all names) ===")
POS_g0 = build_pos_learned(0.0)
r0 = B.evaluate("GAIN=0 (uniform, no-op)", POS_g0)
match = abs(r0["wo"] - B.base_wo) < 1e-6 and abs(r0["wn"] - B.base_wn) < 1e-6
print("  OK -- exact match to baseline." if match else "  *** WARNING: GAIN=0 does NOT match baseline exactly ***")

print("\n=== G84: learned per-name RS weight (IC-gain multiplier), GAIN sweep ===")
results = [r0]
for gain in [5, 10, 20]:
    POS = build_pos_learned(gain)
    results.append(B.evaluate(f"GAIN={gain}", POS))

# only the genuinely non-trivial (gain>0) configs count toward the pass tally
cand_results = results[1:]
passing = [r for r in cand_results if r["passed"]]
print(f"\n{len(passing)}/{len(cand_results)} non-trivial configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for r in passing:
        print(f"  {r['name']:<28} rmean={r['rm']:.1f} n_worse={r['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for r in sorted(cand_results, key=lambda r: -r["rm"]):
        print(f"  {r['name']:<28} OLD={r['wo']:>7.1f} NEW={r['wn']:>7.1f} rmean={r['rm']:>7.1f} "
              f"rfloor={r['rf']:>7.1f} n_worse={r['nworse']}/61")
