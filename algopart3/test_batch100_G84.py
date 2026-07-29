"""
test_batch100_G84.py

QUESTION (G84): the shipped rank-stability blend uses ONE global RS_WEIGHT=0.015 applied uniformly
to all 50 idio names, every day. Does a LEARNED per-name weight -- higher where the RS signal has
recently, causally, actually been predictive for that specific name, lower/zero where it hasn't --
beat the uniform weight?

MECHANISM: for each name i and day t (t >= BOOST_MIN_DAY, so it turns on together with the boost --
before that, fall back to the shipped uniform RS_WEIGHT exactly), compute a trailing causal IC of the
raw (pre-standardization) RS signal against the same-day-indexed realized return, over a fixed
BOOST_IC_L=250-day window (same window length and same same-index feat/ret pairing convention already
used by V10._algo_vol_shares._ic -- feat[a:t] vs ret[a:t] where ret[s] is the return realized starting
at day s):
  ic_i(t) = corr( RS_RAW[i, t-L:t], rs[i, t-L:t] )
Map IC to a per-name multiplier on the shipped weight: mult_i(t) = clip(1 + GAIN*ic_i(t), 0, CAP), so
a name with a strongly positive trailing IC gets an amplified weight (up to CAP*RS_WEIGHT), a name
with a negative trailing IC gets its weight shrunk toward/at 0 (RS blend switched off for that name
that day) -- a "learned", per-name-per-day gate, not a static per-name constant.
  w_i(t) = RS_WEIGHT * mult_i(t)
Everything else (the standardization of the raw signal, the day-level scale factor, the sequencing
after the boost) is identical to the shipped blend.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

WZ_PRE, BOOST, BOOST_K, BOOST_MIN_DAY = B.WZ_PRE, B.BOOST, B.BOOST_K, B.BOOST_MIN_DAY
RS_RAW, RS_WEIGHT = B.RS_RAW, B.RS_WEIGHT
rs, days, nIdio, nt = B.rs, B.days, B.nIdio, B.nt
BOOST_IC_L = B.BOOST_IC_L  # reuse the same 250-day window the boost's own IC test uses

WZ_PREBOOST = np.zeros((nIdio, nt))
for t in days:
    wz = WZ_PRE[:, t].copy()
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    WZ_PREBOOST[:, t] = wz

L = BOOST_IC_L
MIN_DAY = max(BOOST_MIN_DAY, 96 + L)  # need L days of finite RS_RAW history (RS_RAW finite from t=96)


def trailing_ic_matrix():
    """ic_i(t) for t in [MIN_DAY, nt): corr(RS_RAW[i, t-L:t], rs[i, t-L:t]), vectorized across names."""
    IC = np.zeros((nIdio, nt))
    for t in range(MIN_DAY, nt):
        a = t - L
        xs = RS_RAW[:, a:t]          # (nIdio, L)
        ys = rs[:, a:t]               # (nIdio, L), rs col index a..t-1, valid since t <= nt-1 = 999
        mx = xs.mean(1); my = ys.mean(1)
        vx = xs.var(1); vy = ys.var(1)
        cov = ((xs - mx[:, None]) * (ys - my[:, None])).mean(1)
        denom = np.sqrt(vx * vy)
        ok = denom > 1e-20
        ic = np.zeros(nIdio)
        ic[ok] = cov[ok] / denom[ok]
        IC[:, t] = ic
    return IC


IC_MAT = trailing_ic_matrix()


def build_pos_learned(gain, cap):
    WZ = WZ_PREBOOST.copy()
    for t in days:
        wz = WZ_PREBOOST[:, t]
        s = RS_RAW[:, t]
        if not np.isfinite(s).all():
            WZ[:, t] = wz
            continue
        sstd = s.std()
        s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        day_scale = np.abs(wz).mean() + 1e-12
        if t >= MIN_DAY:
            mult = np.clip(1.0 + gain * IC_MAT[:, t], 0.0, cap)
            w = RS_WEIGHT * mult
        else:
            w = np.full(nIdio, RS_WEIGHT)
        WZ[:, t] = (1 - w) * wz + w * s_z * day_scale
    return B.build_pos_from_wz(WZ)


print("\n=== sanity check (idea-specific): gain=0 (mult==1 everywhere) must reproduce v10 exactly ===")
POS0 = build_pos_learned(0.0, 2.0)
r0 = B.evaluate("gain=0 (uniform weight)", POS0)
idea_sanity_ok = sanity_ok and abs(r0["wo"] - B.base_wo) < 0.5 and abs(r0["wn"] - B.base_wn) < 0.5
print("  OK." if idea_sanity_ok else "  *** WARNING: gain=0 does not reproduce v10 -- check logic. ***")

print(f"\n  (context) trailing per-name IC of RS_RAW, day>={MIN_DAY}: mean={IC_MAT[:, MIN_DAY:].mean():.3f} "
      f"std={IC_MAT[:, MIN_DAY:].std():.3f} frac>0={float((IC_MAT[:, MIN_DAY:] > 0).mean()):.3f}")

print("\n=== SWEEP: GAIN (cap=2.0 fixed) ===")
GAINS = [1.0, 2.0, 3.0, 5.0]
results = [B.evaluate(f"gain={g}", build_pos_learned(g, 2.0)) for g in GAINS]

print("\n=== SWEEP: CAP (gain=3.0 fixed) ===")
CAPS = [1.5, 2.0, 3.0]
results += [B.evaluate(f"cap={c}", build_pos_learned(3.0, c)) for c in CAPS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"Best by rolling mean: {best['name']} rmean={best['rm']:.1f} (v10 rmean={B.base_scs.mean():.1f})")
