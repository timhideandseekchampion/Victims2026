"""
test_batch100_G81r.py

QUESTION (G81): the shipped rank-stability signal is a hard 0-or-value step function:
  signal[i] = -short_z[i]  if sign(long_z[i]) != sign(short_z[i])  else 0
i.e. a discontinuity exactly at the long_z/short_z agreement boundary -- disagreement "strength" past
the boundary plays no role (only raw -short_z magnitude does), and there is a hard jump from 0 to
-short_z right at the boundary. Does a CONTINUOUS version -- smoothly fading the signal in/out around
the boundary instead of snapping between fully-off and fully-on -- do better?

MECHANISM: replace the hard indicator 1[disagree] with a smooth logistic gate of the same underlying
quantity (long_z*short_z; negative = disagreement, positive = agreement):
  gate[i]   = sigmoid(-K * long_z[i] * short_z[i])   (-> 1 deep in disagreement, -> 0 deep in
                                                          agreement, = 0.5 exactly at the boundary)
  signal[i] = -short_z[i] * gate[i]
As K grows this approaches the shipped step function; small K spreads the transition out. Same
RS_WEIGHT blend as shipped (only the raw signal fed into that blend changes, not the blend weight).

Reuses batch100_common_gi's cached WZ_PRE/BOOST (expensive part, untouched by this idea) and only
recomputes the cheap per-day cross-sectional short/long return z-scores.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

logp, days, nIdio, nt = B.logp, B.days, B.nIdio, B.nt
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = B.RS_SHORT_W, B.RS_LONG_W, B.RS_WEIGHT
WZ_PRE, BOOST, BOOST_MIN_DAY, BOOST_K = B.WZ_PRE, B.BOOST, B.BOOST_MIN_DAY, B.BOOST_K


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


# raw (unstandardized-to-signal, but per-day cross-sectionally z-scored) short/long return z-scores --
# identical math to V10._rank_stability_signal, just kept separately (pre-gate) instead of collapsed.
SZ = np.full((nIdio, nt), np.nan)
LZ = np.full((nIdio, nt), np.nan)
for t in days:
    if t < max(RS_SHORT_W, RS_LONG_W) + 5:
        continue
    short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
    long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
    sz = short_ret - short_ret.mean(); sstd = sz.std()
    lz = long_ret - long_ret.mean(); lstd = lz.std()
    if sstd < 1e-12 or lstd < 1e-12:
        continue
    SZ[:, t] = sz / sstd
    LZ[:, t] = lz / lstd


def build_pos_continuous(K):
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = WZ_PRE[:, t].copy()
        if t >= BOOST_MIN_DAY:
            wz = wz + BOOST_K * BOOST[:, t]
        sz, lz = SZ[:, t], LZ[:, t]
        if np.isfinite(sz).all() and np.isfinite(lz).all():
            gate = sigmoid(-K * lz * sz)
            raw = -sz * gate
            rstd = raw.std()
            s_z = (raw - raw.mean()) / (rstd + 1e-12) if rstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return B.build_pos_from_wz(WZ)


print("\n=== G81: continuous rank-stability gate sigmoid(-K*long_z*short_z) vs shipped hard step ===")
results = []
for K in [0.5, 1, 2, 4, 8, 16]:
    POS = build_pos_continuous(K)
    results.append(B.evaluate(f"K={K}", POS))

passing = [r for r in results if r["passed"]]
print(f"\n{len(passing)}/{len(results)} configs beat v10 on OLD+NEW+rmean jointly.")
if passing:
    for r in passing:
        print(f"  {r['name']:<28} rmean={r['rm']:.1f} n_worse={r['nworse']}/61")
else:
    print("Ranked by rolling mean, closest first:")
    for r in sorted(results, key=lambda r: -r["rm"]):
        print(f"  {r['name']:<28} OLD={r['wo']:>7.1f} NEW={r['wn']:>7.1f} rmean={r['rm']:>7.1f} "
              f"rfloor={r['rf']:>7.1f} n_worse={r['nworse']}/61")
