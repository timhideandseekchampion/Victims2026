"""
test_batch100_G81.py

QUESTION (G81): the shipped rank-stability signal is a hard 0-or-value step function:
  signal[i] = -short_z[i]  if sign(long_z[i]) != sign(short_z[i])  else 0
i.e. a discontinuity exactly at long_z*short_z == 0 (agreement boundary), and the disagreement
"strength" (how far past the boundary) plays no role once past it (only the raw -short_z magnitude
does). Does a CONTINUOUS version -- one that fades in/out smoothly around the agreement boundary
instead of snapping between "fully on" and "fully off" -- do better?

MECHANISM: replace the hard gate 1[disagree] with a smooth logistic gate of the same underlying
quantity (long_z * short_z, negative = disagreement, positive = agreement):
  gate[i] = sigmoid(-K * long_z[i] * short_z[i])          (-> 1 deep in disagreement, -> 0 deep in
                                                              agreement, = 0.5 exactly at the boundary)
  signal[i] = -short_z[i] * gate[i]
As K -> large this recovers the shipped step function (up to the measure-zero boundary itself); small
K instead spreads the transition out, so days/names near the boundary get a small, continuously-scaled
signal rather than either the full magnitude or exactly zero. Same RS_WEIGHT blend as shipped
(this idea only changes the raw signal fed into that blend, not the blend weight itself).

Reuses batch100_common_gi (RS_RAW / WZ_PRE / BOOST cached) for everything except the RS raw signal
itself, which must be recomputed here since the continuous gate needs sz and lz separately (RS_RAW
only stores the already-gated shipped value).
"""
import numpy as np
import batch100_common_gi as B

print_sanity = B.print_sanity
sanity_ok = print_sanity("(shared cache)")

WZ_PRE, BOOST, BOOST_K, BOOST_MIN_DAY = B.WZ_PRE, B.BOOST, B.BOOST_K, B.BOOST_MIN_DAY
RS_SHORT_W, RS_LONG_W, RS_WEIGHT = B.RS_SHORT_W, B.RS_LONG_W, B.RS_WEIGHT
logp, days, nIdio, nt = B.logp, B.days, B.nIdio, B.nt

WZ_PREBOOST = np.zeros((nIdio, nt))
for t in days:
    wz = WZ_PRE[:, t].copy()
    if t >= BOOST_MIN_DAY:
        wz = wz + BOOST_K * BOOST[:, t]
    WZ_PREBOOST[:, t] = wz


def continuous_rs_raw(K):
    RAW = np.full((nIdio, nt), np.nan)
    for t in days:
        if t < max(RS_SHORT_W, RS_LONG_W) + 5:
            continue
        short_ret = logp[1:, t] - logp[1:, t - RS_SHORT_W]
        long_ret = logp[1:, t] - logp[1:, t - RS_LONG_W]
        sz = short_ret - short_ret.mean(); sstd = sz.std()
        lz = long_ret - long_ret.mean(); lstd = lz.std()
        if sstd < 1e-12 or lstd < 1e-12:
            continue
        sz = sz / sstd; lz = lz / lstd
        gate = 1.0 / (1.0 + np.exp(K * lz * sz))
        RAW[:, t] = -sz * gate
    return RAW


def build_pos_continuous(K):
    RAW = continuous_rs_raw(K)
    WZ = np.zeros((nIdio, nt))
    for t in days:
        wz = WZ_PREBOOST[:, t]
        s = RAW[:, t]
        if np.isfinite(s).all():
            sstd = s.std()
            s_z = (s - s.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - RS_WEIGHT) * wz + RS_WEIGHT * s_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return B.build_pos_from_wz(WZ)


print("\n=== sanity check (idea-specific): K->inf (K=200) must ~reproduce SAFE_llboost_v10 "
      "(logistic gate saturates to the hard step almost everywhere) ===")
POS_hard = build_pos_continuous(200.0)
r_hard = B.evaluate("K=200 (~hard step)", POS_hard)
idea_sanity_ok = sanity_ok and abs(r_hard["wo"] - B.base_wo) < 1.0 and abs(r_hard["wn"] - B.base_wn) < 1.0
print("  OK -- continuous-gate machinery reproduces v10 in the hard-step limit."
      if idea_sanity_ok else "  *** WARNING: does not reproduce v10 in hard-step limit -- check logic. ***")

print("\n=== SWEEP: logistic gate steepness K (smaller K = smoother/more continuous) ===")
KS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
results = [B.evaluate(f"K={k}", build_pos_continuous(k)) for k in KS]

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} K values beat v10 on OLD+NEW+rmean jointly.")
best = max(results, key=lambda c: c["rm"])
print(f"Best by rolling mean: K={best['name']} rmean={best['rm']:.1f} (v10 rmean={B.base_scs.mean():.1f})")
