"""
test_batch100_G82.py

QUESTION (G82): the shipped rank-stability signal only fires when short-term and medium-term trend
DISAGREE (a pullback-within-trend). On AGREEMENT days (short & medium trend point the same way) the
signal is exactly 0 -- i.e. half the (name, day) cells carry no information at all under the shipped
construction. Does the agreement case ALSO carry usable information, with its own independent weight
(distinct from RS_WEIGHT, which only governs the disagreement leg)? Prior expectation stated in the
task: likely OPPOSITE sign to the disagreement leg (disagreement leg fades short-term moves; the
"likely opposite sign" hypothesis for agreement would be to instead RIDE the short-term move, i.e.
+short_z, a momentum-continuation bet when both horizons already agree).

MECHANISM: build a second raw signal, defined only on agreement days (0 on disagreement days --
exact mirror of how the shipped signal is 0 on agreement days):
  agree_raw[i] = SIGN * short_z[i]   if sign(long_z[i]) == sign(short_z[i])  else 0
Standardize cross-sectionally per day exactly like the shipped signal, then blend it in as an
ADDITIONAL, independently-weighted layer on top of the full shipped v10 wz (which already includes
the disagreement leg at RS_WEIGHT):
  wz_new = (1 - W2) * wz_v10 + W2 * agree_z * (|wz_v10|.mean())
Test SIGN in {+1 (momentum/ride), -1 (fade, same direction as the disagreement leg)} x a small
W2 sweep.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

RS_SHORT_W, RS_LONG_W = B.RS_SHORT_W, B.RS_LONG_W
logp, days, nIdio, nt = B.logp, B.days, B.nIdio, B.nt
WZ_V10 = B.WZ_V10


def agree_raw(sign):
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
        agree = np.sign(lz) == np.sign(sz)
        RAW[:, t] = np.where(agree, sign * sz, 0.0)
    return RAW


def build_pos_agree(sign, w2):
    RAW = agree_raw(sign)
    WZ = np.zeros((nIdio, nt))
    for t in days:
        wz = WZ_V10[:, t]
        a = RAW[:, t]
        if np.isfinite(a).all():
            astd = a.std()
            a_z = (a - a.mean()) / (astd + 1e-12) if astd > 1e-12 else np.zeros(nIdio)
            wz = (1 - w2) * wz + w2 * a_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return B.build_pos_from_wz(WZ)


print("\n=== sanity check (idea-specific): W2=0 must reproduce SAFE_llboost_v10 exactly (no-op) ===")
POS0 = build_pos_agree(1.0, 0.0)
r0 = B.evaluate("W2=0 (no-op)", POS0)
idea_sanity_ok = sanity_ok and abs(r0["wo"] - B.base_wo) < 0.5 and abs(r0["wn"] - B.base_wn) < 0.5
print("  OK." if idea_sanity_ok else "  *** WARNING: W2=0 does not reproduce v10 -- check logic. ***")

W2S = [0.005, 0.01, 0.015, 0.02, 0.03]
all_results = []
for sign, tag in [(1.0, "MOMENTUM(+sz)"), (-1.0, "FADE(-sz)")]:
    print(f"\n=== SWEEP: agreement-case signal = {tag}, W2 in {W2S} ===")
    for w2 in W2S:
        r = B.evaluate(f"{tag} W2={w2}", build_pos_agree(sign, w2))
        all_results.append(r)

passing = [c for c in all_results if c["passed"]]
print(f"\n{len(passing)}/{len(all_results)} (sign, W2) configs beat v10 on OLD+NEW+rmean jointly.")
best = max(all_results, key=lambda c: c["rm"])
print(f"Best by rolling mean: {best['name']} rmean={best['rm']:.1f} (v10 rmean={B.base_scs.mean():.1f})")
