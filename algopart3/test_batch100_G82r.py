"""
test_batch100_G82r.py

QUESTION (G82): the shipped rank-stability signal only fires on DISAGREEMENT days (short-term move
opposes medium-term trend); on AGREEMENT days it is exactly 0. Does the agreement case also carry
usable information, with its own independently-weighted layer added on top of the full shipped v10
wz (which already includes the disagreement leg at RS_WEIGHT)?

MECHANISM: mirror-image raw signal, defined only on agreement days:
  agree_raw[i] = SIGN * short_z[i]   if sign(long_z[i]) == sign(short_z[i])  else 0
Standardize cross-sectionally per day exactly like the shipped signal, then blend as an ADDITIONAL,
independently-weighted layer on top of WZ_V10:
  wz_new = (1 - W2) * wz_v10 + W2 * agree_z * (|wz_v10|.mean())
SIGN=+1 tests "ride" (momentum-continuation when both horizons already agree); SIGN=-1 tests "fade"
(same direction as the disagreement leg, applied to the opposite case). Small W2 sweep around
RS_WEIGHT's own scale.
"""
import numpy as np
import batch100_common_gi as B

sanity_ok = B.print_sanity("(shared cache)")

logp, days, nIdio, nt = B.logp, B.days, B.nIdio, B.nt
RS_SHORT_W, RS_LONG_W = B.RS_SHORT_W, B.RS_LONG_W
WZ_V10 = B.WZ_V10

AGREE_RAW = np.full((nIdio, nt), np.nan)
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
    AGREE_RAW[:, t] = np.where(agree, sz, 0.0)


def build_pos_agree(sign_, w2):
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wz = WZ_V10[:, t]
        a = AGREE_RAW[:, t]
        if np.isfinite(a).all():
            raw = sign_ * a
            rstd = raw.std()
            a_z = (raw - raw.mean()) / (rstd + 1e-12) if rstd > 1e-12 else np.zeros(nIdio)
            wz = (1 - w2) * wz + w2 * a_z * (np.abs(wz).mean() + 1e-12)
        WZ[:, t] = wz
    return B.build_pos_from_wz(WZ)


print("\n=== G82: agreement-case signal added on top of v10, SIGN x W2 sweep ===")
results = []
for sign_ in [+1, -1]:
    for w2 in [0.005, 0.015, 0.03]:
        tag = f"SIGN={'+1(ride) ' if sign_ > 0 else '-1(fade) '} W2={w2}"
        POS = build_pos_agree(sign_, w2)
        results.append(B.evaluate(tag, POS))

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
