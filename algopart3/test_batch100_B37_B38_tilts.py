"""
test_batch100_B37_B38_tilts.py

B37: Re-test the idio beta-to-ALGO stability tilt against v10 (originally cleared Stage-1 significance,
p=0.000, but failed Stage-2 against an earlier baseline -- test_batch80_stage2_beta_disp.py: "degrades
other metrics once it's large enough to matter").
B38: Re-test the cross-sectional return-dispersion tilt against v10 (originally cleared Stage-1,
p=0.003, but failed Stage-2 as a per-stock tilt against an earlier baseline -- it predicts the AVERAGE
next-day return across stocks, not which stocks to prefer, so a per-stock differentiating tilt built
from it was expected to have ~zero effect; tested anyway as a uniform book-level tilt, per the original
methodology).

Both tilts, feature construction, and blend formula reused VERBATIM from test_batch80_stage2_beta_disp.py
(same BETA_W=60 rolling covariance-beta-to-ALGO, same stability_feat=-|beta_change|, same
disp_feat=cross-sectional std of idio returns each day) -- only the BASE the tilt is blended into
changes, from the old SAFE_llboost baseline to v10's ACTUAL final wz (WZ_FULL: ridge ensemble +
beta-adjusted target + BLEND reversion + boost + rank-stability blend).
"""
import numpy as np, time
from batch100_shared import (
    nInst, nIdio, nt, r, rs, P_, dlr, days, algo_pos, WZ_FULL, base_wo, base_wn, base_scs,
    SANITY_OK, evaluate
)

print(f"\n=== B37/B38 sanity check (shared precompute) reproduces v10: {'PASS' if SANITY_OK else 'FAIL'} ===")
print(f"  OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")

# ---------------------------------------------------------------------------------------------
# feature construction, verbatim from test_batch80_stage2_beta_disp.py
# ---------------------------------------------------------------------------------------------
BETA_W = 60
print("\ncomputing causal rolling beta-to-ALGO per stock (BETA_W=60) ...")
t0 = time.time()
r0 = r[0]
beta_roll = np.full((nIdio, nt - 1), np.nan)
for j in range(nIdio):
    for t in range(BETA_W, nt - 1):
        x = r0[t - BETA_W:t]; y = rs[j, t - BETA_W:t]
        if x.std() < 1e-12: continue
        beta_roll[j, t] = np.cov(x, y)[0, 1] / (x.var() + 1e-12)
beta_change = np.full((nIdio, nt - 1), np.nan)
beta_change[:, 1:] = np.diff(beta_roll, axis=1)
stability_feat = -np.abs(beta_change)  # higher = more stable
print(f"  done ({time.time()-t0:.0f}s)")

disp_feat = np.nanstd(rs, axis=0)  # (nt-1,) cross-sectional dispersion each day
disp_z = (disp_feat - np.nanmean(disp_feat)) / (np.nanstd(disp_feat) + 1e-12)


def beta_tilt(k):
    if k - 1 >= stability_feat.shape[1] or k - 1 < 0: return None
    return stability_feat[:, k - 1]


def disp_tilt(k):
    if k - 1 >= len(disp_z) or k - 1 < 0: return None
    return np.full(nIdio, disp_z[k - 1])


def build_pos_tilted(tilt_fn, tilt_w):
    POS = np.zeros((nInst, nt))
    for t in days:
        wz = WZ_FULL[:, t].copy()
        if tilt_w > 0:
            tilt = tilt_fn(t)
            if tilt is not None:
                tz = tilt - np.nanmean(tilt)
                tz = tz / (np.nanstd(tz) + 1e-12)
                tz = np.nan_to_num(tz)
                wz = (1 - tilt_w) * wz + tilt_w * tz
        cur = P_[:, t]; lim = (dlr[1:] / cur[1:]).astype(int)
        POS[1:, t] = np.clip(np.sign(wz) * (dlr[1:] / cur[1:]), -lim, lim)
    POS[0, :] = algo_pos
    return POS


print("\n### B37: beta-to-ALGO stability tilt (Stage-1 p=0.000) ###")
b37_results = []
for w in (0.02, 0.05, 0.1, 0.15, 0.2):
    Pz = build_pos_tilted(beta_tilt, w)
    b37_results.append(evaluate(f"beta-stability tilt w={w}", Pz))

print("\n### B38: cross-sectional dispersion, uniform book-level tilt (Stage-1 p=0.003) ###")
b38_results = []
for w in (0.02, 0.05, 0.1):
    Pz = build_pos_tilted(disp_tilt, w)
    b38_results.append(evaluate(f"dispersion uniform-tilt w={w}", Pz))

print("\n=== summary ===")
for label, res in (("B37 beta-stability", b37_results), ("B38 dispersion", b38_results)):
    passing = [c for c in res if c["passed"]]
    print(f"{label}: {len(passing)}/{len(res)} configs beat v10 on OLD+NEW+rmean jointly.")
    for c in sorted(res, key=lambda c: -c["rm"]):
        print(f"  {c['name']:<32} rmean={c['rm']:>7.1f}  rfloor={c['rf']:>7.1f}  n_worse={c['nworse']}/61")
