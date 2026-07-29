"""
test_batch100_J97_pca_resid.py

J97: Test a simple PCA-reconstruction-error per name (residual from projecting onto the top few
principal components) as an anomaly-flag conviction signal.

ARCHITECTURE NOTE (important for how this is testable at all): SAFE_llboost_v10's final idio position
is pos[i] = sign(wz_i) * dlr_i/cur_i -- i.e. FIXED dollar size, SIGN-ONLY. A pure magnitude/"conviction"
rescaling of wz that never flips its sign (e.g. wz *= (1 + gain*|resid_z|)) is a mathematical no-op on
this book: sign(wz*positive_scalar) == sign(wz) always, so it cannot change a single trade. The only way
a "conviction" signal can matter here is as an ADDITIVE component that can flip sign(wz) for borderline
names -- exactly how the existing rank-stability (RS) blend works. So J97 is implemented as an
additive blend, structurally identical to v10's own RS blend, stacked on TOP of v10's full wz:
  wz_new = (1-w)*WZ_V10 + w * sign_choice*resid_z * (mean|WZ_V10|)
tested both signs (resid momentum vs resid reversion) since "anomaly flag" doesn't specify direction.

MECHANISM: at each day t, using a trailing causal window of idio returns (PCA_WIN=250 days, min 60),
z-score each name's returns over the window, eigendecompose the resulting cross-sectional correlation
matrix, keep the top NPC=3 eigenvectors, project today's (t-1's, the latest realized) standardized
return vector onto them, and take the reconstruction residual per name -- the part of today's move NOT
explained by the dominant common factors (a multi-factor generalization of the existing single-factor
beta-demeaning already used in the ridge target).
"""
import numpy as np, time
from batch100_common_gi import (
    nInst, nt, nIdio, rs, days, WZ_V10, build_pos_from_wz, evaluate, print_sanity,
    base_wo, base_wn, base_scs,
)

SANITY_OK = print_sanity("(J97 PCA residual)")

PCA_WIN = 250
NPC = 3
MIN_SAMPLES = 60

print(f"\n=== precompute: PCA (NPC={NPC}, window={PCA_WIN}) reconstruction residual per name/day ===")
t0 = time.time()
RESID = np.full((nIdio, nt), np.nan)
for t in days:
    lo = max(0, t - PCA_WIN)
    win = rs[:, lo:t]
    if win.shape[1] < MIN_SAMPLES:
        continue
    mu = win.mean(1, keepdims=True); sd = win.std(1, keepdims=True) + 1e-12
    winz = (win - mu) / sd
    C = (winz @ winz.T) / winz.shape[1]
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(-evals)[:NPC]
    V = evecs[:, order]
    x = (rs[:, t - 1] - mu[:, 0]) / sd[:, 0]
    coef = V.T @ x
    recon = V @ coef
    RESID[:, t] = x - recon
print(f"  done ({time.time()-t0:.0f}s)")


def build_blend(sign, w):
    WZ = np.full((nIdio, nt), np.nan)
    for t in days:
        wzbase = WZ_V10[:, t]
        res = RESID[:, t]
        if not np.isfinite(res).all():
            WZ[:, t] = wzbase
            continue
        sig = sign * res
        sstd = sig.std()
        sig_z = (sig - sig.mean()) / (sstd + 1e-12) if sstd > 1e-12 else np.zeros(nIdio)
        WZ[:, t] = (1 - w) * wzbase + w * sig_z * (np.abs(wzbase).mean() + 1e-12)
    return build_pos_from_wz(WZ)


print("\n=== SWEEP: sign in {+1 (momentum: follow the unexplained move), "
      "-1 (reversion: fade it)}, weight w in {0.01,0.03,0.07,0.15} ===")
results = []
for sign, tag in [(1.0, "momentum"), (-1.0, "reversion")]:
    for w in (0.01, 0.03, 0.07, 0.15):
        Pz = build_blend(sign, w)
        results.append(evaluate(f"{tag} w={w}", Pz))

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
print(f"\nSANITY_OK={SANITY_OK}  base: OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} "
      f"rfloor={base_scs.min():.1f}")
