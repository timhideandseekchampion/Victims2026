"""Forward-MC the full adaptive file (ridge + OLS-ALGO + gated xs/corr/ar1) vs the OLS-adaptive
baseline (no aux sleeves) across the 3 worlds. If adaptive ~= baseline everywhere, the gates don't
misfire out-of-sample (they stay off). If adaptive < baseline, gates misfire (cost). If >, a sleeve
helpfully activated."""
import numpy as np, importlib.util
from forward_mc import world_b, world_c, fwd_score, real, nInst
from dgp_simulator import DGP

spec = importlib.util.spec_from_file_location(
    "adap", "/home/SIG2026/combinedv3/Arbitrage_Victims_combined_adaptive.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


def make_gp(sleeves):
    def gpf():
        mod.GATED_SLEEVES = sleeves
        mod._cache = {"fit_t": None, "model": None}
        return mod.getMyPosition
    return gpf


NSEED, FUT = 8, 180
dgpA = DGP.fit(real); dgpA.signal_scale = 0.3; T = real.shape[1]
worlds = {"A: VAR": [dgpA.extend(real, FUT, s) for s in range(NSEED)],
          "B: pairs": [world_b(real, FUT, s) for s in range(NSEED)],
          "C: shift": [world_c(dgpA, real, FUT, s) for s in range(NSEED)]}
print(f"Forward MC (median of {NSEED} unseen futures, FUT={FUT}):\n")
print(f"{'config':28}" + "".join(f"{w:>12}" for w in worlds))
for name, sleeves in [("baseline (OLS-ALGO, no aux)", ()),
                      ("adaptive (xs+corr+ar1 gated)", ("xs", "corr", "ar1"))]:
    row = f"{name:28}"
    for wname, panels in worlds.items():
        scs = np.array([fwd_score(make_gp(sleeves), p, T + 1) for p in panels])
        row += f"{np.median(scs):12.0f}"
    print(row)
print("DONE")
