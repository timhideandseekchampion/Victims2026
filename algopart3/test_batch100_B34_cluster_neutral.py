"""
test_batch100_B34_cluster_neutral.py

B34: Re-test the cluster-neutral net-exposure constraint against v10, reusing the already-validated
12-stock idiosyncratic-correlation cluster from prior clustering work (test_stock_clustering.py:
kmeans k=6 on idio residual correlation, avg_within_corr=+0.13, permutation p=0%; the exact cluster
was previously used as a risk constraint in test_q20_item10_cluster_neutral.py against an OLD
baseline). Row indices below are the same 12-stock cluster (price-array indices, ALGO=0), reused
verbatim rather than re-clustering, per the batch instructions.

MECHANISM (identical to test_q20_item10_cluster_neutral.py): after computing v10's normal shipped
positions, on any day the book's net dollar exposure to the 12-stock cluster exceeds +-band, spread
an equal offsetting share-adjustment across the 12 members to bring net exposure back within the
band. A risk-control constraint, not a return-seeking one -- the README's "v7 budget" analysis
already flags every prior risk-control idea (Kelly, vol-targeting, drawdown throttles, confidence
ramps, and this exact cluster-neutral idea) as trading mean for variance at a 30:1 disadvantageous
rate at this operating point, so a priori this is expected to lose or be a wash; testing directly
against v10 rather than assuming that holds.
"""
import numpy as np, time
from batch100_shared import (
    nInst, nt, P_, dlr, POS_BASE, base_wo, base_wn, base_scs, SANITY_OK, evaluate
)

print(f"\n=== B34 sanity check (shared precompute) reproduces v10: {'PASS' if SANITY_OK else 'FAIL'} ===")
print(f"  OLD={base_wo:.1f} NEW={base_wn:.1f} rmean={base_scs.mean():.1f} rfloor={base_scs.min():.1f}")

CLUSTER = [1, 3, 11, 14, 20, 27, 28, 33, 34, 42, 44, 46]  # validated 12-stock cluster (price-array row idx)


def cluster_neutralize(POS, band_dollars):
    POS2 = POS.copy()
    for k in range(nt):
        cur = P_[CLUSTER, k]
        pos_c = POS2[CLUSTER, k]
        net = float((pos_c * cur).sum())
        if abs(net) <= band_dollars:
            continue
        excess = net - np.sign(net) * band_dollars
        adj_dollars_each = excess / len(CLUSTER)
        adj_shares = adj_dollars_each / cur
        new_pos_c = pos_c - adj_shares
        lim = (dlr[CLUSTER] / cur).astype(int)
        new_pos_c = np.clip(new_pos_c, -lim, lim)
        POS2[CLUSTER, k] = new_pos_c
    return POS2


print("\n=== B34 SWEEP: cluster-neutral band in {$0, $20k, $40k, $60k, $80k} ===")
t0 = time.time()
results = []
for band in (0, 20_000, 40_000, 60_000, 80_000):
    Pz = cluster_neutralize(POS_BASE, band)
    results.append(evaluate(f"band=${band:,}", Pz))
print(f"  sweep done ({time.time()-t0:.0f}s)")

passing = [c for c in results if c["passed"]]
print(f"\n{len(passing)}/{len(results)} bands beat v10 on OLD+NEW+rmean jointly.")
for c in sorted(results, key=lambda c: -c["rm"]):
    print(f"  {c['name']:<14} rmean={c['rm']:>7.1f}  rfloor={c['rf']:>7.1f}  n_worse={c['nworse']}/61")
