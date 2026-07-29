"""
test_batch100_I94.py

I94 (DIAGNOSTIC): regime-segmented (vol-tercile) consistency check across v7, v8, v9, v10. Does each
version's improvement over its predecessor hold up WITHIN each vol tercile, or is it concentrated in
one regime (e.g. only high-vol days)?

Regime proxy: trailing 20-day realized vol of the ALGO instrument's log-returns (same VOL_WIN used by
the ALGO leg itself), computed causally, over the 500-day OLD+NEW eval window (days 501-1000). Split
those days into low/med/high vol terciles by this proxy, then score each version's ACTUAL daily PnL
(continuity-correct, from the full walk-forward -- only the day *subset* used for mu/sd changes)
within each tercile.
"""
import numpy as np
import batch100_versions_shared as S

P_, nt = S.P_, S.nt
WIN = (500, nt)
S0, E0 = WIN
days_idx = np.arange(S0 + 1, E0 + 1)  # the 500 scored days, matches daily_pnl's tt>S convention

logp_algo = np.log(P_[0])
r_algo = np.diff(logp_algo)
VOL_WIN = 20
vol = np.full(len(logp_algo), np.nan)
for t in range(VOL_WIN, len(r_algo) + 1):
    vol[t] = r_algo[t - VOL_WIN:t].std()

vol_proxy = np.array([vol[d - 1] for d in days_idx])  # day tt's PnL uses price row tt-1 (0-indexed)
ok = ~np.isnan(vol_proxy)
print(f"vol proxy available for {ok.sum()}/{len(vol_proxy)} of the 500 eval days")

order = np.argsort(vol_proxy[ok])
n = ok.sum()
terc_labels = np.empty(n, dtype=int)
terc_labels[order[:n // 3]] = 0
terc_labels[order[n // 3: 2 * n // 3]] = 1
terc_labels[order[2 * n // 3:]] = 2
TNAMES = ["low-vol", "med-vol", "high-vol"]

PNL = {name: S.daily_pnl(S.POS[name], *WIN)[ok] for name in S.MODULES}


def score(mu, sd):
    if mu <= 0 or sd < 1e-10: return float(mu)
    sr = np.sqrt(250) * mu / sd
    return float(mu * sr ** 2 / (sr ** 2 + 1.0))


print(f"\n=== per-version score within each vol tercile (n_days per tercile ~ {n // 3}) ===")
ORDER = ["v7", "v8", "v9", "v10"]
tercile_scores = {t: {} for t in range(3)}
for t in range(3):
    mask = terc_labels == t
    print(f"\n  -- {TNAMES[t]} tercile (n={mask.sum()}) --")
    for name in ORDER:
        arr = PNL[name][mask]
        sc = score(arr.mean(), arr.std())
        tercile_scores[t][name] = sc
        print(f"    {name:<5} score={sc:8.1f}   mean/day=${arr.mean():.2f}")

print("\n=== does the monotonic v7 < v8 < v9 < v10 improvement hold WITHIN each tercile? ===")
for t in range(3):
    seq = [tercile_scores[t][n] for n in ORDER]
    mono = all(seq[i] < seq[i + 1] for i in range(3))
    print(f"  {TNAMES[t]:<9}: " + " -> ".join(f"{s:7.1f}" for s in seq) + f"   monotonic={mono}")

print("\n=== where does the FULL-WINDOW improvement (v7->v10) concentrate, by tercile? ===")
for t in range(3):
    gain = tercile_scores[t]["v10"] - tercile_scores[t]["v7"]
    print(f"  {TNAMES[t]:<9}: v10-v7 score gain = {gain:+.1f}")
