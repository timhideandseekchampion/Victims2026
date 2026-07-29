"""
test_batch100_I91.py

I91 (DIAGNOSTIC): does the edge-concentration pattern found for v9-to-v10 (top-10 days carry a large
share of the total edge, "lumpier" than v9-to-v8's broad "death by a thousand cuts" edge, per the
README's parametric-stress-test follow-up) also hold for v7-to-v8 and v8-to-v9? Is concentration a
growing trend across the version sequence?

Methodology matches the README table exactly: real-data (v_b - v_a) daily PnL difference over the
500 days spanning the OLD+NEW eval window (days 501-1000), using each version's ACTUAL getMyPosition
positions (from batch100_versions_shared's direct walk-forward simulation, sanity-checked there).
"""
import numpy as np
import batch100_versions_shared as S

nt = S.nt
WIN = (500, nt)  # 500 days: 501-1000, same window the README's v9-vs-v8 / v10-vs-v9 table used

PNL = {name: S.daily_pnl(S.POS[name], *WIN) for name in S.MODULES}
print(f"per-version daily PnL arrays built over days {WIN[0]+1}-{WIN[1]} ({len(PNL['v10'])} days)")


def concentration_report(name_a, name_b, label):
    a, b = PNL[name_a], PNL[name_b]
    diff = b - a
    n = len(diff)
    differing = np.abs(diff) > 1e-9
    n_diff = int(differing.sum())
    win_rate = float((diff[differing] > 0).mean()) if n_diff else float("nan")
    order = np.argsort(-np.abs(diff))
    top5 = diff[order[:5]]; top10 = diff[order[:10]]
    total_abs = np.abs(diff).sum()
    top5_share = float(np.abs(top5).sum() / total_abs) if total_abs > 0 else float("nan")
    top10_share = float(np.abs(top10).sum() / total_abs) if total_abs > 0 else float("nan")
    full_total = float(diff.sum())
    excl_top10 = float(full_total - top10.sum())
    print(f"\n{label} ({name_b} - {name_a}, over {n} days):")
    print(f"  days differing:        {n_diff}/{n} ({100*n_diff/n:.0f}%)")
    print(f"  win rate on differing:  {100*win_rate:.1f}%")
    print(f"  top 5  |diff| share:    {100*top5_share:.1f}%")
    print(f"  top 10 |diff| share:    {100*top10_share:.1f}%")
    print(f"  full total effect:      {full_total:.0f}")
    print(f"  excl. top 10 days:      {excl_top10:.0f}")
    return dict(n_diff=n_diff, n=n, win_rate=win_rate, top5=top5_share, top10=top10_share,
                full_total=full_total, excl_top10=excl_top10)


print("\n=== reference (already known, README): v9-vs-v8 and v10-vs-v9 ===")
print("  v9 vs v8: 317/500 differing (63%), win 52.7%, top5 11.0%, top10 19.0%, 1390 -> 4703")
print("  v10 vs v9: 130/500 differing (26%), win 57.7%, top5 26.5%, top10 40.2%, 10273 -> 4260")

r78 = concentration_report("v7", "v8", "v8 vs v7 (ALGO deadband)")
r89 = concentration_report("v8", "v9", "v9 vs v8 (beta-demean) -- re-derived here, should match README")
r910 = concentration_report("v9", "v10", "v10 vs v9 (rank-stability) -- re-derived here, should match README")

print("\n=== trend across the version sequence (top-10 |diff| share of total edge) ===")
print(f"  v7->v8:  {100*r78['top10']:.1f}%")
print(f"  v8->v9:  {100*r89['top10']:.1f}%   (README: 19.0%)")
print(f"  v9->v10: {100*r910['top10']:.1f}%   (README: 40.2%)")
