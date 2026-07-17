import itertools, sys
from combined_lab import evaluate

def run(grid, base):
    keys = list(grid)
    best = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        knobs = dict(base); knobs.update(dict(zip(keys, combo)))
        r = evaluate(**knobs)
        best.append((r[250][0], r[440][0], r[250][1], r[440][1], dict(zip(keys, combo))))
    best.sort(key=lambda x: -(x[0] + x[1]))   # rank by sum of both windows (robustness)
    print(f"{'S250':>7} {'S440':>7} {'Sh250':>6} {'Sh440':>6}  knobs")
    for s2, s4, h2, h4, k in best:
        print(f"{s2:7.1f} {s4:7.1f} {h2:6.2f} {h4:6.2f}  {k}")

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "blend"
    if which == "blend":
        run({"blend": [0.0, 0.1, 0.2, 0.35, 0.5], "rev_w": [3, 5, 10]},
            dict(half_life=2000, conv_z=0.2, contra_wz=60))
    elif which == "conv":
        run({"conv_z": [0.1, 0.15, 0.2, 0.25, 0.3], "blend": [0.0, 0.2]},
            dict(half_life=2000, rev_w=5, contra_wz=60))
