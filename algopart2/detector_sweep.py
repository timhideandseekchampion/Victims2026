"""detector_sweep.py — EXHAUSTIVE test of the 'time the lead-lag weakness' idea.
Grid of detector variants (absolute-IC and relative-IC timing, many W/slope/ref/bounds).
For EACH variant the metric is EXCESS over the static blend at its OWN mean blend --
i.e. does the TIMING add anything beyond just choosing that average blend? (Without this
control, a 'winning' detector is just blend selection re-labelled, and would overfit on
~1.5 independent windows.) A real detector must show excess_floor > 0 AND excess_mean ~>= 0."""
import numpy as np
import detector as D
run = D.run; IC_LL = D.IC_LL; IC_REV = D.IC_REV; nDays = D.nDays

def trailing(ic, t, W):
    v = [ic[s] for s in range(t - W, t) if s in ic]
    return np.mean(v) if len(v) >= max(5, W // 2) else None

def mk_absIC(W, slope, ref, blo, bhi):
    def fn(t):
        m = trailing(IC_LL, t, W)
        if m is None: return 0.25
        frac = np.clip((ref - m) / ref, -1.0, 1.0)      # +1 = IC collapsed -> more reversion
        return float(np.clip(0.25 + slope * frac, blo, bhi))
    return fn

def mk_relIC(W, blo, bhi):
    def fn(t):
        a = trailing(IC_LL, t, W); b = trailing(IC_REV, t, W)
        if a is None or b is None: return 0.25
        a = max(a, 0.0); b = max(b, 0.0)
        if a + b < 1e-9: return 0.25
        return float(np.clip(b / (a + b), blo, bhi))       # follow whichever signal is working
    return fn

# window sets (overlapping; sample is thin -- that's WHY the control matters)
W500 = [(e - 500, e) for e in range(500, nDays + 1, 20)]
W250 = [(e - 250, e) for e in range(346, nDays + 1, 25)]

# precompute static-blend scores on a fine grid, per window, for the control lookup
BG = np.round(np.arange(0.05, 0.501, 0.01), 2)
static = {L: {b: np.array([run(Sd, Ed, (lambda bb: (lambda t: bb))(b)) for (Sd, Ed) in wins])
              for b in BG} for L, wins in (("500", W500), ("250", W250))}
def control(L, mean_blend):
    b = BG[np.argmin(np.abs(BG - mean_blend))]
    return static[L][b], b

variants = []
for W in (10, 15, 20, 30, 40, 60):
    for slope in (0.10, 0.20, 0.30, 0.40):
        for ref in (0.05, 0.064, 0.079):
            for (blo, bhi) in ((0.15, 0.40), (0.10, 0.50)):
                variants.append((f"absIC W{W} s{slope} r{ref} [{blo},{bhi}]", mk_absIC(W, slope, ref, blo, bhi)))
for W in (10, 15, 20, 30, 40, 60):
    for (blo, bhi) in ((0.15, 0.40), (0.10, 0.50)):
        variants.append((f"relIC W{W} [{blo},{bhi}]", mk_relIC(W, blo, bhi)))
print(f"testing {len(variants)} detector variants, each vs its static-blend control\n")

rows = []
for name, fn in variants:
    rec = {"name": name}
    ok = True
    for L, wins in (("500", W500), ("250", W250)):
        adp = np.array([run(Sd, Ed, fn) for (Sd, Ed) in wins])
        mb = np.mean([fn(t) for (Sd, Ed) in wins for t in range(Sd + 1, Ed) if t >= 96]) if False else \
             np.mean([fn(t) for t in range(500, nDays)])   # mean blend proxy over live-ish region
        ctrl, cb = control(L, mb)
        rec[L] = (adp.mean(), adp.min(), ctrl.mean(), ctrl.min(), cb)
    rows.append(rec)

# a detector "works" only if it beats its control on FLOOR without losing mean, on BOTH horizons
def excess(rec):
    e = []
    for L in ("500", "250"):
        am, af, cm, cf, cb = rec[L]
        e.append((af - cf, am - cm))
    return e
print("Best variants by (floor excess over control), summed across horizons:")
print(f"{'variant':<34}{'500 dFloor':>11}{'500 dMean':>10}{'250 dFloor':>11}{'250 dMean':>10}")
rows.sort(key=lambda r: -(excess(r)[0][0] + excess(r)[1][0]))
for rec in rows[:8]:
    e = excess(rec)
    print(f"{rec['name']:<34}{e[0][0]:>11.1f}{e[0][1]:>10.1f}{e[1][0]:>11.1f}{e[1][1]:>10.1f}")
print("...")
for rec in rows[-2:]:
    e = excess(rec)
    print(f"{rec['name']:<34}{e[0][0]:>11.1f}{e[0][1]:>10.1f}{e[1][0]:>11.1f}{e[1][1]:>10.1f}")

nbeat = sum(1 for r in rows if excess(r)[0][0] > 0 and excess(r)[1][0] > 0 and excess(r)[0][1] >= -5 and excess(r)[1][1] >= -5)
print(f"\nvariants beating control on floor (both horizons) without losing >5 mean: {nbeat}/{len(rows)}")
