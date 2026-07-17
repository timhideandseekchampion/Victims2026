"""Decision harness for the adaptive (forgetting) OLS estimator.

Compares expanding vs rolling vs EWLS on:
  (1) walk-forward cross-sectional IC on the real data (scale-free edge metric),
  (2) eval.py Score / Sharpe / turnover via backtest_full.calcPL, and
  (3) a SIMULATED regime-break recovery test (the only way to show adaptivity on a
      single, stable price file).
Everything reuses the verified eval.py-faithful accounting in backtest_full.
"""
import numpy as np
from scipy.stats import spearmanr
import backtest_full as bt
import adaptive_estimator as ae
import ols_adaptive

SEED = 12345
prc = bt.prcAll                                    # (51, 500)
R = np.diff(np.log(prc), axis=1).T                 # (499, 51): row=day, col=asset
X, Y = R[:-1], R[1:, 1:]                            # predict next-day 50 from today's 51
csd = lambda v: v - v.mean()

SCHEMES = [
    ("expanding",      {"scheme": "expanding"}),
    ("ewls h=250",     {"scheme": "ewls", "half_life": 250}),
    ("ewls h=120  *",  {"scheme": "ewls", "half_life": 120}),
    ("ewls h=60",      {"scheme": "ewls", "half_life": 60}),
    ("rolling N=120",  {"scheme": "rolling", "window": 120}),
]

def wf_ic(cfg, t0=100, refit=5):
    """Walk-forward daily cross-sectional IC + sign hit-rate on the real data."""
    ic, hit, model = [], [], None
    for t in range(t0, len(X)):
        if model is None or (t - t0) % refit == 0:
            model = ae.fit_rows(X[:t], Y[:t], cfg)
        B, mx, my = model
        pred = my + (X[t] - mx) @ B
        ic.append(spearmanr(csd(pred), csd(Y[t]))[0])
        hit.append(np.mean(np.sign(csd(pred)) == np.sign(csd(Y[t]))))
    ic = np.array(ic)
    t = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    return ic.mean(), t, 100 * (ic > 0).mean(), 100 * np.mean(hit)

print("=" * 78, "\n1. WALK-FORWARD CROSS-SECTIONAL IC (real data, t0=100, refit=5)")
print(f"   {'scheme':<15}{'IC':>9}{'t-stat':>9}{'%pos':>8}{'sign-hit%':>11}")
base_ic = None
for name, cfg in SCHEMES:
    m, t, pos, hit = wf_ic(cfg)
    if base_ic is None: base_ic = m
    print(f"   {name:<15}{m:>+9.4f}{t:>9.2f}{pos:>7.0f}%{hit:>10.1f}%")

print("=" * 78, "\n2. eval.py SCORE / SHARPE / TURNOVER (via backtest_full, MAX sizing)")
print(f"   {'scheme':<15}{'Score@250':>11}{'Sh@250':>8}{'Score@400':>11}{'Sh@400':>8}{'Δscore%':>9}")
base_score = None
for name, cfg in SCHEMES:
    ols_adaptive.CONFIG = {**cfg, "refit_every": 1}
    m2 = bt.calcPL(prc, 250, ols_adaptive); s250 = bt.score(m2[0], m2[1])
    m4 = bt.calcPL(prc, 400, ols_adaptive); s400 = bt.score(m4[0], m4[1])
    if base_score is None: base_score = s250
    d = 100 * (s250 - base_score) / base_score
    print(f"   {name:<15}{s250:>11.1f}{m2[2]:>8.2f}{s400:>11.1f}{m4[2]:>8.2f}{d:>+8.1f}%")

# ---------------------------------------------------------------------------
print("=" * 78, f"\n3. SIMULATED REGIME-BREAK RECOVERY (synthetic, seed={SEED})")
# Clean estimator-memory demo: a well-conditioned linear regression whose true
# coefficients FLIP at the break. No price/VAR process (nothing to diverge), no
# static intercept (nothing to swamp the cross-section). This isolates the pure
# mechanism — how fast each window re-learns a changed relationship. NOTE it is
# high-SNR by construction; at the REAL data's SNR (sections 1-2) the coefficient
# fit is noise-dominated, so this recovery advantage is NOT realisable there.
rng = np.random.default_rng(SEED)
p, q, L, BREAK, NOISE = 51, 50, 620, 320, 0.7
Btrue = rng.standard_normal((p, q)) * 0.15
Xs = rng.standard_normal((L, p))
truth = lambda t: Xs[t] @ (Btrue if t < BREAK else -Btrue)   # relationship flips sign at BREAK
Ys = np.array([truth(t) for t in range(L)]) + rng.standard_normal((L, q)) * NOISE

def recovery(cfg, t0=60):
    ic = []
    for t in range(t0, L):
        B, mx, my = ae.fit_rows(Xs[:t], Ys[:t], cfg)     # same fit the strategy uses
        ic.append(spearmanr(csd(my + (Xs[t] - mx) @ B), csd(truth(t)))[0])
    ic = np.array(ic); days = np.arange(t0, L)
    roll = np.convolve(ic, np.ones(15) / 15, mode="same")
    pre = roll[(days >= BREAK - 45) & (days < BREAK - 5)].mean()
    after = np.where((days > BREAK + 5) & (roll >= 0.8 * pre))[0]
    ttr = int(days[after[0]] - BREAK) if len(after) else None
    return pre, roll, days, ttr

print(f"   idealized high-SNR demo; relationship flips sign at day {BREAK}:")
print(f"   {'scheme':<15}{'pre-IC':>9}{'days→80%':>11}")
curves = {}
for name, cfg in [("expanding", {"scheme": "expanding"}),
                  ("ewls h=120", {"scheme": "ewls", "half_life": 120}),
                  ("ewls h=60", {"scheme": "ewls", "half_life": 60}),
                  ("rolling N=90", {"scheme": "rolling", "window": 90})]:
    pre, roll, days, ttr = recovery(cfg)
    curves[name] = (roll, days)
    print(f"   {name:<15}{pre:>+9.3f}{(str(ttr) if ttr is not None else '>300'):>11}")

print("\n   rolling-15 IC vs the (flipped) truth after the break:")
grid = np.arange(BREAK, L - 10, 20)
print("   " + " " * 15 + "".join(f"{d-BREAK:>5}" for d in grid) + "   (days after break)")
for name, (roll, days) in curves.items():
    print(f"   {name:<15}" + "".join(f"{roll[np.argmin(np.abs(days-d))]:>5.2f}" for d in grid))

print("=" * 78)
print("READ: On the real, stable data (sections 1-2) forgetting only COSTS — h=250 is")
print("~free (-1.6%), h=120 costs -12.7%, shorter costs more. Section 3 shows the")
print("mechanism DOES work when a signal is trackable (expanding re-learns a flipped")
print("relationship slowly; short half-lives fast). At this data's true SNR that")
print("benefit is not realisable, so the honest default is near-expanding: h=250.")
