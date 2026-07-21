"""gate_sweep.py — exhaustive switch-config search on the responsiveness/whipsaw/capture trilemma.
Grid: GATE_MODE x window x margin x persistence. Metric caches (_SIG/_RET/_ICD/_PN...) are pure
functions of (name, day), independent of the gate knobs, so we build them ONCE per price panel and
sweep every config cheaply on top. Flags any config that is a CLEAN WIN vs the current ic gate:
  real >= 690 (inert)  AND  momentum > ic  AND  flip >= 220k (chop-robust)  AND  noise >= 50k.
"""
import numpy as np, pandas as pd
import SAFE_rotate as R
prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape
commRate = np.full(nInst, 1e-4); commRate[0] = 2e-5
dlr = np.full(nInst, 10_000.0); dlr[0] = 100_000.0

def make_ext(kind, cross_mom=0.6, T_ext=150, period=25, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc).copy(); vol = np.diff(logp[1:], axis=1).std()
    names = logp[1:].copy(); K = 5
    for step in range(T_ext):
        trail = names[:, -1] - names[:, -K]; tc = trail - trail.mean()
        if kind == "noise": drift = np.zeros(50)
        elif kind == "flip":
            sgn = 1.0 if (step // period) % 2 == 0 else -1.0; drift = sgn * cross_mom * (tc/(tc.std()+1e-9)) * vol
        else: drift = cross_mom * (tc/(tc.std()+1e-9)) * vol
        drift -= drift.mean(); noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0)); full[:, :nDays] = prc
    return full

def clearR():
    for c in ("_SIG", "_RET", "_ICD", "_AZ", "_XC", "_PN"): getattr(R, c).clear()

def evalone(P, S, E):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(S, E + 1):
        cur = P[:, t - 1]; newPos = R.getMyPosition(P[:, :t]) if t < E else cp
        dP = newPos - cp; cash -= cur.dot(dP) + comm; comm = np.sum(cur * np.abs(dP) * commRate); cp = newPos
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: pll.append(pl)
    p = np.array(pll); mu, sd = p.mean(), p.std()
    return p.sum(), (mu * (np.sqrt(250)*mu/sd)**2 / ((np.sqrt(250)*mu/sd)**2 + 1) if mu > 0 else mu)

def setcfg(mode, W, margin, P):
    R.GATE_MODE, R.ROT_W, R.PNL_MARGIN, R.ROT_P = mode, W, margin, P

# config grid
CFGS = [("ic", 40, 0.0, 7)]                                    # current baseline
for W in (30, 40, 60):
    for P in (5, 7):
        for mar in (0.0, 100.0, 300.0):
            CFGS += [("pnl", W, mar, P), ("sharpe", W, mar, P)]
for W in (30, 40, 60):
    CFGS += [("softblend", W, 0.0, 7)]

panels = {"real": (prc, 500, 750), "momentum": (make_ext("momentum"), nDays, nDays+150),
          "flip": (make_ext("flip"), nDays, nDays+150), "noise": (make_ext("noise"), nDays, nDays+150)}
res = {}
for pname, (P, S, E) in panels.items():
    clearR()                                                   # new panel -> rebuild metric cache once
    for cfg in CFGS:
        setcfg(*cfg)
        tot, sc = evalone(P, S, E)
        res.setdefault(cfg, {})[pname] = (sc if pname == "real" else tot)

ic = res[("ic", 40, 0.0, 7)]
print(f"{'mode':<10}{'W':>3}{'mar':>6}{'P':>3}{'REALsc':>8}{'MOM':>9}{'FLIP':>9}{'NOISE':>8}{'  clean-win?':>12}")
for cfg in CFGS:
    m, W, mar, P = cfg; r = res[cfg]
    win = r["real"] >= 690 and r["momentum"] > ic["momentum"] and r["flip"] >= 220000 and r["noise"] >= 50000
    tag = "  <-- WIN" if (win and cfg != ("ic",40,0.0,7)) else ("  (current)" if cfg == ("ic",40,0.0,7) else "")
    print(f"{m:<10}{W:>3}{mar:>6.0f}{P:>3}{r['real']:>8.0f}{r['momentum']:>9.0f}{r['flip']:>9.0f}{r['noise']:>8.0f}{tag}")
print("\nCLEAN WIN = real>=690 AND momentum>ic AND flip>=220k AND noise>=50k (beats current on capture without whipsaw/real cost).")
