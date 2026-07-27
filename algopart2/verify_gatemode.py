"""verify_gatemode.py — GATE_MODE in the FULL book (not the proxy). Checks:
 real 500-750: every mode must stay ~694 (inert -> no false switching on seen data)
 momentum regime: pnl/sharpe should capture MORE than ic (faster switch = more days on momentum)
 flip/noise: whipsaw / degradation bounded
Reports total PnL + #rotation days per mode per regime.
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

def ev(P, S, E, mode):
    R.GATE_MODE = mode; clearR()
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []; rot = 0; n = 0
    for t in range(S, E + 1):
        cur = P[:, t - 1]
        newPos = R.getMyPosition(P[:, :t]) if t < E else cp
        if t < E and t >= R.WARMUP + R.ROT_W + R.ROT_P:
            n += 1
            if mode == "softblend":                     # tilt-day: blend's signs differ from pure champion
                if (np.sign(R._blend_wz(t)) != np.sign(R._SIG[t]["champ"])).any(): rot += 1
            elif R._choose(t) != "champ": rot += 1
        dP = newPos - cp; cash -= cur.dot(dP) + comm; comm = np.sum(cur * np.abs(dP) * commRate); cp = newPos
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > S: pll.append(pl)
    p = np.array(pll); mu, sd = p.mean(), p.std()
    sc = mu * (np.sqrt(250) * mu / sd) ** 2 / ((np.sqrt(250) * mu / sd) ** 2 + 1) if mu > 0 else mu
    return p.sum(), sc, rot, n

MODES = ["ic", "pnl", "sharpe", "softblend"]
print("REAL 500-750 (must stay ~694, rot days 0):")
for m in MODES:
    tot, sc, rot, n = ev(prc, 500, 750, m)
    print(f"  {m:<7} score={sc:>6.0f}   rot-days {rot}/{n}")

for kind in ("momentum", "flip", "noise"):
    full = make_ext(kind)
    print(f"\n{kind.upper()} regime 750-900 (total PnL / rotation days):")
    for m in MODES:
        tot, sc, rot, n = ev(full, nDays, nDays + 150, m)
        print(f"  {m:<7} total={tot:>9.0f}   rot-days {rot}/{n}")
R.GATE_MODE = "ic"    # leave the file default unchanged until we decide
