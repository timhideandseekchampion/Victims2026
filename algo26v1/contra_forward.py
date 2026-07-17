"""Do the ALGO-contra K/WZ cells hold up on UNSEEN futures, or are they just fitting our
recent half? Score each cell on synthetic futures of 3 worlds (VAR / pairs / structure-shift).
A durable edge shows a consistent + contribution across worlds; an H2-fit shows noise."""
import numpy as np
from forward_mc import _ridge_fit, world_b, world_c, fwd_score, real, nInst
from dgp_simulator import DGP


def make_ridge(K=30, WZ=60, contra=200_000, hl=500):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = _ridge_fit(ret[:, :-1].T, ret[1:, 1:].T, hl=hl); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]; rev = 0.0
        if contra > 0 and t > K+WZ+2:
            lpA = np.log(prc[0]); mv = lpA[K:]-lpA[:-K]; z = (mv[-1]-mv[-WZ:].mean())/(mv[-WZ:].std()+1e-12)
            rev = float(np.clip(-np.clip(z, -3, 3)*contra/prc[0, -1], -cap, cap))
        rA = ret[0]; rAc = rA-rA.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    return gp


NSEED, FUT = 14, 220
dgpA = DGP.fit(real); dgpA.signal_scale = 0.3
T = real.shape[1]
worlds = {
    "A: VAR": [dgpA.extend(real, FUT, s) for s in range(NSEED)],
    "B: pairs": [world_b(real, FUT, s) for s in range(NSEED)],
    "C: shift": [world_c(dgpA, real, FUT, s) for s in range(NSEED)],
}
cells = {
    "ALGO OFF (idio only)":  dict(contra=0),
    "K30/WZ60 (current)":    dict(K=30, WZ=60),
    "K30/WZ20 (@250-max)":   dict(K=30, WZ=20),
    "K40/WZ40 (balanced)":   dict(K=40, WZ=40),
    "K5/WZ40 (short)":       dict(K=5,  WZ=40),
}
print(f"Forward MC across worlds: median score on {NSEED} unseen futures each (FUT={FUT}).\n")
print(f"{'cell':24}" + "".join(f"{w:>12}" for w in worlds))
base = {}
for name, kw in cells.items():
    row = f"{name:24}"
    for wname, panels in worlds.items():
        scs = np.array([fwd_score(lambda kw=kw: make_ridge(**kw), p, T+1) for p in panels])
        m = np.median(scs); row += f"{m:12.0f}"
        base.setdefault(wname, {})[name] = m
    print(row)
print("\n=== ALGO-leg CONTRIBUTION per world (cell - 'ALGO OFF'), median ===")
for wname in worlds:
    off = base[wname]["ALGO OFF (idio only)"]
    deltas = {n: base[wname][n]-off for n in cells if n != "ALGO OFF (idio only)"}
    print(f"  {wname:10} " + "  ".join(f"{n.split()[0]}:{d:+5.0f}" for n, d in deltas.items()))
