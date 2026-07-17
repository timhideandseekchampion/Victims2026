"""Does the ALGO-autocorr gate survive on UNSEEN futures, or is the +30 @250 just recent-half
fitting? Score 'always-on' vs 'OFF-when-trending' across the 3 forward-MC worlds."""
import numpy as np
from forward_mc import _ridge_fit, world_b, world_c, fwd_score, real, nInst
from dgp_simulator import DGP


def make(gate=None):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = _ridge_fit(ret[:, :-1].T, ret[1:, 1:].T, hl=500); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]
        lpA = np.log(prc[0]); rA = lpA[1:]-lpA[:-1]; mv = lpA[30:]-lpA[:-30]
        z = (mv[-1]-mv[-60:].mean())/(mv[-60:].std()+1e-12); dollars = 200_000
        if gate is not None:
            ac = np.corrcoef(rA[-40:-1], rA[-39:])[0, 1]
            if ac > gate: dollars = 0.0
        rev = float(np.clip(-np.clip(z, -3, 3)*dollars/prc[0, -1], -cap, cap))
        rA0 = ret[0]; rAc = rA0-rA0.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    return gp


NSEED, FUT = 14, 220
dgpA = DGP.fit(real); dgpA.signal_scale = 0.3; T = real.shape[1]
worlds = {"A: VAR": [dgpA.extend(real, FUT, s) for s in range(NSEED)],
          "B: pairs": [world_b(real, FUT, s) for s in range(NSEED)],
          "C: shift": [world_c(dgpA, real, FUT, s) for s in range(NSEED)]}
print(f"Forward MC (median of {NSEED} unseen futures):\n")
print(f"{'rule':28}" + "".join(f"{w:>12}" for w in worlds))
for name, kw in [("always on", dict()), ("OFF when trending (ac>0.05)", dict(gate=0.05))]:
    row = f"{name:28}"
    for wname, panels in worlds.items():
        scs = np.array([fwd_score(lambda kw=kw: make(**kw), p, T+1) for p in panels])
        row += f"{np.median(scs):12.0f}"
    print(row)
