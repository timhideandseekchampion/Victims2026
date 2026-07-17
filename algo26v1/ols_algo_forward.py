"""Is the OLS-adaptive ALGO leg the best of both — matches fade in reverting regimes, but
auto-follows (no bleed) if the regime turns trending? Forward-MC fade vs ols across 3 worlds."""
import numpy as np
from forward_mc import _ridge_fit, world_b, world_c, fwd_score, real, nInst
from dgp_simulator import DGP
from ols_algo import algo_signal, K, WZ, DOLLARS


def make(mode="fade"):
    c = {"t": None, "m": None}
    def gp(prc):
        ni, t = prc.shape; pos = np.zeros(ni)
        if t < 95: return pos.astype(int)
        lp = np.log(prc); ret = lp[:, 1:]-lp[:, :-1]
        if c["t"] != t: c["m"] = _ridge_fit(ret[:, :-1].T, ret[1:, 1:].T, hl=500); c["t"] = t
        B, mx, my = c["m"]; pred = my+(ret[:, -1]-mx)@B; w = pred-pred.mean()
        s = np.sign(w)*(10000/prc[1:, -1]); pos[1:] = np.where(np.abs(w) >= 0.2*(np.std(w)+1e-12), s, 0.0)
        cap = 100000/prc[0, -1]
        sig = algo_signal(np.log(prc[0]), mode)
        rev = float(np.clip(sig*DOLLARS/prc[0, -1], -cap, cap))
        rA0 = ret[0]; rAc = rA0-rA0.mean(); den = rAc@rAc+1e-12
        betas = ((ret[1:]-ret[1:].mean(1, keepdims=True))@rAc)/den
        net = (pos[1:]*prc[1:, -1])@betas; room = max(cap-abs(rev), 0.0)
        pos[0] = rev+float(np.clip(-net/prc[0, -1], -room, room)); return pos.astype(int)
    return gp


NSEED, FUT = 12, 200
dgpA = DGP.fit(real); dgpA.signal_scale = 0.3; T = real.shape[1]
worlds = {"A: VAR": [dgpA.extend(real, FUT, s) for s in range(NSEED)],
          "B: pairs": [world_b(real, FUT, s) for s in range(NSEED)],
          "C: shift": [world_c(dgpA, real, FUT, s) for s in range(NSEED)]}
print(f"Forward MC (median of {NSEED} unseen futures):\n")
print(f"{'mode':16}" + "".join(f"{w:>12}" for w in worlds))
for name, mode in [("fade", "fade"), ("ols adaptive", "ols")]:
    row = f"{name:16}"
    for wname, panels in worlds.items():
        scs = np.array([fwd_score(lambda m=mode: make(m), p, T+1) for p in panels])
        row += f"{np.median(scs):12.0f}"
    print(row)
print("DONE")
