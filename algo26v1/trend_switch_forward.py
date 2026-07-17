"""Does the fade->follow trend switch survive on UNSEEN futures, or is it fitting a few trend
episodes in our one path? Forward-MC across the 3 worlds + report how often 'follow' fires."""
import numpy as np
from forward_mc import _ridge_fit, world_b, world_c, fwd_score, real, nInst
from dgp_simulator import DGP

TRIG = {"n": 0, "tot": 0}


def make(mode="fade", ac_thr=0.03, tstat_thr=1.5, count=False):
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
        seg = rA[-30:]; drift_t = seg.mean()/(seg.std()/np.sqrt(len(seg))+1e-12)
        ac = np.corrcoef(rA[-40:-1], rA[-39:])[0, 1]
        trending = (ac > ac_thr) and (abs(drift_t) > tstat_thr)
        if count:
            TRIG["tot"] += 1; TRIG["n"] += int(trending)
        if mode == "switch" and trending:
            sh = np.sign(mv[-1])*min(abs(z), 3)*dollars/prc[0, -1]     # follow
        else:
            sh = -np.clip(z, -3, 3)*dollars/prc[0, -1]                 # fade
        rev = float(np.clip(sh, -cap, cap))
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
print(f"{'mode':22}" + "".join(f"{w:>12}" for w in worlds))
for name, kw in [("fade always", dict(mode="fade")), ("switch->follow", dict(mode="switch"))]:
    row = f"{name:22}"
    for wname, panels in worlds.items():
        scs = np.array([fwd_score(lambda kw=kw: make(**kw), p, T+1) for p in panels])
        row += f"{np.median(scs):12.0f}"
    print(row)
# trigger frequency on the real path
TRIG["n"] = TRIG["tot"] = 0
fwd_score(lambda: make(mode="switch", count=True), np.array(real), real.shape[1]-250)
print(f"\n'follow' fired on {TRIG['n']}/{TRIG['tot']} days ({100*TRIG['n']/max(TRIG['tot'],1):.0f}%) over the real last-250")
