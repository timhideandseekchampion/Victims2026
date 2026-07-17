"""Test the AENO~NWIG cointegration pair overlay layered onto the v4 book."""
import numpy as np
from combined_lab import (_ewls_ridge_fit, prcAll, nInst, nt, commRate,
                          dlrPosLimit, score)

AENO, NWIG = 1, 20   # 0-indexed columns


def make_getpos(pair_dollars=0.0, zw=60, blend=0.0, rev_w=10):
    cache = {"fit_t": None, "model": None}

    def gp(prcSoFar):
        ni, t = prcSoFar.shape
        pos = np.zeros(ni)
        if t < 60:
            return pos
        lp = np.log(prcSoFar)
        ret = lp[:, 1:] - lp[:, :-1]
        if cache["fit_t"] != t:
            cache["model"] = _ewls_ridge_fit(ret[:, :-1].T, ret[1:, 1:].T, 2000, 0.1)
            cache["fit_t"] = t
        B, mx, my = cache["model"]
        pred = my + (ret[:, -1] - mx) @ B
        w = pred - pred.mean()
        wz = w / (np.std(w) + 1e-12)
        if blend > 0:
            r = ret[1:, -rev_w:].sum(1); r -= r.mean()
            wz = (1 - blend) * wz + blend * (-r / (np.std(r) + 1e-12))
        sized = np.sign(wz) * (10_000 / prcSoFar[1:, -1])
        keep = np.abs(wz) >= 0.2 * (np.std(wz) + 1e-12)
        pos[1:] = np.where(keep, sized, 0.0)
        # ALGO contrarian
        cap = 100_000 / prcSoFar[0, -1]; rev_sh = 0.0
        if t > 92:
            lpA = np.log(prcSoFar[0]); move = lpA[30:] - lpA[:-30]
            z = (move[-1] - move[-60:].mean()) / (move[-60:].std() + 1e-12)
            rev_sh = -float(np.clip(z, -3, 3)) * 200_000 / prcSoFar[0, -1]
        rev_sh = float(np.clip(rev_sh, -cap, cap))
        # AENO~NWIG pair overlay (fade the log-spread) BEFORE hedge/cap accounting
        if pair_dollars > 0 and t > zw + 2:
            beta = np.polyfit(lp[NWIG, -zw:], lp[AENO, -zw:], 1)[0]
            spr = lp[AENO] - beta * lp[NWIG]
            z = (spr[-1] - spr[-zw:].mean()) / (spr[-zw:].std() + 1e-12)
            u = -float(np.clip(z, -2, 2)) / 2.0
            pos[AENO] += u * pair_dollars / prcSoFar[AENO, -1]
            pos[NWIG] -= u * beta * pair_dollars / prcSoFar[NWIG, -1]
        # hedge
        rA = ret[0]; rAc = rA - rA.mean(); denom = rAc @ rAc + 1e-12
        betas = ((ret[1:] - ret[1:].mean(1, keepdims=True)) @ rAc) / denom
        net_beta = (pos[1:] * prcSoFar[1:, -1]) @ betas
        hedge_sh = -net_beta / prcSoFar[0, -1]
        room = max(cap - abs(rev_sh), 0.0)
        pos[0] = rev_sh + float(np.clip(hedge_sh, -room, room))
        return pos.astype(int)
    return gp


def calcPL(gp, ndays):
    cash = 0; curPos = np.zeros(nInst); value = 0; comm = 0; pll = []
    startDay = nt - ndays
    for t in range(startDay, nt + 1):
        prc = prcAll[:, :t]; cur = prc[:, -1]
        if t < nt:
            lim = (dlrPosLimit / cur).astype(int)
            npos = np.clip(gp(prc), -lim, lim).astype(int)
        else:
            npos = np.array(curPos)
        d = npos - curPos; cash -= cur.dot(d) + comm
        dv = cur * np.abs(d); comm = (dv * commRate).sum()
        curPos = np.array(npos); pl = cash + curPos.dot(cur) - value
        value = cash + curPos.dot(cur)
        if t > startDay: pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    return score(mu, sd), (np.sqrt(250) * mu / sd if sd > 0 else 0)


print(f"{'config':28} {'S@250':>8} {'S@440':>8}")
for name, kw in [
    ("v4 (no pair)", dict()),
    ("+ pair $3k", dict(pair_dollars=3_000)),
    ("+ pair $6k", dict(pair_dollars=6_000)),
    ("+ pair $10k", dict(pair_dollars=10_000)),
    ("+ pair $10k zw=40", dict(pair_dollars=10_000, zw=40)),
]:
    s2, _ = calcPL(make_getpos(**kw), 250)
    s4, _ = calcPL(make_getpos(**kw), 440)
    print(f"{name:28} {s2:8.1f} {s4:8.1f}")
