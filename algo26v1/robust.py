"""Rolling-window robustness: v4 (blend=0) vs combined (blend=0.2,rev_w=10)."""
import numpy as np
from combined_lab import make_getpos, prcAll, nInst, nt, commRate, dlrPosLimit, score


def pl_window(getPosition, startDay, endDay):
    """Score PnL over trading days (startDay, endDay]; needs full history up to each t."""
    cash = 0; curPos = np.zeros(nInst); totDV = 0; value = 0; comm = 0; pll = []
    for t in range(startDay, endDay + 1):
        prc = prcAll[:, :t]; cur = prc[:, -1]
        if t < endDay:
            npos = getPosition(prc)
            lim = (dlrPosLimit / cur).astype(int)
            npos = np.clip(npos, -lim, lim).astype(int)
        else:
            npos = np.array(curPos)
        d = npos - curPos
        cash -= cur.dot(d) + comm
        dv = cur * np.abs(d); totDV += dv.sum(); comm = (dv * commRate).sum()
        curPos = np.array(npos)
        pl = cash + curPos.dot(cur) - value
        value = cash + curPos.dot(cur)
        if t > startDay:
            pll.append(pl)
    pll = np.array(pll); mu, sd = pll.mean(), pll.std()
    return score(mu, sd)


configs = {
    "v4 (blend=0)": dict(half_life=2000, conv_z=0.2, blend=0.0, rev_w=10, contra_wz=60),
    "combined    ": dict(half_life=2000, conv_z=0.2, blend=0.2, rev_w=10, contra_wz=60),
}

# rolling 120-day folds ending at 260,320,380,440,500 (each fresh module = state-independent)
ends = [260, 320, 380, 440, 500]
W = 120
print(f"120-day rolling folds (each is an independent fresh run):")
print(f"{'end day':>8} " + " ".join(f"{k:>14}" for k in configs))
sums = {k: [] for k in configs}
for e in ends:
    row = f"{e:>8} "
    for k, kn in configs.items():
        s = pl_window(make_getpos(**kn), e - W, e)
        sums[k].append(s); row += f"{s:14.1f} "
    print(row)
print("-" * 40)
print(f"{'mean':>8} " + " ".join(f"{np.mean(sums[k]):14.1f}" for k in configs))
print(f"{'min ':>8} " + " ".join(f"{np.min(sums[k]):14.1f}" for k in configs))
