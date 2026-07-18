"""oos_detector.py — is the detector edge REAL or overfit? Two honest arbiters:
(A) DISTRIBUTION of excess-over-correct-control across all 156 variants. If timing has signal,
    most variants beat control (median > 0). If noise, median ~0 and 'winners' are the lucky tail.
(B) OUT-OF-SAMPLE: pick the best variant on EARLY windows, test on LATE (disjoint) windows.
All controls are per-window static-at-the-detector's-own-mean-blend (the correct control)."""
import numpy as np
import stability as S
ridge_z = S.ridge_z; revz = S.revz; r_all = S.r_all; logp = S.logp; nDays = S.nDays; ENS = S.ENS
prc = S.prc; dlr = S.dlr; commRate = S.commRate; nInst = S.nInst

IC_LL = {}; IC_REV = {}
for t in range(96, nDays):
    fll = np.mean([ridge_z(t, hl) for hl in ENS], 0); frev = revz(t, 10)
    fwd = r_all[1:, t - 1]; fwd = fwd - fwd.mean()
    if fwd.std() > 1e-12:
        IC_LL[t] = float(np.corrcoef(fll, fwd)[0, 1]); IC_REV[t] = float(np.corrcoef(frev, fwd)[0, 1])

def run(Sd, Ed, blend_fn):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            wz = S.wzsig(t, blend_fn(t)); pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]; lpA = logp[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            pos[0] = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else: pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm; comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    return S.stats(pll)[3]

def trailing(ic, t, W):
    v = [ic[s] for s in range(t - W, t) if s in ic]
    return np.mean(v) if len(v) >= max(5, W // 2) else None
def mk_absIC(W, slope, ref, blo, bhi):
    def fn(t):
        m = trailing(IC_LL, t, W)
        if m is None: return 0.25
        return float(np.clip(0.25 + slope * np.clip((ref - m) / ref, -1, 1), blo, bhi))
    return fn
def mk_relIC(W, blo, bhi):
    def fn(t):
        a = trailing(IC_LL, t, W); b = trailing(IC_REV, t, W)
        if a is None or b is None: return 0.25
        a = max(a, 0.0); b = max(b, 0.0)
        return 0.25 if a + b < 1e-9 else float(np.clip(b / (a + b), blo, bhi))
    return fn
variants = []
for W in (10, 15, 20, 30, 40, 60):
    for slope in (0.10, 0.20, 0.30, 0.40):
        for ref in (0.05, 0.064, 0.079):
            for bb in ((0.15, 0.40), (0.10, 0.50)):
                variants.append((f"absIC W{W} s{slope} r{ref} {bb}", mk_absIC(W, slope, ref, *bb)))
for W in (10, 15, 20, 30, 40, 60):
    for bb in ((0.15, 0.40), (0.10, 0.50)):
        variants.append((f"relIC W{W} {bb}", mk_relIC(W, *bb)))

# 250d windows (more of them) split train (early) / test (late, as disjoint as 750 days allow)
ALL = [(e - 250, e) for e in range(346, nDays + 1, 20)]
TRAIN = [w for w in ALL if w[1] <= 520]      # ends 346..500
TEST = [w for w in ALL if w[1] >= 660]       # ends 660..746 (earliest test win 410-660 vs latest train 250-500: minimal overlap)
BG = np.round(np.arange(0.05, 0.501, 0.01), 2)
staticW = {w: {b: run(w[0], w[1], (lambda bb: (lambda t: bb))(b)) for b in BG} for w in set(TRAIN) | set(TEST)}
def ctrl(w, fn):
    mb = np.mean([fn(t) for t in range(max(w[0], 96), w[1])])
    return staticW[w][BG[np.argmin(np.abs(BG - mb))]]

def excess_on(wins, fn):
    a = np.array([run(w[0], w[1], fn) for w in wins]); c = np.array([ctrl(w, fn) for w in wins])
    return a.mean() - c.mean(), a.min() - c.min()

# (A) distribution across all variants on ALL split windows
allw = TRAIN + TEST
exm = []
for nm, fn in variants:
    em, ef = excess_on(allw, fn); exm.append(em)
exm = np.array(exm)
print(f"(A) excess_MEAN over correct control across {len(variants)} variants (all windows):")
print(f"    median {np.median(exm):+.1f}   mean {exm.mean():+.1f}   p10 {np.percentile(exm,10):+.1f}   p90 {np.percentile(exm,90):+.1f}   frac>0 {100*np.mean(exm>0):.0f}%")

# (B) OOS: pick best on TRAIN, report on TEST
tr = [(nm, fn, excess_on(TRAIN, fn)) for nm, fn in variants]
tr.sort(key=lambda x: -(x[2][0] + x[2][1]))
print(f"\n(B) OUT-OF-SAMPLE (train ends<=500, test ends>=660):")
print(f"    {'variant':<30}{'TRAIN dMean':>12}{'TRAIN dFloor':>13}{'TEST dMean':>12}{'TEST dFloor':>13}")
for nm, fn, (trm, trf) in tr[:5]:
    tem, tef = excess_on(TEST, fn)
    print(f"    {nm:<30}{trm:>12.1f}{trf:>13.1f}{tem:>12.1f}{tef:>13.1f}")
