"""detector.py — can a short-term detector time the lead-lag vs reversion weight?
(1) Is rolling lead-lag IC PERSISTENT (recent IC predicts next IC)? If not, no detector can work.
(2) Backtest an adaptive-blend book that raises reversion weight when recent lead-lag IC is weak,
    vs static SAFE(b.30) / QUAL(b.20). No look-ahead: trailing IC uses only realized past days."""
import numpy as np, pandas as pd
import stability as S
np = S.np; ridge_z = S.ridge_z; revz = S.revz; r_all = S.r_all; nDays = S.nDays; ENS = S.ENS
prc = S.prc; dlr = S.dlr; logp = S.logp; commRate = S.commRate; nInst = S.nInst

# ---- per-day realized ICs (forecast made at t, realized return = r_all[:, t-1]) ----
IC_LL = {}; IC_REV = {}
for t in range(96, nDays):                     # r_all[:, t-1] known for t<=nDays-1... range stops at nDays-1
    fll = np.mean([ridge_z(t, hl) for hl in ENS], 0)
    frev = revz(t, 10)
    fwd = r_all[1:, t - 1]; fwd = fwd - fwd.mean()
    if fwd.std() > 1e-12:
        IC_LL[t] = float(np.corrcoef(fll, fwd)[0, 1])
        IC_REV[t] = float(np.corrcoef(frev, fwd)[0, 1])
ts = np.array(sorted(IC_LL)); ll = np.array([IC_LL[t] for t in ts]); rev = np.array([IC_REV[t] for t in ts])
print(f"mean daily IC: lead-lag {ll.mean():.4f}   reversion {rev.mean():.4f}   (n={len(ll)} days)")

# ---- (1) persistence of rolling lead-lag IC ----
print("\n(1) PERSISTENCE — does trailing-W lead-lag IC predict the NEXT-W lead-lag IC?")
for W in (10, 20, 40, 60):
    roll = pd.Series(ll).rolling(W).mean().values
    a = roll[W - 1:-W]; b = roll[2 * W - 1:]         # trailing block vs next block
    m = ~(np.isnan(a) | np.isnan(b))
    c = np.corrcoef(a[m], b[m])[0, 1]
    # also: AR(1) of the W-rolling series
    x = roll[~np.isnan(roll)]; ar1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    print(f"   W={W:>2}:  corr(trailing IC, next-block IC) = {c:+.3f}   rolling-series AR(1) = {ar1:+.3f}")

# ---- (2) adaptive-blend backtest (no look-ahead) ----
def adaptive_blend(t, W, b_lo=0.15, b_hi=0.40, ic_ref=None):
    # trailing lead-lag IC over realized days strictly before t
    past = [IC_LL[s] for s in range(t - W, t) if s in IC_LL]
    if len(past) < max(5, W // 2): return 0.25
    m = np.mean(past)
    if ic_ref is None: ic_ref = 0.079
    # weak lead-lag (m < ref) -> raise blend toward b_hi; strong -> lower toward b_lo
    frac = np.clip((ic_ref - m) / ic_ref, -1.0, 1.0)          # +1 = IC collapsed, -1 = IC double
    return float(np.clip(0.25 + 0.20 * frac, b_lo, b_hi))

def run(Sd, Ed, blend_fn):
    cash = 0.0; cp = np.zeros(nInst); value = 0.0; comm = 0.0; pll = []
    for t in range(Sd, Ed + 1):
        cur = prc[:, t - 1]; pos = np.zeros(nInst)
        if t < Ed and t >= 96:
            blend = blend_fn(t)
            wz = S.wzsig(t, blend); pos[1:] = np.sign(wz) * (dlr[1:] / cur[1:])
            cap = dlr[0] / cur[0]; lpA = logp[0, :t]; mv = lpA[30:] - lpA[:-30]
            z = (mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12)
            pos[0] = float(np.clip(-np.clip(z, -3, 3) / 3.0 * (1_000_000 / cur[0]), -cap, cap))
            lim = (dlr / cur).astype(int); pos = np.clip(pos, -lim, lim).astype(int)
        else: pos = cp.copy()
        dp = pos - cp; cash -= cur.dot(dp) + comm; comm = np.sum(cur * np.abs(dp) * commRate); cp = pos.copy()
        pl = cash + cp.dot(cur) - value; value = cash + cp.dot(cur)
        if t > Sd: pll.append(pl)
    return S.stats(pll)[3]

print("\n(2) ADAPTIVE-blend detector vs static, per-window score:")
books = {"SAFE b.30": lambda t: 0.30, "QUAL b.20": lambda t: 0.20,
         "adaptive W20": lambda t: adaptive_blend(t, 20), "adaptive W40": lambda t: adaptive_blend(t, 40)}
for L, lo, step in ((500, 500, 40), (250, 396, 100)):
    ends = list(range(lo, nDays + 1, step))
    print(f"  -- {L}d windows --")
    res = {name: np.array([run(e - L, e, fn) for e in ends]) for name, fn in books.items()}
    print("   " + "".join(f"{n:>14}" for n in books))
    for i, e in enumerate(ends):
        print(f"   {e-L:>3}-{e:<3}" + "".join(f"{res[n][i]:14.1f}" for n in books))
    print(f"   {'mean':>7}  " + "".join(f"{res[n].mean():13.1f}" for n in books))
    print(f"   {'floor':>7} " + "".join(f"{res[n].min():13.1f}" for n in books))
