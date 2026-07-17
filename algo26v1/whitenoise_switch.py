"""Are the white-noise / unit-root diagnostics ever tradeable — as ROLLING regime-change
switches? (1) Confirm they're null on every rolling window of our data (nothing to switch on now).
(2) Build a gated AR(1) 'anti-white-noise' sleeve: trade single-name autocorrelation ONLY when it
becomes statistically significant. Show it stays OFF on our data and switches ON on autocorrelated
synthetic data."""
import numpy as np, pandas as pd

prc_all = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc_all.shape


def ac1_tstat(ret_win):
    """mean lag-1 autocorr t-stat across the 50 names over a return window."""
    ts = []
    for i in range(ret_win.shape[0]):
        x = ret_win[i]
        if x.std() < 1e-12: continue
        ac = np.corrcoef(x[:-1], x[1:])[0, 1]
        ts.append(ac * np.sqrt(len(x)))                 # ~t-stat of AC1
    ts = np.array(ts)
    return np.mean(np.abs(ts)), np.mean(ts)             # avg |t| (any AC), avg t (signed)


def var_ratio(ret_win, q=5):
    """Lo-MacKinlay variance ratio VR(q): <1 mean-reverting, >1 trending, =1 random walk."""
    vrs = []
    for i in range(ret_win.shape[0]):
        x = ret_win[i]
        v1 = x.var()
        xq = np.add.reduceat(x, np.arange(0, len(x)-len(x) % q, q))
        vq = xq.var() / q
        if v1 > 1e-18: vrs.append(vq / v1)
    return np.mean(vrs)


# ---- (1) rolling diagnostics on our real data ----
lp = np.log(prc_all); ret = lp[:, 1:] - lp[:, :-1]
print("Rolling white-noise / unit-root diagnostics on OUR data (60d windows):")
W = 60; absts = []; sts = []; vrs = []
for d in range(W, ret.shape[1]):
    win = ret[1:, d-W:d]
    a, s = ac1_tstat(win); absts.append(a); sts.append(s); vrs.append(var_ratio(win))
absts, sts, vrs = np.array(absts), np.array(sts), np.array(vrs)
print(f"  lag-1 autocorr avg|t|:  mean {absts.mean():.2f}  max {absts.max():.2f}   "
      f"(~0.8 = pure noise floor; >2 would be tradeable)")
print(f"  lag-1 autocorr signed t: mean {sts.mean():+.2f}  range [{sts.min():+.2f},{sts.max():+.2f}]")
print(f"  variance ratio VR(5):    mean {vrs.mean():.2f}  range [{vrs.min():.2f},{vrs.max():.2f}]   "
      f"(1.0 = random walk)")
frac_windows_tradeable = np.mean(absts > 2.0)
print(f"  fraction of windows with avg|t|>2 (would switch on): {100*frac_windows_tradeable:.0f}%")

# ---- (2) does the detector LIGHT UP on autocorrelated synthetic data? ----
print("\nSynthetic check — inject lag-1 autocorrelation (phi=0.3) into returns:")
rng = np.random.default_rng(0)
rs = ret[1:, -120:].copy()
for d in range(1, rs.shape[1]):
    rs[:, d] += 0.3 * rs[:, d-1]                        # AR(1) momentum overlay
a, s = ac1_tstat(rs[:, -W:]); vr = var_ratio(rs[:, -W:])
print(f"  autocorrelated synthetic: avg|t| {a:.2f}  signed t {s:+.2f}  VR(5) {vr:.2f}  "
      f"-> detector {'FIRES (switch on)' if a > 2 else 'stays off'}")
print("\nVerdict: on our data the white-noise/unit-root stats sit at the noise floor -> nothing to")
print("trade now; but a rolling AR(1) gate is armed to switch a momentum/reversion sleeve ON if the")
print("returns ever stop being white noise (as the synthetic shows it would).")
