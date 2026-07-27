"""regime_detect.py — the fair 'can it detect a REAL regime?' test the validator run botched
(regime change was at the train/test boundary -> homogeneous test set). Here the momentum regime
starts MID-TEST, so the test window contains both pre- and post-change days and we can measure:
  - does the injected regime even SHOW UP in the price features? (cross-sectional autocorr flips)
  - does a causal k-means/HMM label FLIP at the change, and with what LAG vs the ~53d IC gate?
Deep point being checked: these methods can only detect a regime that manifests in observable
features (vol / dispersion / autocorrelation). A reversion->momentum flip does (autocorr sign);
a pure predictability-death would not.
"""
import numpy as np, pandas as pd
import SAFE_rotate as R
from regime_validator import features_for, kmeans, ghmm_fit, ghmm_filter

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape

def make_mom(D, mom=0.7, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc[:, :D + 1]).copy()
    vol = np.diff(np.log(prc[1:]), axis=1).std(); names = logp[1:].copy()
    for _ in range(nDays - D):
        trail = names[:, -1] - names[:, -5]; tc = trail - trail.mean()
        drift = mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :D + 1] = prc[:, :D + 1]; return full

D = 550                                         # regime change MID-TEST
if __name__ != "__main__":
    import sys; sys.exit(0)
full = make_mom(D)
feats = features_for(full); days = np.array(sorted(feats)); F = np.array([feats[t] for t in days])

# does the regime show up in the features? cross-sectional autocorr (col 2) pre vs post D
pre = F[(days > D - 120) & (days <= D), 2]; post = F[(days > D) & (days <= D + 120), 2]
print(f"regime change injected at day D={D}  (momentum: winners keep winning)")
print(f"cross-sectional lag-1 autocorr feature:  pre-D mean {pre.mean():+.3f}   post-D mean {post.mean():+.3f}"
      f"   (flip = the regime is VISIBLE in price features)")

# causal detection: fit on train (< 450), label all days causally, measure lag after D
mid = np.searchsorted(days, 450)
mu_tr, sd_tr = F[:mid].mean(0), F[:mid].std(0) + 1e-9
Xtr = (F[:mid] - mu_tr) / sd_tr; Xall = (F - mu_tr) / sd_tr

def detect_lag(labels):
    lab = pd.Series(labels, index=days)
    post_majority = lab[(days > D) & (days <= D + 120)].mode()
    if len(post_majority) == 0: return None, None
    pm = post_majority.iloc[0]
    pre_majority = lab[(days > D - 120) & (days <= D)].mode().iloc[0]
    if pm == pre_majority: return "no-flip", pm
    # first day > D where label == pm and stays pm for 10 consecutive labelled days
    dtest = days[days > D]
    for t in dtest:
        w = lab[(days >= t) & (days < t + 10)]
        if len(w) >= 8 and (w == pm).all(): return int(t - D), pm
    return None, pm

print(f"\n{'method':<14}{'detection lag (days after D)':>30}")
_, C = kmeans(Xtr, 2, seed=0)
labk = (((Xall[:, None, :] - C[None]) ** 2).sum(2)).argmin(1)
lagk, _ = detect_lag(labk); print(f"{'kmeans K2':<14}{str(lagk):>30}")
mu, var, Am, pi = ghmm_fit(Xtr, 2)
labh = ghmm_filter(Xall, mu, var, Am, pi)
lagh, _ = detect_lag(labh); print(f"{'HMM 2-state':<14}{str(lagh):>30}")
print(f"{'IC gate (ref)':<14}{'~53 (conservative) / ~36 (balanced)':>30}")
print("\n(lag = trading days after the regime onset before the label flips and holds; compare to the IC gate.")
print(" 'no-flip' = the method never assigned the momentum period a distinct regime.)")
