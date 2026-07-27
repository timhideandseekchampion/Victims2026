"""regime_validator.py — can a Markov-chain (Gaussian HMM) or k-means regime label VALIDATE a
signal switch? The honest OOS test: fit the regime model on a TRAIN window only, assign each TEST
day a label CAUSALLY (features + filter use data <= t), then ask whether the label separates the
FUTURE realized IC of champion (lead-lag+reversion) vs momJT (momentum) -- i.e. does the regime
label tell us, out-of-sample, which signal is about to work?

A useful validator => one regime has high future champ IC, another high future momJT IC, OOS.
Noise => all regimes have the same future IC (labels track variance, not a real regime).

Run on (A) real 750-day data (expected: reversion everywhere -> no separation) and
(B) a SYNTHETIC injected momentum regime (does the method detect a regime that genuinely exists,
and how fast vs the IC gate?). numpy-only (no sklearn/scipy/hmmlearn in this env).
"""
import numpy as np, pandas as pd
import SAFE_rotate as R

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nDays = prc.shape

# ---------------- regime features (all causal: use prc[:, :t] only) ----------------
FW = 40   # feature lookback window
def features_for(P):
    """feature matrix, one row per day t in [FW+2, T]. Uses only data up to t (no look-ahead)."""
    logp = np.log(P); r = logp[:, 1:] - logp[:, :-1]; T = P.shape[1]
    rows = {}
    for t in range(FW + 2, T + 1):
        rr = r[:, :t - 1]                                   # returns known at decision t (cols 0..t-2)
        win = rr[:, -FW:]
        names = win[1:]                                     # 50 x FW
        vol = names.std(1).mean()                           # avg per-name vol
        disp = names[:, -1].std()                           # cross-sectional dispersion today
        # cross-sectional lag-1 autocorr (momentum>0 vs reversion<0)
        acs = []
        for s in range(1, FW):
            a = names[:, s - 1] - names[:, s - 1].mean(); b = names[:, s] - names[:, s].mean()
            d = np.sqrt((a @ a) * (b @ b))
            if d > 1e-12: acs.append(a @ b / d)
        xsac = np.mean(acs) if acs else 0.0
        # index 30-day trend strength |z|
        lpA = logp[0, :t]
        idxtr = 0.0
        if len(lpA) > 30 + 60:
            mv = lpA[30:] - lpA[:-30]; idxtr = abs((mv[-1] - mv[-60:].mean()) / (mv[-60:].std() + 1e-12))
        rows[t] = np.array([vol, disp, xsac, idxtr])
    return rows

# ---------------- k-means & Gaussian HMM (numpy) ----------------
def kmeans(X, K, seed=0, iters=50):
    rng = np.random.default_rng(seed); C = X[rng.choice(len(X), K, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2); lab = d.argmin(1)
        newC = np.array([X[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
        if np.allclose(newC, C): break
        C = newC
    return lab, C

def _emis_logB(X, mu, var):
    n, d = X.shape; K = len(mu); logB = np.zeros((n, K))
    for k in range(K):
        logB[:, k] = -0.5 * (((X - mu[k]) ** 2 / var[k]).sum(1) + np.log(2 * np.pi * var[k]).sum())
    return logB

def ghmm_fit(X, K, iters=40, seed=0):
    """2/3-state diagonal Gaussian HMM via scaled Baum-Welch. Returns (mu,var,A,pi)."""
    n, d = X.shape; lab, _ = kmeans(X, K, seed)
    mu = np.array([X[lab == k].mean(0) if (lab == k).any() else X.mean(0) for k in range(K)])
    var = np.array([X[lab == k].var(0) + 1e-2 if (lab == k).any() else X.var(0) + 1e-2 for k in range(K)])
    A = np.full((K, K), 1.0 / K); pi = np.full(K, 1.0 / K)
    for _ in range(iters):
        logB = _emis_logB(X, mu, var); B = np.exp(logB - logB.max(1, keepdims=True))
        alpha = np.zeros((n, K)); c = np.zeros(n)
        alpha[0] = pi * B[0]; c[0] = alpha[0].sum() + 1e-300; alpha[0] /= c[0]
        for t in range(1, n):
            alpha[t] = (alpha[t - 1] @ A) * B[t]; c[t] = alpha[t].sum() + 1e-300; alpha[t] /= c[t]
        beta = np.zeros((n, K)); beta[-1] = 1.0
        for t in range(n - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta; gamma /= gamma.sum(1, keepdims=True) + 1e-300
        xi = np.zeros((K, K))
        for t in range(n - 1):
            mmx = (alpha[t][:, None] * A) * (B[t + 1] * beta[t + 1])[None, :]; xi += mmx / (mmx.sum() + 1e-300)
        pi = gamma[0]; A = xi / (xi.sum(1, keepdims=True) + 1e-300)
        for k in range(K):
            w = gamma[:, k]; sw = w.sum() + 1e-300
            mu[k] = (w[:, None] * X).sum(0) / sw
            var[k] = (w[:, None] * (X - mu[k]) ** 2).sum(0) / sw + 1e-2
    return mu, var, A, pi

def ghmm_filter(X, mu, var, A, pi):
    """causal filtered state posterior (alpha[t] depends only on obs <= t). Returns argmax label/day."""
    n, K = X.shape[0], len(mu); logB = _emis_logB(X, mu, var); B = np.exp(logB - logB.max(1, keepdims=True))
    alpha = np.zeros((n, K)); alpha[0] = pi * B[0]; alpha[0] /= alpha[0].sum() + 1e-300
    for t in range(1, n):
        alpha[t] = (alpha[t - 1] @ A) * B[t]; alpha[t] /= alpha[t].sum() + 1e-300
    return alpha.argmax(1)

# ---------------- target: future realized IC of champ vs momJT ----------------
def ic_series(P, sig):
    R._SIG.clear(); R._RET.clear(); R._ICD.clear(); R._ensure_cache(P)
    T = P.shape[1]; return {n: R._ic1(sig, n) for n in range(R.WARMUP, T - 1)}

def run_case(P, label, H=20):
    print(f"\n===== {label} =====")
    feats = features_for(P); days = np.array(sorted(feats))
    F = np.array([feats[t] for t in days])
    icch = ic_series(P, "champ"); icmj = ic_series(P, "momJT")
    # forward IC diff over [t, t+H): momJT edge minus champ edge (what a validator must predict)
    def fwd(t):
        cs = [icmj[n] - icch[n] for n in range(t, t + H) if n in icmj and n in icch]
        return np.mean(cs) if cs else np.nan
    y = np.array([fwd(t) for t in days])
    ok = ~np.isnan(y); days, F, y = days[ok], F[ok], y[ok]
    n = len(days); mid = n // 2
    tr, te = slice(0, mid), slice(mid, n)
    mu_tr, sd_tr = F[tr].mean(0), F[tr].std(0) + 1e-9
    Xtr = (F[tr] - mu_tr) / sd_tr; Xall = (F - mu_tr) / sd_tr
    print(f"  days {days[0]}-{days[-1]}  (train {days[0]}-{days[mid-1]}, test {days[mid]}-{days[-1]})  H={H}")
    print(f"  {'method':<14}{'regime':>7}{'n_test':>8}{'mean fwd(momJT-champ)IC':>26}")
    for name, K, labeler in [
        ("kmeans K2", 2, "km"), ("kmeans K3", 3, "km"),
        ("HMM 2-state", 2, "hmm"), ("HMM 3-state", 3, "hmm")]:
        if labeler == "km":
            _, C = kmeans(Xtr, K, seed=0)
            lab = (((Xall[:, None, :] - C[None]) ** 2).sum(2)).argmin(1)   # nearest train center (causal)
        else:
            mu, var, Am, pi = ghmm_fit(Xtr, K)
            lab = ghmm_filter(Xall, mu, var, Am, pi)                       # causal filtered state
        labte = lab[te]; yte = y[te]
        spreads = []
        for k in range(K):
            msk = labte == k; nk = int(msk.sum())
            mk = yte[msk].mean() if nk else np.nan
            spreads.append(mk if nk else np.nan)
            print(f"  {name if k==0 else '':<14}{k:>7}{nk:>8}{mk:>26.4f}")
        valid = [s for s in spreads if not np.isnan(s)]
        sep = (max(valid) - min(valid)) if len(valid) > 1 else 0.0
        print(f"  {'':<14}{'SEP':>7}{'':>8}{sep:>26.4f}   (OOS regime separation of future IC; ~0 = no validation value)")
    # baseline: does trailing champ IC bin predict fwd IC diff OOS? (the non-persistence control)
    tic = np.array([icch.get(t, np.nan) for t in days])
    hi = (tic[te] > np.nanmedian(tic[tr])); yte = y[te]
    print(f"  {'trailIC hi/lo':<14}{'hi':>7}{int(hi.sum()):>8}{yte[hi].mean():>26.4f}")
    print(f"  {'(baseline)':<14}{'lo':>7}{int((~hi).sum()):>8}{yte[~hi].mean():>26.4f}")

def make_mom(D=400, T_ext=None, mom=0.7, seed=1):
    rng = np.random.default_rng(seed); logp = np.log(prc[:, :D + 1]).copy()
    vol = np.diff(np.log(prc[1:]), axis=1).std(); names = logp[1:].copy()
    for _ in range((nDays - D)):
        trail = names[:, -1] - names[:, -5]; tc = trail - trail.mean()
        drift = mom * (tc / (tc.std() + 1e-9)) * vol; drift -= drift.mean()
        noise = rng.normal(0, vol, 50); noise -= noise.mean()
        names = np.concatenate([names, (names[:, -1] + drift + noise)[:, None]], axis=1)
    full = np.exp(np.concatenate([names.mean(0, keepdims=True), names], axis=0))
    full[:, :D + 1] = prc[:, :D + 1]; return full

if __name__ == "__main__":
    run_case(prc, "REAL 750-day data (expect: no regime -> ~0 separation)")
    run_case(make_mom(), "SYNTHETIC momentum from day 400 (regime at train/test boundary — see regime_detect.py)")
