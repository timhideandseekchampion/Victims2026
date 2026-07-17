"""Full diagnostic battery: run every edge test on the EARLY window (0-500) vs the RECENT window
(500-750) to detect structure that EMERGED recently — a new edge we could switch on. Anything
notably stronger/significant in 500-750 (and absent early) is a candidate a top team may exploit."""
import numpy as np, pandas as pd
from scipy import stats

prc = pd.read_csv("prices.txt", sep=r"\s+", header=0).values.T
nInst, nt = prc.shape
lp = np.log(prc); RET = lp[:, 1:] - lp[:, :-1]


def win_ret(a, b):                                          # returns for days (a,b]
    return RET[:, a:b]


def diagnostics(R):
    """R = (51, T) returns for a window. Return a dict of edge diagnostics."""
    A = R[1:]; algo = R[0]; T = A.shape[1]; d = {}
    # 1. single-name lag-1 autocorr (white noise / momentum)
    ac1 = np.array([np.corrcoef(A[i, :-1], A[i, 1:])[0, 1] for i in range(50) if A[i].std() > 0])
    d["ac1_mean_t"] = ac1.mean() * np.sqrt(T)               # signed: + momentum, - reversion
    d["ac1_absmean_t"] = np.mean(np.abs(ac1)) * np.sqrt(T)
    # 2. variance ratio VR(5)
    vrs = []
    for i in range(50):
        x = A[i]; v1 = x.var()
        xq = np.add.reduceat(x, np.arange(0, len(x)-len(x) % 5, 5)); vq = xq.var()/5
        if v1 > 1e-18: vrs.append(vq/v1)
    d["VR5"] = np.mean(vrs)
    # 3-5. cross-sectional signal ICs (reversion horizons, momentum) predicting next-day return
    def xs_ic(hfn):
        ics = []
        for t in range(25, T-1):
            sig = hfn(A, t)
            if sig is None: continue
            sig = sig - sig.mean(); fwd = A[:, t+1]
            if sig.std() > 0 and fwd.std() > 0: ics.append(np.corrcoef(sig, fwd)[0, 1])
        return np.mean(ics), (np.mean(ics)/(np.std(ics)/np.sqrt(len(ics))+1e-12))
    d["xs_rev1_IC"], d["xs_rev1_t"] = xs_ic(lambda A, t: -A[:, t])
    d["xs_rev5_IC"], d["xs_rev5_t"] = xs_ic(lambda A, t: -A[:, t-4:t+1].sum(1))
    d["xs_rev20_IC"], d["xs_rev20_t"] = xs_ic(lambda A, t: -A[:, t-19:t+1].sum(1))
    d["xs_mom5_IC"], d["xs_mom5_t"] = xs_ic(lambda A, t: A[:, t-4:t+1].sum(1))
    # 6. ALGO (market factor) trend/autocorr
    d["ALGO_ac1_t"] = np.corrcoef(algo[:-1], algo[1:])[0, 1] * np.sqrt(T)
    d["ALGO_drift_t"] = algo.mean()/(algo.std()/np.sqrt(T)+1e-12)
    # 7. vol clustering (ARCH: autocorr of squared returns)
    sq = A**2; archs = [np.corrcoef(sq[i, :-1], sq[i, 1:])[0, 1] for i in range(50) if sq[i].std() > 0]
    d["ARCH_t"] = np.mean(archs) * np.sqrt(T)
    # 8. tails
    flat = ((A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True)+1e-12)).ravel()
    d["excess_kurt"] = stats.kurtosis(flat)
    d["skew"] = stats.skew(flat)
    # 9. factor concentration
    C = np.corrcoef(A); ev = np.linalg.eigvalsh(C)[::-1]
    d["factor1_share"] = ev[0]/ev.sum()
    # 10. ridge lead-lag IC (reference edge) — cheap OLS VAR(1)
    X = A[:, :-1].T; Y = A[:, 1:].T; Xc = X-X.mean(0); Yc = Y-Y.mean(0)
    B = np.linalg.lstsq(Xc, Yc, rcond=None)[0]
    h = len(Xc)//2
    pred = (X[h:]-X[:h].mean(0)) @ np.linalg.lstsq(Xc[:h], Yc[:h], rcond=None)[0]
    ics = [np.corrcoef(pred[k], Yc[h:][k])[0, 1] for k in range(len(pred)) if Yc[h:][k].std() > 0]
    d["leadlag_IC"] = np.mean(ics)
    return d


early = diagnostics(win_ret(250, 500))
recent = diagnostics(win_ret(500, nt-1))
print(f"{'diagnostic':18} {'EARLY 250-500':>14} {'RECENT 500-750':>16}  {'flag':>6}")
for k in early:
    e, r = early[k], recent[k]
    # flag if recent is materially different (esp. a new significant edge)
    flag = ""
    if "_t" in k and abs(r) > 2 and abs(r) > abs(e) + 1.5: flag = "*** NEW"
    elif k in ("VR5", "factor1_share", "excess_kurt") and abs(r-e) > 0.15*abs(e)+0.05: flag = "shift"
    elif "IC" in k and abs(r) > abs(e) + 0.02 and abs(r) > 0.03: flag = "stronger"
    print(f"{k:18} {e:14.4f} {r:16.4f}  {flag:>10}")
print("\n*** NEW / stronger = structure that emerged in 500-750 (a switchable edge). blank = no change.")
